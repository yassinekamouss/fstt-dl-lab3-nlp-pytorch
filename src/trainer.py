"""Training, EarlyStopping, Optuna hyperparameter tuning, and LM benchmarking for Part 1."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    BOS_IDX,
    EOS_IDX,
    MASK_IDX,
    PAD_IDX,
    ArabicClassificationDataset,
    ArabicLanguageModelDataset,
    MaskedLMCollator,
    Vocabulary,
    classification_collate_fn,
    lm_collate_fn,
)
from src.metrics import compute_bleu_scores
from src.models import SequenceLanguageModel, build_classifier, build_language_model

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored validation metric stops improving."""

    def __init__(self, mode: str = "max", patience: int = 5, min_delta: float = 1e-4) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.counter: int = 0

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Update state with current epoch metric; return True when training should stop."""
        if self._is_better(score):
            self.best_score = float(score)
            self.best_epoch = epoch
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(labels: Sequence[int] | np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights normalized to mean 1.0."""
    arr = np.asarray(labels, dtype=int)
    counts = np.bincount(arr, minlength=num_classes).astype(float)
    total = float(arr.shape[0])
    weights = np.ones(num_classes, dtype=float)
    for idx in range(num_classes):
        if counts[idx] > 0:
            weights[idx] = total / (num_classes * counts[idx])
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _evaluate_classifier_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for input_ids, lengths, labels in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            bsz = labels.size(0)
            total_loss += float(loss.item()) * bsz
            total_samples += bsz
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    acc = float(accuracy_score(all_targets, all_preds))
    f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, f1_macro


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 6,
    class_weights: torch.Tensor | None = None,
    checkpoint_path: str | Any | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a sequence classifier with AdamW, LR scheduling, clipping, and EarlyStopping."""
    model = model.to(device)
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    stopper = EarlyStopping(mode="max", patience=patience)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "lr": [],
    }

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bsz = labels.size(0)
            running_loss += float(loss.item()) * bsz
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += bsz

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_f1 = _evaluate_classifier_epoch(model, val_loader, criterion, device)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1)
        history["lr"].append(current_lr)

        scheduler.step(val_f1)
        should_stop = stopper.step(val_f1, model, epoch)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} macro_f1={val_f1:.3f}"
            )

        if should_stop:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {stopper.best_epoch}).")
            break

    elapsed = time.perf_counter() - start_time
    if stopper.best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in stopper.best_state_dict.items()})
        if checkpoint_path is not None:
            torch.save(stopper.best_state_dict, str(checkpoint_path))

    return {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_val_f1": float(stopper.best_score or 0.0),
        "train_time_sec": float(elapsed),
    }


def predict_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run batched inference and return targets, predictions, probabilities, and latency."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for input_ids, lengths, labels in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_prob = np.vstack(all_probs) if all_probs else np.empty((0, 0), dtype=float)
    return {
        "y_true": np.asarray(all_targets, dtype=int),
        "y_pred": np.asarray(all_preds, dtype=int),
        "y_prob": y_prob,
        "latency_ms_per_sample": elapsed_ms / max(len(all_targets), 1),
    }


def tune_architecture(
    arch: str,
    train_df: Any,
    val_df: Any,
    vocab: Vocabulary,
    cfg: dict[str, Any],
    device: torch.device,
    text_col: str = "text_lemmatized",
    label_col: str = "label",
) -> tuple[dict[str, Any], Any]:
    """Tune hyperparameters for one sequence architecture using Optuna (or grid fallback)."""
    import pandas as pd

    seed = int(cfg["project"]["seed"])
    max_seq_len = int(cfg["preprocessing"]["max_seq_len"])
    num_classes = len(cfg["preprocessing"]["class_names"])
    tune_cfg = cfg["part1_models"]["tuning"]
    space = tune_cfg["search_space"]
    n_trials = int(tune_cfg["n_trials_per_arch"])
    epochs = int(tune_cfg["epochs"])
    patience = int(tune_cfg["patience"])
    weight_decay = float(cfg["part1_models"]["final_training"]["weight_decay"])
    grad_clip = float(cfg["part1_models"]["final_training"]["grad_clip"])

    train_texts = train_df[text_col].astype(str).tolist()
    train_labels = train_df[label_col].astype(int).tolist()
    val_texts = val_df[text_col].astype(str).tolist()
    val_labels = val_df[label_col].astype(int).tolist()

    class_weights = compute_class_weights(train_labels, num_classes=num_classes)
    train_ds = ArabicClassificationDataset(train_texts, train_labels, vocab=vocab, max_len=max_seq_len)
    val_ds = ArabicClassificationDataset(val_texts, val_labels, vocab=vocab, max_len=max_seq_len)

    trial_records: list[dict[str, Any]] = []

    def _run_one(params: dict[str, Any], trial_id: int) -> float:
        torch.manual_seed(seed + trial_id)
        batch_size = int(params["batch_size"])
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=classification_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=classification_collate_fn,
        )
        model = build_classifier(
            arch=arch,
            vocab_size=len(vocab),
            num_classes=num_classes,
            embedding_dim=int(params["embedding_dim"]),
            hidden_dim=int(params["hidden_dim"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            pooling=str(params["pooling"]),
            pad_idx=PAD_IDX,
        )
        res = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=float(params["lr"]),
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            patience=patience,
            class_weights=class_weights,
            verbose=False,
        )
        row = {
            "arch": arch,
            "trial": trial_id,
            **params,
            "best_val_f1": res["best_val_f1"],
            "best_epoch": res["best_epoch"],
            "n_params": model.count_parameters(),
        }
        trial_records.append(row)
        return float(res["best_val_f1"])

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "embedding_dim": trial.suggest_categorical("embedding_dim", list(space["embedding_dim"])),
                "hidden_dim": trial.suggest_categorical("hidden_dim", list(space["hidden_dim"])),
                "num_layers": trial.suggest_categorical("num_layers", list(space["num_layers"])),
                "dropout": trial.suggest_float("dropout", float(space["dropout"][0]), float(space["dropout"][1]), step=0.1),
                "lr": trial.suggest_float("lr", float(space["lr"][0]), float(space["lr"][1]), log=True),
                "batch_size": trial.suggest_categorical("batch_size", list(space["batch_size"])),
                "pooling": trial.suggest_categorical("pooling", list(space["pooling"])),
            }
            return _run_one(params, trial.number)

        study.optimize(objective, n_trials=n_trials)
        best_params = dict(study.best_params)
    except ImportError:
        logger.warning("Optuna unavailable; falling back to structured grid search.")
        rng = np.random.default_rng(seed)
        best_score = -1.0
        best_params = {}
        for idx in range(n_trials):
            params = {
                "embedding_dim": int(rng.choice(space["embedding_dim"])),
                "hidden_dim": int(rng.choice(space["hidden_dim"])),
                "num_layers": int(rng.choice(space["num_layers"])),
                "dropout": float(round(rng.uniform(space["dropout"][0], space["dropout"][1]), 2)),
                "lr": float(rng.choice([5e-4, 1e-3, 2e-3])),
                "batch_size": int(rng.choice(space["batch_size"])),
                "pooling": str(rng.choice(space["pooling"])),
            }
            score = _run_one(params, idx)
            if score > best_score:
                best_score = score
                best_params = params

    trials_df = pd.DataFrame(trial_records).sort_values("best_val_f1", ascending=False).reset_index(drop=True)
    return best_params, trials_df


def train_and_evaluate_lm(
    arch: str,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    test_texts: Sequence[str],
    vocab: Vocabulary,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train & evaluate a word-level Language Model (Perplexity + BLEU) for Part 1.

    Causal models (rnn, gru, lstm) are evaluated autoregressively via next-token loss
    and prefix-to-continuation generation. Non-causal ``birnn`` is evaluated via a
    masked-cloze objective (pseudo-perplexity + masked token reconstruction BLEU).
    """
    lm_cfg = cfg["part1_models"]["lm_benchmark"]
    seed = int(cfg["project"]["seed"])
    epochs = int(lm_cfg["epochs"])
    batch_size = int(lm_cfg["batch_size"])
    max_len = int(lm_cfg["max_len"])
    prefix_len = int(lm_cfg["prefix_len"])
    gen_max_tokens = int(lm_cfg["gen_max_tokens"])
    mask_prob = float(lm_cfg["mask_prob"])

    torch.manual_seed(seed)
    train_ds = ArabicLanguageModelDataset(train_texts, vocab=vocab, max_len=max_len)
    val_ds = ArabicLanguageModelDataset(val_texts, vocab=vocab, max_len=max_len)
    test_ds = ArabicLanguageModelDataset(test_texts, vocab=vocab, max_len=max_len)

    model: SequenceLanguageModel = build_language_model(
        arch=arch,
        vocab_size=len(vocab),
        embedding_dim=int(lm_cfg["embedding_dim"]),
        hidden_dim=int(lm_cfg["hidden_dim"]),
        num_layers=int(lm_cfg["num_layers"]),
        dropout=float(lm_cfg["dropout"]),
        pad_idx=PAD_IDX,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lm_cfg["lr"]), weight_decay=1e-4)
    is_bidirectional = arch.lower() == "birnn"
    ignore_idx = -100 if is_bidirectional else PAD_IDX
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    stopper = EarlyStopping(mode="min", patience=3)

    if is_bidirectional:
        train_collator = MaskedLMCollator(mask_prob=mask_prob, seed=seed)
        eval_collator = MaskedLMCollator(mask_prob=mask_prob, seed=seed + 999)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_collator)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_collator)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lm_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lm_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lm_collate_fn)

    def _eval_lm_loss(loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in loader:
                if is_bidirectional:
                    inputs, targets, _, lengths = batch
                else:
                    inputs, targets, lengths = batch
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                logits = model(inputs, lengths)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                n_valid = int((targets != ignore_idx).sum().item())
                total_loss += float(loss.item()) * max(n_valid, 1)
                total_tokens += max(n_valid, 1)
        return total_loss / max(total_tokens, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if is_bidirectional:
                inputs, targets, _, lengths = batch
            else:
                inputs, targets, lengths = batch
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss = _eval_lm_loss(val_loader)
        if stopper.step(val_loss, model, epoch):
            break

    if stopper.best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in stopper.best_state_dict.items()})

    test_loss = _eval_lm_loss(test_loader)
    perplexity = float(math.exp(min(test_loss, 20.0)))

    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    samples: list[dict[str, str]] = []

    model.eval()
    if is_bidirectional:
        with torch.no_grad():
            for inputs, _, padded_gold, lengths in test_loader:
                inputs, lengths = inputs.to(device), lengths.to(device)
                logits = model(inputs, lengths)
                preds = logits.argmax(dim=-1).cpu()
                reconstructed = inputs.cpu().clone()
                mask_pos = reconstructed == MASK_IDX
                reconstructed[mask_pos] = preds[mask_pos]
                for row_idx in range(padded_gold.size(0)):
                    ref_txt = vocab.decode(padded_gold[row_idx].tolist(), skip_special=True)
                    hyp_txt = vocab.decode(reconstructed[row_idx].tolist(), skip_special=True)
                    ref_toks, hyp_toks = ref_txt.split(), hyp_txt.split()
                    if ref_toks and hyp_toks:
                        references.append(ref_toks)
                        hypotheses.append(hyp_toks)
                        if len(samples) < 3:
                            samples.append(
                                {
                                    "prompt_or_masked": vocab.decode(inputs[row_idx].cpu().tolist(), skip_special=False),
                                    "reference": ref_txt,
                                    "generated": hyp_txt,
                                }
                            )
                if len(references) >= 40:
                    break
        eval_mode = "masked_cloze (non-causal)"
    else:
        for seq in test_ds.sequences:
            if len(seq) < prefix_len + 4:
                continue
            prefix = seq[: prefix_len + 1].tolist()
            gold_cont_ids = seq[prefix_len + 1 :].tolist()
            gen_full_ids = model.generate(
                prefix_ids=prefix,
                max_new_tokens=min(gen_max_tokens, len(gold_cont_ids) + 4),
                eos_id=EOS_IDX,
                pad_id=PAD_IDX,
                temperature=0.8,
                top_k=20,
            )
            gen_cont_ids = gen_full_ids[len(prefix) :]
            ref_txt = vocab.decode(gold_cont_ids, skip_special=True)
            hyp_txt = vocab.decode(gen_cont_ids, skip_special=True)
            if ref_txt.split() and hyp_txt.split():
                references.append(ref_txt.split())
                hypotheses.append(hyp_txt.split())
                if len(samples) < 3:
                    samples.append(
                        {
                            "prompt_or_masked": vocab.decode(prefix, skip_special=True),
                            "reference": ref_txt,
                            "generated": hyp_txt,
                        }
                    )
            if len(references) >= 40:
                break
        eval_mode = "causal_autoregressive"

    bleu = compute_bleu_scores(references, hypotheses)
    return {
        "arch": arch,
        "evaluation_mode": eval_mode,
        "test_loss": float(test_loss),
        "perplexity": perplexity,
        **bleu,
        "n_eval_samples": len(references),
        "samples": samples,
    }
