"""AraGPT2 loading, dataset tokenization, Trainer compatibility, and decoding utilities."""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from src.metrics import compute_bleu_scores, compute_rouge_and_diversity

logger = logging.getLogger(__name__)


def load_aragpt2(model_name: str = "aubmindlab/aragpt2-base") -> tuple[Any, Any, Any]:
    """Load AraGPT2 model, tokenizer, and optional AraBERT preprocessor safely."""
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    arabert_prep = None
    try:
        from arabert.preprocess import ArabertPreprocessor

        arabert_prep = ArabertPreprocessor(model_name=model_name)
    except Exception as exc:
        logger.warning("ArabertPreprocessor unavailable (%s); continuing without it.", exc)

    return model, tokenizer, arabert_prep


def build_lm_datasets(
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    test_texts: Sequence[str],
    tokenizer: Any,
    block_size: int = 256,
    arabert_prep: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Convert raw Arabic paragraphs into chunked causal LM HuggingFace Datasets."""
    from datasets import Dataset

    eos = tokenizer.eos_token or ""

    def _prepare_split(texts: Sequence[str]) -> Dataset:
        formatted = []
        for txt in texts:
            clean = str(txt).strip()
            if not clean:
                continue
            if arabert_prep is not None:
                clean = arabert_prep.preprocess(clean)
            formatted.append(f"{clean} {eos}".strip())

        raw_ds = Dataset.from_dict({"text": formatted})

        def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(batch["text"], truncation=False)

        tok_ds = raw_ds.map(_tokenize, batched=True, remove_columns=["text"])

        def _group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_len = len(concatenated["input_ids"])
            if total_len >= block_size:
                usable_len = (total_len // block_size) * block_size
                result = {
                    k: [t[i : i + block_size] for i in range(0, usable_len, block_size)]
                    for k, t in concatenated.items()
                }
            elif total_len > 0:
                result = {k: [t] for k, t in concatenated.items()}
            else:
                result = {k: [] for k in concatenated.keys()}
            result["labels"] = [list(ids) for ids in result["input_ids"]]
            return result

        return tok_ds.map(_group_texts, batched=True)

    return _prepare_split(train_texts), _prepare_split(val_texts), _prepare_split(test_texts)


def build_training_arguments(
    output_dir: str | Path,
    cfg: dict[str, Any],
    use_fp16: bool = True,
) -> Any:
    """Create TrainingArguments compatible across transformers v4 and v5 APIs."""
    from transformers import TrainingArguments

    gpt2_cfg = cfg["part2_gpt2"]
    seed = int(cfg["project"]["seed"])
    sig_params = inspect.signature(TrainingArguments.__init__).parameters

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": int(gpt2_cfg["epochs"]),
        "per_device_train_batch_size": int(gpt2_cfg["batch_size"]),
        "per_device_eval_batch_size": int(gpt2_cfg["batch_size"]),
        "gradient_accumulation_steps": int(gpt2_cfg["grad_accum_steps"]),
        "learning_rate": float(gpt2_cfg["lr"]),
        "weight_decay": float(gpt2_cfg["weight_decay"]),
        "warmup_ratio": float(gpt2_cfg["warmup_ratio"]),
        "lr_scheduler_type": "cosine",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "logging_steps": 20,
        "fp16": bool(use_fp16 and torch.cuda.is_available()),
        "seed": seed,
        "report_to": "none",
    }

    if "eval_strategy" in sig_params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig_params:
        kwargs["evaluation_strategy"] = "epoch"

    # Drop any argument unsupported by an older/newer version gracefully
    filtered = {k: v for k, v in kwargs.items() if k in sig_params}
    return TrainingArguments(**filtered)


def build_trainer(
    model: Any,
    args: Any,
    train_ds: Any,
    val_ds: Any,
    tokenizer: Any,
    data_collator: Any | None = None,
) -> Any:
    """Build Hugging Face Trainer safely handling tokenizer vs processing_class."""
    from transformers import DataCollatorForLanguageModeling, Trainer

    collator = data_collator or DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sig_params = inspect.signature(Trainer.__init__).parameters

    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    if "processing_class" in sig_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_params:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def evaluate_perplexity(trainer: Any, dataset: Any) -> float:
    """Evaluate causal LM loss and convert it to Perplexity."""
    metrics = trainer.evaluate(eval_dataset=dataset)
    loss = float(metrics["eval_loss"])
    return float(math.exp(min(loss, 20.0)))


def generate_paragraph(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    strategy: str = "nucleus",
    arabert_prep: Any | None = None,
    max_new_tokens: int = 150,
    min_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.35,
    no_repeat_ngram_size: int = 3,
    num_beams: int = 5,
) -> str:
    """Generate an Arabic paragraph from a seed sentence using greedy, beam, or nucleus decoding."""
    model.eval()
    prepared_prompt = arabert_prep.preprocess(prompt) if arabert_prep is not None else prompt
    encoded = tokenizer(prepared_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    common_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    strat = strategy.lower()
    if strat == "greedy":
        gen_kwargs = {**common_kwargs, "do_sample": False}
    elif strat == "beam":
        gen_kwargs = {
            **common_kwargs,
            "do_sample": False,
            "num_beams": num_beams,
            "early_stopping": True,
        }
    elif strat == "nucleus":
        gen_kwargs = {
            **common_kwargs,
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    else:
        raise ValueError(f"Unknown decoding strategy '{strategy}'. Choose greedy, beam, or nucleus.")

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if arabert_prep is not None:
        try:
            decoded = arabert_prep.unpreprocess(decoded)
        except Exception:
            pass
    return " ".join(decoded.split())


def evaluate_generation_quality(
    model: Any,
    tokenizer: Any,
    test_texts: Sequence[str],
    device: torch.device,
    arabert_prep: Any | None = None,
    strategy: str = "nucleus",
    n_samples: int = 25,
    prefix_words: int = 8,
    **gen_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate generated continuations against held-out Arabic Wikipedia paragraphs."""
    refs_tokens: list[list[str]] = []
    hyps_tokens: list[list[str]] = []
    refs_str: list[str] = []
    hyps_str: list[str] = []
    qualitative: list[dict[str, str]] = []

    for text in test_texts:
        words = str(text).split()
        if len(words) < prefix_words + 10:
            continue
        prompt = " ".join(words[:prefix_words])
        ref_cont = " ".join(words[prefix_words : prefix_words + 120])

        generated_full = generate_paragraph(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            strategy=strategy,
            arabert_prep=arabert_prep,
            **gen_kwargs,
        )
        # Strip the repeated prompt prefix so metrics measure newly generated continuation
        if generated_full.startswith(prompt):
            gen_cont = generated_full[len(prompt) :].strip()
        else:
            gen_words = generated_full.split()
            gen_cont = " ".join(gen_words[prefix_words:]) if len(gen_words) > prefix_words else generated_full

        if not gen_cont.split() or not ref_cont.split():
            continue

        refs_str.append(ref_cont)
        hyps_str.append(gen_cont)
        refs_tokens.append(ref_cont.split())
        hyps_tokens.append(gen_cont.split())

        if len(qualitative) < 3:
            qualitative.append(
                {
                    "prompt": prompt,
                    "reference": ref_cont,
                    "generated": gen_cont,
                }
            )

        if len(refs_str) >= n_samples:
            break

    bleu = compute_bleu_scores(refs_tokens, hyps_tokens)
    rouge_div = compute_rouge_and_diversity(refs_str, hyps_str)
    return {
        "strategy": strategy,
        "n_evaluated": len(refs_str),
        **bleu,
        **rouge_div,
        "samples": qualitative,
    }
