"""Evaluation metrics (Classification, Arabic-aware ROUGE/BLEU, Distinct) and plots."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

_PUNCT_STRIP_RE = re.compile(r"[.,;:!?؟،؛«»\"'()\[\]{}\-—_]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def format_arabic_for_plot(text: str) -> str:
    """Reshape and apply BiDi ordering so Arabic labels render cleanly in Matplotlib."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def _tokenize_for_nlg(text: str) -> list[str]:
    cleaned = _PUNCT_STRIP_RE.sub(" ", text)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned.split() if cleaned else []


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: np.ndarray | None,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute Accuracy, Precision, Recall, F1 (Macro & Weighted), ROC-AUC, and CM."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    roc_auc_macro: float | None = None
    roc_auc_weighted: float | None = None
    if y_prob is not None and len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc_macro = float(
                roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
            roc_auc_weighted = float(
                roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="weighted", labels=labels)
            )
        except Exception:
            roc_auc_macro = None
            roc_auc_weighted = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def compute_bleu_scores(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute NLTK corpus BLEU-1/2/4 (in [0, 1]) and SacreBLEU (in [0, 100])."""
    if not references or not hypotheses:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0, "sacrebleu": 0.0}

    ref_wrapped = [[list(ref)] for ref in references]
    hyp_list = [list(hyp) for hyp in hypotheses]

    bleu_1 = bleu_2 = bleu_4 = 0.0
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

        smooth = SmoothingFunction().method1
        bleu_1 = float(corpus_bleu(ref_wrapped, hyp_list, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth))
        bleu_2 = float(corpus_bleu(ref_wrapped, hyp_list, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth))
        bleu_4 = float(corpus_bleu(ref_wrapped, hyp_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
    except ImportError:
        pass

    sacre_score = 0.0
    try:
        import sacrebleu

        ref_strs = [" ".join(r) for r in references]
        hyp_strs = [" ".join(h) for h in hypotheses]
        sacre_score = float(sacrebleu.corpus_bleu(hyp_strs, [ref_strs]).score)
    except Exception:
        sacre_score = bleu_4 * 100.0

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4,
        "sacrebleu": sacre_score,
    }


def compute_rouge_and_diversity(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, float]:
    """Compute Arabic-safe ROUGE-1, ROUGE-2, ROUGE-L F1 scores and Distinct-1/2 diversity.

    Note: Python's default ``rouge_score`` tokenizer uses an ASCII regex ``[a-z0-9]+``
    that strips Arabic script completely. This implementation operates directly on
    normalized Unicode whitespace tokens so Arabic text is scored accurately.
    """
    if not references or not hypotheses:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
        }

    r1_scores: list[float] = []
    r2_scores: list[float] = []
    rl_scores: list[float] = []

    all_unigrams: list[tuple[str, ...]] = []
    all_bigrams: list[tuple[str, ...]] = []

    for ref_str, hyp_str in zip(references, hypotheses):
        ref_toks = _tokenize_for_nlg(ref_str)
        hyp_toks = _tokenize_for_nlg(hyp_str)

        all_unigrams.extend(_ngrams(hyp_toks, 1))
        all_bigrams.extend(_ngrams(hyp_toks, 2))

        for n, bucket in ((1, r1_scores), (2, r2_scores)):
            ref_ng = Counter(_ngrams(ref_toks, n))
            hyp_ng = Counter(_ngrams(hyp_toks, n))
            overlap = sum((ref_ng & hyp_ng).values())
            total_ref = sum(ref_ng.values())
            total_hyp = sum(hyp_ng.values())
            if overlap == 0 or total_ref == 0 or total_hyp == 0:
                bucket.append(0.0)
            else:
                prec = overlap / total_hyp
                rec = overlap / total_ref
                bucket.append(2.0 * prec * rec / (prec + rec))

        lcs = _lcs_length(ref_toks, hyp_toks)
        if lcs == 0 or not ref_toks or not hyp_toks:
            rl_scores.append(0.0)
        else:
            prec_l = lcs / len(hyp_toks)
            rec_l = lcs / len(ref_toks)
            rl_scores.append(2.0 * prec_l * rec_l / (prec_l + rec_l))

    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    return {
        "rouge1": float(np.mean(r1_scores)) if r1_scores else 0.0,
        "rouge2": float(np.mean(r2_scores)) if r2_scores else 0.0,
        "rougeL": float(np.mean(rl_scores)) if rl_scores else 0.0,
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
    }


def plot_eda_distributions(df: Any, save_dir: str | Path, class_names: Sequence[str]) -> Path:
    """Create and save a 2x2 EDA summary figure for the scraped & processed Arabic corpus."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "eda_distributions.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis — Arabic Wikipedia AI Corpus", fontsize=14, fontweight="bold")

    # 1. Continuous relevance score distribution
    ax = axes[0, 0]
    ax.hist(df["score"], bins=20, color="#2563eb", edgecolor="white", alpha=0.85)
    ax.axvline(4.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Threshold 4.0")
    ax.axvline(7.0, color="#16a34a", linestyle="--", linewidth=1.5, label="Threshold 7.0")
    ax.set_title("Distribution des Scores de Pertinence [0, 10]")
    ax.set_xlabel("Score")
    ax.set_ylabel("Nombre de paragraphes")
    ax.legend()

    # 2. Discretized class counts
    ax = axes[0, 1]
    counts = df["label"].value_counts().sort_index()
    bar_labels = [class_names[i] if i < len(class_names) else str(i) for i in counts.index]
    colors = ["#ef4444", "#f59e0b", "#10b981"][: len(counts)]
    bars = ax.bar(bar_labels, counts.values, color=colors, edgecolor="white")
    ax.set_title("Répartition des Classes Discrétisées")
    ax.set_ylabel("Nombre de paragraphes")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5, str(val), ha="center", va="bottom", fontweight="bold")

    # 3. Token count distribution
    ax = axes[1, 0]
    token_col = "token_count" if "token_count" in df.columns else "word_count"
    ax.hist(df[token_col], bins=25, color="#7c3aed", edgecolor="white", alpha=0.85)
    ax.set_title("Longueur des Paragraphes (après prétraitement)")
    ax.set_xlabel("Nombre de tokens")
    ax.set_ylabel("Fréquence")

    # 4. Top 15 most frequent Arabic tokens
    ax = axes[1, 1]
    text_col = "text_lemmatized" if "text_lemmatized" in df.columns else "text"
    freq = Counter(tok for txt in df[text_col].astype(str) for tok in txt.split())
    top_items = freq.most_common(15)[::-1]
    if top_items:
        words_display = [format_arabic_for_plot(w) for w, _ in top_items]
        vals = [c for _, c in top_items]
        ax.barh(words_display, vals, color="#0ea5e9", edgecolor="white")
    ax.set_title("Top 15 Lemmes Arabes les plus Fréquents")
    ax.set_xlabel("Occurrences")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_learning_curves(histories: dict[str, dict[str, Any]], save_path: str | Path) -> Path:
    """Plot Train/Val Loss and Validation Macro-F1 curves across the 4 Part 1 models."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"rnn": "#ef4444", "birnn": "#f59e0b", "gru": "#10b981", "lstm": "#2563eb"}

    for arch, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = colors.get(arch, None)
        axes[0].plot(epochs, hist["train_loss"], label=arch.upper(), color=color, linewidth=2)
        axes[1].plot(epochs, hist["val_loss"], label=arch.upper(), color=color, linewidth=2)
        axes[2].plot(epochs, hist["val_f1_macro"], label=arch.upper(), color=color, linewidth=2)

    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    axes[2].set_title("Validation Macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    plt.suptitle("Courbes d'Apprentissage — RNN vs Bi-RNN vs GRU vs LSTM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    results: dict[str, dict[str, Any]],
    class_names: Sequence[str],
    save_path: str | Path,
) -> Path:
    """Plot a 2x2 grid of confusion matrices for the 4 sequence architectures."""
    import seaborn as sns

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()

    for idx, (arch, res) in enumerate(results.items()):
        if idx >= 4:
            break
        cm = np.asarray(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes_flat[idx],
        )
        axes_flat[idx].set_title(
            f"{arch.upper()} — Acc: {res['accuracy']:.3f} | Macro-F1: {res['f1_macro']:.3f}"
        )
        axes_flat[idx].set_xlabel("Prédit")
        axes_flat[idx].set_ylabel("Réel")

    plt.suptitle("Matrices de Confusion (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(
    cls_df: Any,
    lm_df: Any | None,
    save_path: str | Path,
) -> Path:
    """Plot comparative bar charts for classification metrics and LM Perplexity/BLEU."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_panels = 2 if lm_df is not None and not lm_df.empty else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    ax0 = axes[0]
    archs = cls_df["arch"].str.upper().tolist()
    x = np.arange(len(archs))
    width = 0.25
    ax0.bar(x - width, cls_df["accuracy"], width, label="Accuracy", color="#2563eb")
    ax0.bar(x, cls_df["f1_macro"], width, label="Macro-F1", color="#10b981")
    ax0.bar(x + width, cls_df["f1_weighted"], width, label="Weighted-F1", color="#f59e0b")
    ax0.set_xticks(x)
    ax0.set_xticklabels(archs)
    ax0.set_ylim(0, 1.05)
    ax0.set_title("Comparaison Classification (Test Set)")
    ax0.set_ylabel("Score")
    ax0.grid(axis="y", linestyle="--", alpha=0.4)
    ax0.legend()

    if n_panels == 2 and lm_df is not None:
        ax1 = axes[1]
        lm_archs = lm_df["arch"].str.upper().tolist()
        x1 = np.arange(len(lm_archs))
        ax1.bar(x1 - 0.18, lm_df["bleu_1"], 0.35, label="BLEU-1", color="#0ea5e9")
        ax1.bar(x1 + 0.18, lm_df["bleu_4"], 0.35, label="BLEU-4", color="#7c3aed")
        ax1.set_xticks(x1)
        ax1.set_xticklabels(lm_archs)
        ax1.set_title("Benchmark Language Modeling — BLEU-1 & BLEU-4")
        ax1.set_ylabel("BLEU [0, 1]")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
