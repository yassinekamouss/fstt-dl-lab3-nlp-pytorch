"""Arabic NLP preprocessing pipeline, score discretization, and anti-leakage splitting."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z0-9#]+;")
_BRACKET_RE = re.compile(r"\[[^\]]{1,40}\]")
_TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL_RE = re.compile(r"\u0640")
_ALEF_RE = re.compile(r"[أإآٱ]")
_NON_ARABIC_LETTERS_RE = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")
_ARABIC_TOKEN_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A]{2,}")
_MULTI_SPACE_RE = re.compile(r"\s+")

_WIKI_ARABIC_STOPWORDS: tuple[str, ...] = (
    "انظر",
    "ايضا",
    "مراجع",
    "وصلات",
    "خارجية",
    "مصدر",
    "المصدر",
    "حيث",
    "وذلك",
    "اي",
    "بينما",
    "خلال",
    "يمكن",
    "تعد",
    "يعد",
    "تم",
    "تمت",
    "عام",
    "سنة",
    "وهو",
    "وهي",
    "التي",
    "الذي",
    "الذين",
    "اللذين",
    "اللتين",
    "بشكل",
    "بسبب",
    "بواسطة",
    "منها",
    "منهم",
    "عليها",
    "عليه",
    "فيها",
    "فيه",
    "اليه",
    "اليها",
    "وقد",
    "فقد",
    "لقد",
    "كما",
    "كذلك",
    "اكثر",
    "اقل",
    "احد",
    "احدى",
    "غير",
    "دون",
    "نحو",
    "حوالي",
    "قرابة",
    "بعض",
    "جميع",
    "كلها",
    "كله",
    "تلك",
    "ذلك",
    "هذا",
    "هذه",
    "هؤلاء",
    "اولئك",
    "كان",
    "كانت",
    "يكون",
    "تكون",
    "اصبح",
    "اصبحت",
    "اصبحوا",
    "تعتبر",
    "يعتبر",
    "تسمى",
    "يسمى",
    "الى",
    "على",
    "من",
    "في",
    "عن",
    "مع",
    "او",
    "ثم",
    "لكن",
    "بل",
    "حتى",
    "منذ",
    "بعد",
    "قبل",
    "عند",
    "بين",
    "ضمن",
    "حول",
    "ضد",
    "نفس",
    "مثل",
)


class ArabicTextPreprocessor:
    """End-to-end Arabic NLP preprocessor (normalization, tokenization, stopwords, stem/lemma)."""

    def __init__(
        self,
        normalize_teh_marbuta: bool = False,
        extra_stopwords: Iterable[str] | None = None,
    ) -> None:
        self.normalize_teh_marbuta = normalize_teh_marbuta
        self._araby = self._load_pyarabic()
        self._isri_stemmer = None
        self._snowball_stemmer = None
        self._qalsadi_lemmatizer = None
        self._lemma_cache: dict[str, str] = {}
        self.stopwords: set[str] = self.load_arabic_stopwords(extra_stopwords=extra_stopwords)

    @staticmethod
    def _load_pyarabic() -> Any:
        try:
            import pyarabic.araby as araby

            return araby
        except ImportError:
            return None

    def normalize_arabic(self, text: str) -> str:
        """Clean and normalize Arabic text (remove diacritics, tatweel, Latin, unify Alef/Ya)."""
        if not isinstance(text, str):
            return ""
        cleaned = _URL_RE.sub(" ", text)
        cleaned = _HTML_ENTITY_RE.sub(" ", cleaned)
        cleaned = _BRACKET_RE.sub(" ", cleaned)

        if self._araby is not None:
            cleaned = self._araby.strip_tashkeel(cleaned)
            cleaned = self._araby.strip_tatweel(cleaned)
        else:
            cleaned = _TASHKEEL_RE.sub("", cleaned)
            cleaned = _TATWEEL_RE.sub("", cleaned)

        cleaned = _ALEF_RE.sub("ا", cleaned)
        cleaned = cleaned.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
        if self.normalize_teh_marbuta:
            cleaned = cleaned.replace("ة", "ه")

        cleaned = _NON_ARABIC_LETTERS_RE.sub(" ", cleaned)
        return _MULTI_SPACE_RE.sub(" ", cleaned).strip()

    def tokenize(self, text: str, min_token_len: int = 2) -> list[str]:
        """Tokenize normalized Arabic text into words."""
        if not text:
            return []
        if self._araby is not None:
            raw_tokens = self._araby.tokenize(text)
            tokens = [
                tok
                for tok in raw_tokens
                if len(tok) >= min_token_len and _ARABIC_TOKEN_RE.fullmatch(tok) is not None
            ]
            return tokens
        return [tok for tok in _ARABIC_TOKEN_RE.findall(text) if len(tok) >= min_token_len]

    def load_arabic_stopwords(self, extra_stopwords: Iterable[str] | None = None) -> set[str]:
        """Load NLTK Arabic stopwords + Wikipedia domain stopwords, all normalized."""
        raw_stops: list[str] = list(_WIKI_ARABIC_STOPWORDS)
        try:
            import nltk
            from nltk.corpus import stopwords as nltk_stopwords

            try:
                raw_stops.extend(nltk_stopwords.words("arabic"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                raw_stops.extend(nltk_stopwords.words("arabic"))
        except Exception as exc:
            logger.warning("Could not load NLTK Arabic stopwords (%s); using built-in list.", exc)

        if extra_stopwords is not None:
            raw_stops.extend(extra_stopwords)

        normalized_stops: set[str] = set()
        for word in raw_stops:
            norm = self.normalize_arabic(str(word))
            if norm:
                for tok in norm.split():
                    normalized_stops.add(tok)
        return normalized_stops

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Filter out normalized Arabic stopwords from a token list."""
        return [tok for tok in tokens if tok not in self.stopwords]

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Apply NLTK ISRIStemmer to Arabic tokens."""
        if not tokens:
            return []
        if self._isri_stemmer is None:
            try:
                from nltk.stem.isri import ISRIStemmer

                self._isri_stemmer = ISRIStemmer()
            except ImportError:
                logger.warning("NLTK ISRIStemmer unavailable; returning original tokens.")
                return list(tokens)
        return [self._isri_stemmer.stem(tok) or tok for tok in tokens]

    def snowball_stem_tokens(self, tokens: list[str]) -> list[str]:
        """Apply NLTK Arabic SnowballStemmer for comparative analysis."""
        if not tokens:
            return []
        if self._snowball_stemmer is None:
            try:
                from nltk.stem.snowball import SnowballStemmer

                self._snowball_stemmer = SnowballStemmer("arabic")
            except ImportError:
                logger.warning("NLTK SnowballStemmer unavailable; returning original tokens.")
                return list(tokens)
        return [self._snowball_stemmer.stem(tok) or tok for tok in tokens]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Lemmatize Arabic tokens with Qalsadi, caching lookups and normalizing output."""
        if not tokens:
            return []
        if self._qalsadi_lemmatizer is None:
            try:
                import qalsadi.lemmatizer

                self._qalsadi_lemmatizer = qalsadi.lemmatizer.Lemmatizer()
            except ImportError:
                logger.warning("Qalsadi unavailable; falling back to ISRI stemming.")
                return self.stem_tokens(tokens)

        lemmas: list[str] = []
        for tok in tokens:
            cached = self._lemma_cache.get(tok)
            if cached is not None:
                lemmas.append(cached)
                continue

            try:
                raw_lemma = self._qalsadi_lemmatizer.lemmatize(tok)
                if isinstance(raw_lemma, (list, tuple)):
                    raw_lemma = raw_lemma[0] if raw_lemma else tok
                norm_lemma = self.normalize_arabic(str(raw_lemma))
                # Qalsadi occasionally returns multi-word or empty forms; fall back cleanly.
                lemma = norm_lemma.split()[0] if norm_lemma else tok
            except Exception:
                lemma = tok

            self._lemma_cache[tok] = lemma
            lemmas.append(lemma)
        return lemmas

    def process_text(self, text: str) -> dict[str, Any]:
        """Run all preprocessing stages on a single Arabic text sample."""
        text_clean = self.normalize_arabic(text)
        tokens_clean = self.tokenize(text_clean)
        tokens_nostop = self.remove_stopwords(tokens_clean)
        tokens_stemmed = self.stem_tokens(tokens_nostop)
        tokens_lemmatized = self.lemmatize_tokens(tokens_nostop)

        return {
            "text_clean": text_clean,
            "tokens_nostop": tokens_nostop,
            "text_nostop": " ".join(tokens_nostop),
            "text_stemmed": " ".join(tokens_stemmed),
            "text_lemmatized": " ".join(tokens_lemmatized),
            "token_count": len(tokens_lemmatized),
        }

    def process_dataframe(
        self,
        df: Any,
        text_col: str = "text",
        min_tokens: int = 5,
        progress: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """Apply full Arabic preprocessing to a DataFrame and collect vocabulary stats."""
        import pandas as pd

        iterator: Iterable[str] = df[text_col].astype(str).tolist()
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, desc="Arabic NLP Preprocessing")
            except ImportError:
                pass

        processed_rows = [self.process_text(text) for text in iterator]
        proc_df = pd.DataFrame(processed_rows)

        out = df.reset_index(drop=True).copy()
        for col in ("text_clean", "text_nostop", "text_stemmed", "text_lemmatized", "token_count"):
            out[col] = proc_df[col]

        raw_vocab = set(tok for txt in out[text_col].astype(str) for tok in txt.split())
        clean_vocab = set(tok for txt in out["text_clean"] for tok in txt.split())
        nostop_vocab = set(tok for txt in out["text_nostop"] for tok in txt.split())
        stemmed_vocab = set(tok for txt in out["text_stemmed"] for tok in txt.split())
        lemma_vocab = set(tok for txt in out["text_lemmatized"] for tok in txt.split())

        before_count = len(out)
        out = out[out["token_count"] >= min_tokens].reset_index(drop=True)
        after_count = len(out)

        stats: dict[str, Any] = {
            "rows_before_filter": before_count,
            "rows_after_filter": after_count,
            "rows_dropped": before_count - after_count,
            "vocab_raw": len(raw_vocab),
            "vocab_clean": len(clean_vocab),
            "vocab_nostop": len(nostop_vocab),
            "vocab_stemmed": len(stemmed_vocab),
            "vocab_lemmatized": len(lemma_vocab),
        }
        return out, stats


def discretize_scores(
    scores: Iterable[float],
    bins: tuple[float, ...] | list[float] = (0.0, 4.0, 7.0, 10.01),
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Discretize continuous relevance scores in [0, 10] into integer class indices."""
    arr = np.asarray(list(scores), dtype=float)
    boundaries = np.asarray(bins, dtype=float)
    if len(boundaries) < 3:
        raise ValueError("bins must define at least two intervals (length >= 3).")

    internal_edges = boundaries[1:-1]
    clipped = np.clip(arr, boundaries[0], boundaries[-1] - 1e-6)
    labels = np.searchsorted(internal_edges, clipped, side="right").astype(np.int64)

    n_classes = len(boundaries) - 1
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError(f"Expected {n_classes} class names, got {len(class_names)}.")

    names = [class_names[int(lbl)] for lbl in labels]
    return labels, names


def stratified_group_split(
    df: Any,
    label_col: str = "label",
    group_col: str = "title",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Split a DataFrame into train/val/test preserving class balance and group boundaries.

    Ensures paragraphs from the same Wikipedia article never cross between splits,
    preventing article-level data leakage.
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be positive and sum to < 1.0.")

    test_ratio = 1.0 - train_ratio - val_ratio
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)

    labels = sorted(df[label_col].unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n_classes = len(labels)

    total_per_class = np.zeros(n_classes, dtype=float)
    for lbl, count in df[label_col].value_counts().items():
        total_per_class[label_to_idx[lbl]] = float(count)

    # Target class counts for each split (3 x n_classes)
    target_counts = np.outer(ratios, total_per_class)

    groups: list[tuple[str, int, np.ndarray]] = []
    for group_name, grp_df in df.groupby(group_col, sort=False):
        counts = np.zeros(n_classes, dtype=float)
        for lbl, c in grp_df[label_col].value_counts().items():
            counts[label_to_idx[lbl]] = float(c)
        groups.append((str(group_name), len(grp_df), counts))

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    # Assign largest groups first for smoother class-balance optimization.
    groups.sort(key=lambda item: item[1], reverse=True)

    split_counts = np.zeros((3, n_classes), dtype=float)
    split_groups: list[list[str]] = [[], [], []]

    for group_name, _, grp_vec in groups:
        best_split = 0
        best_cost = float("inf")
        for s_idx in range(3):
            candidate = split_counts.copy()
            candidate[s_idx] += grp_vec
            # Relative squared deviation from target counts across splits and classes
            denom = np.maximum(target_counts, 1.0)
            cost = float(np.sum(((candidate - target_counts) / denom) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_split = s_idx
        split_counts[best_split] += grp_vec
        split_groups[best_split].append(group_name)

    train_set = set(split_groups[0])
    val_set = set(split_groups[1])
    test_set = set(split_groups[2])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    group_series = df[group_col].astype(str)
    train_df = df[group_series.isin(train_set)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df[group_series.isin(val_set)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = df[group_series.isin(test_set)].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df
