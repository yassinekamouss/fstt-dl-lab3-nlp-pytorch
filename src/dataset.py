"""Vocabulary management and PyTorch Dataset / DataLoader utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
MASK_TOKEN = "<MASK>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
MASK_IDX = 4


class Vocabulary:
    """Word-level vocabulary built exclusively from the training corpus."""

    SPECIAL_TOKENS: tuple[str, ...] = (
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        MASK_TOKEN,
    )

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)
        }
        self.id_to_token: list[str] = list(self.SPECIAL_TOKENS)
        self.token_freqs: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.id_to_token)

    def fit(
        self,
        texts: Iterable[str],
        min_freq: int = 2,
        max_size: int = 20000,
    ) -> "Vocabulary":
        """Build vocabulary mappings from an iterable of tokenized whitespace strings."""
        counter: Counter[str] = Counter()
        for text in texts:
            if isinstance(text, str):
                counter.update(text.split())

        self.token_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_token = list(self.SPECIAL_TOKENS)
        self.token_freqs = {}

        eligible = [
            (tok, freq)
            for tok, freq in counter.items()
            if freq >= min_freq and tok not in self.token_to_id
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))

        capacity = max(0, max_size - len(self.id_to_token))
        for tok, freq in eligible[:capacity]:
            idx = len(self.id_to_token)
            self.token_to_id[tok] = idx
            self.id_to_token.append(tok)
            self.token_freqs[tok] = freq

        return self

    def encode(
        self,
        text: str,
        max_len: int | None = None,
        add_bos_eos: bool = False,
    ) -> list[int]:
        """Encode a whitespace-tokenized string into vocabulary IDs."""
        tokens = text.split() if isinstance(text, str) else []
        if add_bos_eos and max_len is not None and max_len >= 2:
            tokens = tokens[: max_len - 2]
        elif max_len is not None:
            tokens = tokens[:max_len]

        ids = [self.token_to_id.get(tok, UNK_IDX) for tok in tokens]
        if add_bos_eos:
            ids = [BOS_IDX] + ids + [EOS_IDX]
        if not ids:
            ids = [UNK_IDX]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Decode a sequence of token IDs back into a string."""
        special_set = set(self.SPECIAL_TOKENS)
        words: list[str] = []
        for idx in ids:
            if 0 <= idx < len(self.id_to_token):
                tok = self.id_to_token[idx]
            else:
                tok = UNK_TOKEN
            if skip_special and tok in special_set:
                continue
            words.append(tok)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        """Persist vocabulary mappings to UTF-8 JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id_to_token": self.id_to_token,
            "token_freqs": self.token_freqs,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a persisted Vocabulary from JSON."""
        in_path = Path(path)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        vocab = cls()
        vocab.id_to_token = list(payload["id_to_token"])
        vocab.token_to_id = {tok: idx for idx, tok in enumerate(vocab.id_to_token)}
        vocab.token_freqs = dict(payload.get("token_freqs", {}))
        return vocab


class ArabicClassificationDataset(Dataset):
    """PyTorch dataset for sequence relevance classification."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocabulary,
        max_len: int = 160,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        self.sequences: list[torch.Tensor] = [
            torch.tensor(vocab.encode(txt, max_len=max_len), dtype=torch.long)
            for txt in texts
        ]
        self.labels: list[int] = [int(lbl) for lbl in labels]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx]


def classification_collate_fn(
    batch: Sequence[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and return (padded_ids, lengths, labels)."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([max(1, int(seq.numel())) for seq in sequences], dtype=torch.long)
    padded_ids = pad_sequence(list(sequences), batch_first=True, padding_value=PAD_IDX)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_ids, lengths, label_tensor


class ArabicLanguageModelDataset(Dataset):
    """PyTorch dataset for causal and masked sequence language modeling benchmarks."""

    def __init__(
        self,
        texts: Sequence[str],
        vocab: Vocabulary,
        max_len: int = 48,
    ) -> None:
        self.sequences: list[torch.Tensor] = []
        for txt in texts:
            ids = vocab.encode(txt, max_len=max_len, add_bos_eos=True)
            if len(ids) >= 4:
                self.sequences.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def lm_collate_fn(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate causal next-token LM batches: input=seq[:-1], target=seq[1:]."""
    inputs = [seq[:-1] for seq in batch]
    targets = [seq[1:] for seq in batch]
    lengths = torch.tensor([max(1, int(seq.numel())) for seq in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    return padded_inputs, padded_targets, lengths


class MaskedLMCollator:
    """Collate masked-cloze batches for non-causal Bidirectional RNN evaluation."""

    def __init__(
        self,
        mask_prob: float = 0.15,
        pad_id: int = PAD_IDX,
        mask_id: int = MASK_IDX,
        special_ids: tuple[int, ...] = (PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, MASK_IDX),
        seed: int = 42,
    ) -> None:
        self.mask_prob = mask_prob
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.special_ids = set(special_ids)
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(
        self, batch: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = list(batch)
        lengths = torch.tensor([max(1, int(seq.numel())) for seq in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True, padding_value=self.pad_id)

        inputs = padded.clone()
        targets = torch.full_like(padded, fill_value=-100)

        eligible = torch.ones_like(padded, dtype=torch.bool)
        for sp_id in self.special_ids:
            eligible &= padded != sp_id

        rand_probs = torch.rand(padded.shape, generator=self.generator)
        mask_positions = eligible & (rand_probs < self.mask_prob)

        # Ensure at least one token per sequence is masked when eligible tokens exist.
        for row_idx in range(padded.shape[0]):
            if not mask_positions[row_idx].any():
                elig_indices = torch.nonzero(eligible[row_idx], as_tuple=False).flatten()
                if elig_indices.numel() > 0:
                    chosen = elig_indices[len(elig_indices) // 2]
                    mask_positions[row_idx, chosen] = True

        inputs[mask_positions] = self.mask_id
        targets[mask_positions] = padded[mask_positions]
        return inputs, targets, padded, lengths
