"""PyTorch recurrent sequence models (RNN, Bi-RNN, GRU, LSTM) for Part 1."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ArchType = Literal["rnn", "birnn", "gru", "lstm"]
PoolingType = Literal["last", "mean", "mean_max"]

SUPPORTED_ARCHS: tuple[str, ...] = ("rnn", "birnn", "gru", "lstm")
SUPPORTED_POOLING: tuple[str, ...] = ("last", "mean", "mean_max")


def _create_recurrent_core(
    arch: str,
    input_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> tuple[nn.Module, int, bool]:
    """Instantiate a recurrent backbone and return (module, output_dim, bidirectional)."""
    arch_norm = arch.lower()
    if arch_norm not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}.")

    rec_dropout = dropout if num_layers > 1 else 0.0
    bidirectional = arch_norm == "birnn"

    if arch_norm in ("rnn", "birnn"):
        core: nn.Module = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rec_dropout,
        )
    elif arch_norm == "gru":
        core = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=rec_dropout,
        )
    else:
        core = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=rec_dropout,
        )

    out_dim = hidden_dim * (2 if bidirectional else 1)
    return core, out_dim, bidirectional


class SequenceClassifier(nn.Module):
    """Sequence classifier supporting RNN, Bidirectional RNN, GRU, and LSTM."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        arch: ArchType = "lstm",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: PoolingType = "mean_max",
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in SUPPORTED_POOLING:
            raise ValueError(f"Unsupported pooling '{pooling}'. Choose from {SUPPORTED_POOLING}.")

        self.arch = arch.lower()
        self.pooling = pooling
        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn, rec_out_dim, self.bidirectional = _create_recurrent_core(
            arch=self.arch,
            input_size=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        feat_dim = rec_out_dim * (2 if pooling == "mean_max" else 1)
        self.norm = nn.LayerNorm(feat_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

    def _pool(
        self,
        outputs: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "last":
            h_n = hidden[0] if isinstance(hidden, tuple) else hidden
            if self.bidirectional:
                return torch.cat([h_n[-2], h_n[-1]], dim=-1)
            return h_n[-1]

        seq_len = outputs.size(1)
        valid_steps = lengths.clamp(min=1, max=seq_len).to(outputs.device)
        mask_bool = torch.arange(seq_len, device=outputs.device).unsqueeze(0) < valid_steps.unsqueeze(1)
        mask_f = mask_bool.unsqueeze(-1).type_as(outputs)

        mean_pool = (outputs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        if self.pooling == "mean":
            return mean_pool

        max_masked = outputs.masked_fill(~mask_bool.unsqueeze(-1), -1e9)
        max_pool = max_masked.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute class logits for variable-length padded sequences."""
        embedded = self.embed_dropout(self.embedding(input_ids))
        lengths_cpu = lengths.detach().cpu().clamp(min=1, max=input_ids.size(1))
        packed = pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        pooled = self._pool(outputs, hidden, lengths)
        features = self.head_dropout(self.norm(pooled))
        return self.fc(features)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SequenceLanguageModel(nn.Module):
    """Word-level recurrent language model across RNN, Bi-RNN (cloze), GRU, and LSTM."""

    def __init__(
        self,
        vocab_size: int,
        arch: ArchType = "lstm",
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.arch = arch.lower()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn, rec_out_dim, self.bidirectional = _create_recurrent_core(
            arch=self.arch,
            input_size=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.proj = nn.Linear(rec_out_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token logits of shape (batch, seq_len, vocab_size)."""
        embedded = self.dropout(self.embedding(input_ids))
        if lengths is not None:
            lengths_cpu = lengths.detach().cpu().clamp(min=1, max=input_ids.size(1))
            packed = pack_padded_sequence(
                embedded,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            outputs, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=input_ids.size(1),
            )
        else:
            outputs, _ = self.rnn(embedded)
        return self.proj(self.dropout(outputs))

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: list[int],
        max_new_tokens: int,
        eos_id: int,
        pad_id: int = 0,
        unk_id: int = 1,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> list[int]:
        """Autoregressively generate token IDs for causal recurrent models."""
        if self.bidirectional:
            raise ValueError("Autoregressive generation is only valid for causal models (rnn, gru, lstm).")

        self.eval()
        device = next(self.parameters()).device
        generated = list(prefix_ids)

        for _ in range(max_new_tokens):
            inp = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.forward(inp)[:, -1, :] / max(temperature, 1e-5)
            logits[:, pad_id] = -float("inf")
            logits[:, unk_id] = -float("inf")

            if top_k > 0 and top_k < logits.size(-1):
                top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_vals, dim=-1)
                sampled_rel = torch.multinomial(probs, num_samples=1)
                next_id = int(top_idx.gather(-1, sampled_rel).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id == eos_id:
                break
            generated.append(next_id)

        return generated

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_classifier(
    arch: str,
    vocab_size: int,
    num_classes: int,
    **kwargs: object,
) -> SequenceClassifier:
    """Factory helper for SequenceClassifier."""
    return SequenceClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        arch=arch,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def build_language_model(
    arch: str,
    vocab_size: int,
    **kwargs: object,
) -> SequenceLanguageModel:
    """Factory helper for SequenceLanguageModel."""
    return SequenceLanguageModel(
        vocab_size=vocab_size,
        arch=arch,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
