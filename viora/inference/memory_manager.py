"""Manage temporal memory + cosine retrieval for one video stream.

Wraps :class:`TemporalMemory` with a running state and a question-conditioned
retrieval over the stored tokens — the simple, replaceable cosine baseline the
long-video and streaming paths use to avoid feeding every token to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from viora.models.temporal.temporal_memory import MemoryState, TemporalMemory


@dataclass
class Retrieved:
    tokens: torch.Tensor       # [K, D]
    timestamps: torch.Tensor   # [K]
    scores: torch.Tensor       # [K] cosine similarity
    indices: torch.Tensor      # [K] positions into memory


class MemoryManager:
    def __init__(self, memory: TemporalMemory, device=None) -> None:
        self.memory = memory
        self.device = device
        self.state: MemoryState = memory.new_state(device=device)

    def reset(self) -> None:
        self.state = self.memory.new_state(device=self.device)

    def add_events(self, events: torch.Tensor, timestamps: torch.Tensor, importance=None) -> None:
        """Add ``[E, D]`` event tokens with ``[E]`` timestamps to memory."""
        self.state = self.memory.update(self.state, events, timestamps, importance=importance)

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> Retrieved:
        """Return the top-``k`` memory tokens most similar to ``query`` ``[D]``."""
        tokens, ts = self.memory.read(self.state)  # [M, D], [M]
        if tokens.shape[0] == 0:
            empty = tokens.new_zeros(0)
            return Retrieved(tokens, ts, empty, empty.long())
        sims = F.cosine_similarity(tokens, query.unsqueeze(0), dim=-1)  # [M]
        k = min(top_k, tokens.shape[0])
        scores, idx = sims.topk(k)
        return Retrieved(tokens[idx], ts[idx], scores, idx)

    def summary(self) -> dict[str, int]:
        return {
            "short_term": int(self.state.short_tokens.shape[0]),
            "long_term": int(self.state.long_tokens.shape[0]),
            "total": self.state.num_tokens,
            "budget": self.memory.cfg.max_tokens,
        }
