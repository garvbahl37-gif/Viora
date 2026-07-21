"""Bounded, timestamp-aware temporal memory.

Enables long-video / streaming understanding without sending every frame to the
LLM. Three conceptual tiers within one hard token budget:

* **short-term**  — the most recent ``short_term_size`` event tokens, verbatim;
* **long-term**   — up to ``long_term_size`` slots holding older, *compressed*
  events (mean-merged or FIFO/importance-evicted);
* (**recent** is simply the newer portion of short-term.)

Every entry keeps its timestamp. A learnable importance head scores events (an
*uncalibrated* score until trained); deterministic FIFO and importance-aware
policies are provided as baselines. Update logic is non-differentiable selection
by design — gradients flow through the tokens and importance head, not the
discrete eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from viora.utils.config import MemoryConfig


@dataclass
class MemoryState:
    """Per-sequence memory. ``short_*`` are verbatim; ``long_*`` are compressed."""

    short_tokens: torch.Tensor   # [S, D]
    short_ts: torch.Tensor       # [S]
    short_imp: torch.Tensor      # [S]
    long_tokens: torch.Tensor    # [L, D]
    long_ts: torch.Tensor        # [L] representative (mean) time
    long_imp: torch.Tensor       # [L]
    long_counts: torch.Tensor    # [L] events merged into each slot

    @property
    def num_tokens(self) -> int:
        return int(self.short_tokens.shape[0] + self.long_tokens.shape[0])


class TemporalMemory(nn.Module):
    def __init__(self, cfg: MemoryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        budget = cfg.short_term_size + cfg.long_term_size
        if cfg.max_tokens < budget:
            raise ValueError(
                f"max_tokens={cfg.max_tokens} < short+long={budget}; raise the budget"
            )
        if cfg.eviction not in ("fifo", "importance"):
            raise ValueError(f"unknown eviction '{cfg.eviction}'")
        self.importance_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim // 2), nn.GELU(), nn.Linear(cfg.dim // 2, 1)
        )

    def score_importance(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[*, D] -> [*]`` in (0, 1). Uncalibrated until trained."""
        return torch.sigmoid(self.importance_head(tokens)).squeeze(-1)

    def new_state(self, device=None, dtype=torch.float32) -> MemoryState:
        d = self.cfg.dim
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731
        return MemoryState(z(0, d), z(0), z(0), z(0, d), z(0), z(0), z(0))

    # ---------------------------------------------------------------- update
    def update(
        self,
        state: MemoryState,
        events: torch.Tensor,           # [E, D]
        timestamps: torch.Tensor,       # [E]
        importance: torch.Tensor | None = None,  # [E]
    ) -> MemoryState:
        """Add events (in time order) and enforce the budget. Returns a new state."""
        if importance is None:
            importance = self.score_importance(events)

        short_tokens = torch.cat([state.short_tokens, events], dim=0)
        short_ts = torch.cat([state.short_ts, timestamps.to(state.short_ts.dtype)])
        short_imp = torch.cat([state.short_imp, importance.to(state.short_imp.dtype)])
        long = (state.long_tokens, state.long_ts, state.long_imp, state.long_counts)

        # overflow oldest short-term entries into long-term, preserving order
        overflow = short_tokens.shape[0] - self.cfg.short_term_size
        if overflow > 0:
            for i in range(overflow):
                long = self._push_long(long, short_tokens[i], short_ts[i], short_imp[i])
            short_tokens = short_tokens[overflow:]
            short_ts = short_ts[overflow:]
            short_imp = short_imp[overflow:]

        lt, lts, limp, lc = long
        return replace(
            state,
            short_tokens=short_tokens, short_ts=short_ts, short_imp=short_imp,
            long_tokens=lt, long_ts=lts, long_imp=limp, long_counts=lc,
        )

    def _push_long(self, long, tok, t, im):
        lt, lts, limp, lc = long
        tok = tok.unsqueeze(0)
        if lt.shape[0] < self.cfg.long_term_size:
            return (
                torch.cat([lt, tok], 0),
                torch.cat([lts, t.view(1)]),
                torch.cat([limp, im.view(1)]),
                torch.cat([lc, torch.ones(1, device=lc.device, dtype=lc.dtype)]),
            )
        # long-term full -> pick a victim slot
        victim = int(torch.argmin(limp)) if self.cfg.eviction == "importance" else 0

        if self.cfg.compression == "mean":
            c = lc[victim]
            lt, lts, limp, lc = lt.clone(), lts.clone(), limp.clone(), lc.clone()
            lt[victim] = (lt[victim] * c + tok.squeeze(0)) / (c + 1)
            lts[victim] = (lts[victim] * c + t) / (c + 1)
            limp[victim] = torch.maximum(limp[victim], im)
            lc[victim] = c + 1
            return (lt, lts, limp, lc)

        # compression == "drop": replace the victim with the new event
        keep = torch.ones(lt.shape[0], dtype=torch.bool, device=lt.device)
        keep[victim] = False
        return (
            torch.cat([lt[keep], tok], 0),
            torch.cat([lts[keep], t.view(1)]),
            torch.cat([limp[keep], im.view(1)]),
            torch.cat([lc[keep], torch.ones(1, device=lc.device, dtype=lc.dtype)]),
        )

    # ------------------------------------------------------------------ read
    def read(self, state: MemoryState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all memory tokens ``[M, D]`` and timestamps ``[M]`` (long then short)."""
        tokens = torch.cat([state.long_tokens, state.short_tokens], dim=0)
        ts = torch.cat([state.long_ts, state.short_ts], dim=0)
        return tokens, ts
