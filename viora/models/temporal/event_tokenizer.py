"""Event tokenizer.

A small set of learnable **event queries** cross-attend to the temporal feature
sequence and are refined by self-attention, compressing dense per-frame features
``[B, T', D]`` into ``E`` semantic **event tokens** ``[B, E, D]``. The final
cross-attention map is returned for interpretability (which moments each event
attends to) — deliberately *not* claimed to be exact event boundaries, since that
would require supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from viora.models.common import TransformerBlock, make_norm
from viora.utils.config import EventConfig


@dataclass
class EventOutput:
    event_tokens: torch.Tensor          # [B, E, D]
    event_attention: torch.Tensor       # [B, E, T'] last-layer cross-attention (query->time)
    temporal_relevance: torch.Tensor    # [B, T'] attention mass each moment received


class EventTokenizer(nn.Module):
    def __init__(self, cfg: EventConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.queries = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.dim, cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                norm_type=cfg.norm_type, use_sdpa=cfg.use_sdpa, cross_attention=True,
            )
            for _ in range(cfg.num_layers)
        )
        self.norm = make_norm(cfg.norm_type, cfg.dim)

    def forward(
        self, temporal_features: torch.Tensor, temporal_mask: torch.Tensor | None = None
    ) -> EventOutput:
        """``temporal_features``: ``[B, T', D]`` -> :class:`EventOutput`."""
        b = temporal_features.shape[0]
        q = self.queries.expand(b, -1, -1)  # [B, E, D]
        weights = None
        for i, block in enumerate(self.blocks):
            last = i == len(self.blocks) - 1
            out = block(
                q, context=temporal_features,
                context_key_padding_mask=temporal_mask,
                return_cross_weights=last,
            )
            q, w = out if last else (out, None)
            if w is not None:
                weights = w  # [B, E, T']
        q = self.norm(q)

        if weights is None:  # single-layer edge case with sdpa off-path unused
            weights = torch.zeros(b, self.cfg.num_queries, temporal_features.shape[1],
                                  device=q.device, dtype=q.dtype)
        relevance = weights.mean(dim=1)  # [B, T'] mean attention across event queries
        return EventOutput(event_tokens=q, event_attention=weights, temporal_relevance=relevance)
