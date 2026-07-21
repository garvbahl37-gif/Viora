"""Perceiver-style multimodal resampler.

The LLM cannot consume thousands of video/event/memory tokens. A fixed set of
``num_queries`` learnable queries cross-attend to whatever visual tokens are
provided (spatiotemporal tokens, event tokens, memory — concatenated) and are
refined by self-attention, yielding a constant ``[B, Q, D]`` regardless of input
length. This decouples LLM cost from video length.
"""

from __future__ import annotations

import torch
from torch import nn

from viora.models.common import TransformerBlock, make_norm
from viora.utils.config import ResamplerConfig


class PerceiverResampler(nn.Module):
    def __init__(self, cfg: ResamplerConfig, input_dim: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim or cfg.dim
        self.queries = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        # project inputs to query dim if they differ
        self.input_proj = (
            nn.Linear(self.input_dim, cfg.dim) if self.input_dim != cfg.dim else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.dim, cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                norm_type=cfg.norm_type, use_sdpa=cfg.use_sdpa, cross_attention=True,
            )
            for _ in range(cfg.num_layers)
        )
        self.norm = make_norm(cfg.norm_type, cfg.dim)

    def forward(
        self, visual_tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``visual_tokens``: ``[B, M, input_dim]`` -> ``[B, Q, D]``."""
        b = visual_tokens.shape[0]
        ctx = self.input_proj(visual_tokens)
        q = self.queries.expand(b, -1, -1)
        for block in self.blocks:
            q = block(q, context=ctx, context_key_padding_mask=key_padding_mask)
        return self.norm(q)
