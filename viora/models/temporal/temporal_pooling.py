"""Pooling utilities that bridge spatiotemporal tokens and temporal sequences.

* :class:`SpatialPool` collapses the ``h·w`` spatial tokens of each frame to one
  per-frame vector — ``[B, N, D] -> [B, T', D]`` — by mean or attention pooling.
* :func:`masked_mean` averages a temporal sequence honoring a validity mask.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from viora.models.attention.attention_utils import MultiHeadAttention
from viora.models.embeddings.tubelet_embedding import TokenGrid


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int = 1) -> torch.Tensor:
    """Mean over ``dim`` using ``mask`` (``True`` = valid); safe when a row is empty."""
    if mask is None:
        return x.mean(dim=dim)
    m = mask.unsqueeze(-1).to(x.dtype)  # [..., 1]
    total = (x * m).sum(dim=dim)
    count = m.sum(dim=dim).clamp_min(1.0)
    return total / count


class SpatialPool(nn.Module):
    """Collapse per-frame spatial tokens to a single vector per temporal position."""

    def __init__(self, dim: int, mode: str = "mean", num_heads: int = 8, use_sdpa: bool = True) -> None:
        super().__init__()
        if mode not in ("mean", "attention"):
            raise ValueError(f"unknown spatial pool mode '{mode}'")
        self.mode = mode
        if mode == "attention":
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.attn = MultiHeadAttention(dim, num_heads, use_sdpa=use_sdpa)

    def forward(self, tokens: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
        """``tokens``: ``[B, t·s, D]`` -> ``[B, t, D]``."""
        t = grid.t
        x = rearrange(tokens, "b (t s) d -> (b t) s d", t=t)
        if self.mode == "mean":
            pooled = x.mean(dim=1)  # [(b t), D]
        else:
            bt = x.shape[0]
            q = self.query.expand(bt, 1, -1)
            pooled = self.attn(q, context=x).squeeze(1)  # [(b t), D]
        return rearrange(pooled, "(b t) d -> b t d", t=t)
