"""Spatial attention — tokens attend within their own temporal slice.

Reshape ``[B, N=t·s, D]`` -> ``[(B·t), s, D]`` so each of the ``t`` frames is an
independent batch element whose ``s = h·w`` spatial tokens self-attend; then
reshape back. No mask needed: every spatial position within a frame is valid.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from viora.models.attention.attention_utils import MultiHeadAttention
from viora.models.embeddings.tubelet_embedding import TokenGrid


class SpatialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=proj_drop, use_sdpa=use_sdpa,
        )

    def forward(self, x: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
        """``x``: ``[B, t·s, D]`` -> ``[B, t·s, D]``."""
        t, s = grid.t, grid.num_spatial
        x = rearrange(x, "b (t s) d -> (b t) s d", t=t, s=s)
        x = self.attn(x)
        return rearrange(x, "(b t) s d -> b (t s) d", t=t)
