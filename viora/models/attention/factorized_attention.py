"""Factorized spatiotemporal attention (ViViT-style "factorised self-attention").

Instead of one attention over all ``N = t·h·w`` tokens (cost ``O(N²)``), factorize
into spatial attention (``O(t·s²)``) followed by temporal attention (``O(s·t²)``).
For video ``s ≫ t``, this is dramatically cheaper while still mixing information
across the full spatiotemporal volume over two steps.

This module owns its two pre-norms, residual connections, and stochastic depth so
a transformer block can treat it as a single attention sublayer.
"""

from __future__ import annotations

import torch
from torch import nn

from viora.models.attention.spatial_attention import SpatialAttention
from viora.models.attention.temporal_attention import TemporalAttention
from viora.models.common import DropPath, make_norm
from viora.models.embeddings.tubelet_embedding import TokenGrid


class FactorizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_type: str = "layernorm",
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.norm_s = make_norm(norm_type, dim)
        self.spatial = SpatialAttention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=proj_drop, use_sdpa=use_sdpa,
        )
        self.norm_t = make_norm(norm_type, dim)
        self.temporal = TemporalAttention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=proj_drop, use_sdpa=use_sdpa,
        )
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        grid: TokenGrid,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.spatial(self.norm_s(x), grid))
        x = x + self.drop_path(self.temporal(self.norm_t(x), grid, temporal_mask=temporal_mask))
        return x
