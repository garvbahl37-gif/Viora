"""Temporal attention — tokens attend across time at a fixed spatial location.

Reshape ``[B, N=t·s, D]`` -> ``[(B·s), t, D]`` so each of the ``s = h·w`` spatial
locations is an independent batch element whose ``t`` temporal tokens self-attend.
A per-video ``temporal_mask[B, t]`` (``True`` = valid frame) is expanded to a key
padding mask so padded frames in variable-length batches are ignored.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from viora.models.attention.attention_utils import MultiHeadAttention
from viora.models.embeddings.tubelet_embedding import TokenGrid


class TemporalAttention(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        grid: TokenGrid,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: ``[B, t·s, D]`` -> ``[B, t·s, D]``. ``temporal_mask``: ``[B, t]``."""
        t, s = grid.t, grid.num_spatial
        x = rearrange(x, "b (t s) d -> (b s) t d", t=t, s=s)

        kpm = None
        if temporal_mask is not None:
            # [B, t] -> [B, s, t] -> [(B·s), t]
            kpm = temporal_mask[:, None, :].expand(-1, s, -1).reshape(-1, t)

        x = self.attn(x, key_padding_mask=kpm)
        return rearrange(x, "(b s) t d -> b (t s) d", s=s)
