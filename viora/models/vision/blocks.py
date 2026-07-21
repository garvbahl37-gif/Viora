"""Pre-norm spatiotemporal transformer block.

Supports two attention modes selected by config:

* ``"full"``       — one attention over all ``N = t·h·w`` tokens.
* ``"factorized"`` — spatial-then-temporal attention (see FactorizedAttention).

Both share the same feed-forward sublayer, stochastic depth, and an optional
gradient-checkpointing path for memory-heavy video training.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from torch import nn

from viora.models.attention.attention_utils import MultiHeadAttention
from viora.models.attention.factorized_attention import FactorizedAttention
from viora.models.common import DropPath, build_mlp, make_norm
from viora.models.embeddings.tubelet_embedding import TokenGrid


class SpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        attention_mode: str = "factorized",
        norm_type: str = "layernorm",
        mlp_type: str = "mlp",
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_sdpa: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if attention_mode not in ("full", "factorized"):
            raise ValueError(f"unknown attention_mode '{attention_mode}'")
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing

        if attention_mode == "full":
            self.norm1 = make_norm(norm_type, dim)
            self.attn = MultiHeadAttention(
                dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                proj_drop=drop, use_sdpa=use_sdpa,
            )
        else:
            self.attn = FactorizedAttention(
                dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                drop_path=drop_path, norm_type=norm_type, use_sdpa=use_sdpa,
            )

        self.norm2 = make_norm(norm_type, dim)
        self.mlp = build_mlp(mlp_type, dim, mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path)

    def _attn(
        self, x: torch.Tensor, grid: TokenGrid, temporal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.attention_mode == "full":
            kpm = None
            if temporal_mask is not None:
                # [B, t] -> [B, N] over t·s tokens
                kpm = temporal_mask[:, :, None].expand(-1, grid.t, grid.num_spatial)
                kpm = rearrange(kpm, "b t s -> b (t s)")
            return x + self.drop_path(self.attn(self.norm1(x), key_padding_mask=kpm))
        # factorized handles its own norms/residual/drop_path
        return self.attn(x, grid, temporal_mask=temporal_mask)

    def forward(
        self,
        x: torch.Tensor,
        grid: TokenGrid,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            x = checkpoint.checkpoint(self._attn, x, grid, temporal_mask, use_reentrant=False)
        else:
            x = self._attn(x, grid, temporal_mask)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
