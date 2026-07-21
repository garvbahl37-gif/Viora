"""Hierarchical temporal encoder.

Operates on per-frame features ``[B, T', D]`` and models time at two scales:

* **local** layers with a banded attention window (radius ``local_window``) learn
  short motion / events without paying full ``O(T'^2)`` cost;
* **global** layers with full temporal attention relate distant clips for
  long-range structure.

The first ``depth - global_layers`` blocks are local; the rest are global. All
blocks respect a per-video ``temporal_mask[B, T']`` so padded batches are safe.
"""

from __future__ import annotations

import torch
from torch import nn

from viora.models.common import TransformerBlock, make_norm
from viora.utils.config import TemporalConfig


def local_band_mask(length: int, radius: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[L, L]`` where position ``i`` may attend to ``j`` iff ``|i-j| < radius``."""
    idx = torch.arange(length, device=device)
    return (idx[None, :] - idx[:, None]).abs() < radius


class HierarchicalTemporalEncoder(nn.Module):
    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = cfg.gradient_checkpointing
        n_global = min(cfg.global_layers, cfg.depth)
        n_local = cfg.depth - n_global
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()

        self.is_local: list[bool] = [True] * n_local + [False] * n_global
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.dim, cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio, norm_type=cfg.norm_type, mlp_type=cfg.mlp_type,
                drop=cfg.drop_rate, drop_path=dpr[i], use_sdpa=cfg.use_sdpa,
            )
            for i in range(cfg.depth)
        )
        self.norm = make_norm(cfg.norm_type, cfg.dim)

    def forward(
        self, x: torch.Tensor, temporal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x``: ``[B, T', D]`` -> ``[B, T', D]`` (contextualized across time)."""
        length = x.shape[1]
        band = local_band_mask(length, self.cfg.local_window, x.device)
        for block, is_local in zip(self.blocks, self.is_local, strict=True):
            mask = band if is_local else None
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, None, temporal_mask, None, mask, use_reentrant=False
                )
            else:
                x = block(x, self_key_padding_mask=temporal_mask, self_attn_mask=mask)
        return self.norm(x)
