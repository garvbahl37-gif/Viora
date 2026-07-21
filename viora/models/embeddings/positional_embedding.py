"""Spatiotemporal positional embeddings.

Video needs *explicit* temporal structure, kept separable from spatial structure
so each can be learned, fixed (sinc-cos), or timestamp-derived independently.

Given tokens ``[B, N, D]`` laid out by a :class:`TokenGrid` ``(t, h, w)`` we add:

* a **spatial** embedding shared across time    — shape ``[h·w, D]`` broadcast over ``t``
* a **temporal** embedding shared across space   — shape ``[t, D]``    broadcast over ``h·w``

The temporal embedding can instead be derived from real per-frame **timestamps**
(``[B, t]`` seconds), which makes variable frame spacing representable — important
for adaptive sampling and long videos.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn

from viora.models.embeddings.tubelet_embedding import TokenGrid


def build_1d_sincos(length: int, dim: int, *, max_period: float = 10000.0) -> torch.Tensor:
    """Standard 1D sin-cos positional table, shape ``[length, dim]`` (dim even)."""
    if dim % 2:
        raise ValueError(f"sincos dim must be even, got {dim}")
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # [L,1]
    i = torch.arange(dim // 2, dtype=torch.float32)  # [D/2]
    div = torch.exp(-math.log(max_period) * (2 * i / dim))  # [D/2]
    angles = pos * div  # [L, D/2]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [L, D]


def sincos_from_timestamps(ts: torch.Tensor, dim: int, *, max_period: float = 10000.0) -> torch.Tensor:
    """1D sin-cos of continuous timestamps. ``ts`` ``[...,]`` -> ``[..., dim]``."""
    if dim % 2:
        raise ValueError(f"sincos dim must be even, got {dim}")
    i = torch.arange(dim // 2, dtype=torch.float32, device=ts.device)
    div = torch.exp(-math.log(max_period) * (2 * i / dim))  # [D/2]
    angles = ts.unsqueeze(-1) * div  # [..., D/2]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [..., D]


def build_2d_sincos(h: int, w: int, dim: int) -> torch.Tensor:
    """2D sin-cos over an ``h×w`` grid, shape ``[h·w, D]`` (dim divisible by 4 ideal)."""
    half = dim // 2
    # each axis gets half the channels; pad if odd via 1D fallback
    if half % 2:
        return build_1d_sincos(h * w, dim)
    eh = build_1d_sincos(h, half)  # [h, half]
    ew = build_1d_sincos(w, half)  # [w, half]
    grid = torch.cat(
        [
            eh.unsqueeze(1).expand(h, w, half),
            ew.unsqueeze(0).expand(h, w, half),
        ],
        dim=-1,
    )  # [h, w, D]
    return grid.reshape(h * w, dim)


class SpatioTemporalPositionalEmbedding(nn.Module):
    """Adds separable spatial + temporal positional information to video tokens."""

    def __init__(
        self,
        dim: int,
        grid: TokenGrid,
        *,
        spatial: str = "learnable",
        temporal: str = "learnable",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base_grid = grid
        self.spatial_kind = spatial
        self.temporal_kind = temporal
        self.drop = nn.Dropout(dropout)

        if spatial == "learnable":
            self.spatial_embed = nn.Parameter(torch.zeros(1, grid.num_spatial, dim))
            nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        elif spatial == "sincos":
            self.register_buffer("spatial_embed", build_2d_sincos(grid.h, grid.w, dim).unsqueeze(0))
        elif spatial == "none":
            self.spatial_embed = None
        else:
            raise ValueError(f"unknown spatial pos kind '{spatial}'")

        if temporal == "learnable":
            self.temporal_embed = nn.Parameter(torch.zeros(1, grid.t, dim))
            nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        elif temporal == "sincos":
            self.register_buffer("temporal_embed", build_1d_sincos(grid.t, dim).unsqueeze(0))
        elif temporal == "none":
            self.temporal_embed = None
        else:
            raise ValueError(f"unknown temporal pos kind '{temporal}'")

    def _spatial(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.spatial_embed is None:
            return None
        emb = self.spatial_embed  # [1, h0*w0, D]
        if h * w == emb.shape[1]:
            return emb.to(device=device, dtype=dtype)
        # interpolate spatial grid (bicubic) when the input grid differs from init.
        emb2d = rearrange(emb, "1 (h w) d -> 1 d h w", h=self.base_grid.h, w=self.base_grid.w)
        emb2d = nn.functional.interpolate(emb2d, size=(h, w), mode="bicubic", align_corners=False)
        return rearrange(emb2d, "1 d h w -> 1 (h w) d").to(device=device, dtype=dtype)

    def _temporal(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.temporal_embed is None:
            return None
        emb = self.temporal_embed  # [1, t0, D]
        if t == emb.shape[1]:
            return emb.to(device=device, dtype=dtype)
        emb1d = rearrange(emb, "1 t d -> 1 d t")
        emb1d = nn.functional.interpolate(emb1d, size=t, mode="linear", align_corners=False)
        return rearrange(emb1d, "1 d t -> 1 t d").to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        grid: TokenGrid,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add positional embeddings.

        Args:
            x: ``[B, N, D]`` tokens, ``N = t·h·w``.
            grid: layout of ``x``.
            timestamps: optional ``[B, t]`` seconds; when given, the temporal
                embedding is sin-cos of these values (overrides the learned/fixed
                table) so uneven frame spacing is represented.
        """
        b, n, d = x.shape
        t, h, w = grid.t, grid.h, grid.w
        if n != t * h * w:
            raise ValueError(f"token count {n} != t*h*w={t * h * w}")

        x = rearrange(x, "b (t s) d -> b t s d", t=t)  # [B, t, h*w, D]

        sp = self._spatial(h, w, x.device, x.dtype)
        if sp is not None:
            x = x + sp.unsqueeze(1)  # [1,1,h*w,D]

        if timestamps is not None:
            tmp = sincos_from_timestamps(timestamps.to(x.dtype), self.dim)  # [B, t, D]
            x = x + tmp.unsqueeze(2)  # [B,t,1,D]
        else:
            tmp = self._temporal(t, x.device, x.dtype)
            if tmp is not None:
                x = x + tmp.unsqueeze(2)  # [1,t,1,D]

        x = rearrange(x, "b t s d -> b (t s) d")
        return self.drop(x)
