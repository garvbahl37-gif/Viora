"""Tubelet embedding — 3D patchification of a video into tokens.

A *tubelet* is a small spatiotemporal cube (``tubelet_size`` frames ×
``patch_size`` × ``patch_size`` pixels). A single strided ``Conv3d`` projects each
non-overlapping tubelet to an embedding vector, turning a video into a sequence of
tokens exactly as ViT patchifies an image — but with a temporal extent.

Shapes::

    input   x    : [B, C, T,  H,  W]
    conv    z    : [B, D, T', H', W']   T'=T/tt, H'=H/ps, W'=W/ps
    output  z    : [B, N, D]            N = T'·H'·W'
    grid         : TokenGrid(t=T', h=H', w=W')
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn


@dataclass(frozen=True)
class TokenGrid:
    """Spatiotemporal layout of a flattened ``[B, N, D]`` token sequence.

    Tokens are ordered ``(t, h, w)`` row-major, so ``index = ((ti*h)+hi)*w + wi``.
    Keeping this explicit lets factorized attention and positional embeddings
    reshape unambiguously.
    """

    t: int
    h: int
    w: int

    @property
    def num_tokens(self) -> int:
        return self.t * self.h * self.w

    @property
    def num_spatial(self) -> int:
        return self.h * self.w


class TubeletEmbedding(nn.Module):
    """Conv3D tubelet patchifier.

    Args:
        in_channels: input channels (3 for RGB).
        embed_dim: output token dimension ``D``.
        tubelet_size: temporal patch depth ``tt``.
        patch_size: spatial patch side ``ps`` (square).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        tubelet_size: int = 2,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if tubelet_size < 1 or patch_size < 1:
            raise ValueError("tubelet_size and patch_size must be >= 1")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        kernel = (tubelet_size, patch_size, patch_size)
        # stride == kernel -> non-overlapping tubelets (true patchification).
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel, stride=kernel)

    def grid_for(self, num_frames: int, height: int, width: int) -> TokenGrid:
        """Token grid produced for a given clip size (validates divisibility)."""
        if num_frames % self.tubelet_size:
            raise ValueError(
                f"num_frames={num_frames} not divisible by tubelet_size={self.tubelet_size}"
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"({height}x{width}) not divisible by patch_size={self.patch_size}"
            )
        return TokenGrid(
            t=num_frames // self.tubelet_size,
            h=height // self.patch_size,
            w=width // self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, TokenGrid]:
        """Patchify a video clip.

        Args:
            x: ``[B, C, T, H, W]``.

        Returns:
            tokens ``[B, N, D]`` and the :class:`TokenGrid` describing their layout.
        """
        if x.dim() != 5:
            raise ValueError(f"expected 5D [B,C,T,H,W], got shape {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {c}")
        grid = self.grid_for(t, h, w)  # validates divisibility

        z = self.proj(x)  # [B, D, T', H', W']
        z = rearrange(z, "b d t h w -> b (t h w) d")  # [B, N, D]
        return z, grid
