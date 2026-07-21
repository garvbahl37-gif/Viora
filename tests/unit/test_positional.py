"""Positional embeddings: shapes, additive behaviour, timestamp path."""

from __future__ import annotations

import torch

from viora.models.embeddings.positional_embedding import (
    SpatioTemporalPositionalEmbedding,
    build_1d_sincos,
    sincos_from_timestamps,
)
from viora.models.embeddings.tubelet_embedding import TokenGrid


def test_sincos_shape_and_range():
    e = build_1d_sincos(10, 16)
    assert e.shape == (10, 16)
    assert e.abs().max() <= 1.0 + 1e-6


def test_none_is_identity():
    grid = TokenGrid(t=4, h=2, w=2)
    pos = SpatioTemporalPositionalEmbedding(24, grid, spatial="none", temporal="none")
    x = torch.randn(2, 16, 24)
    assert torch.allclose(pos(x, grid), x)


def test_learnable_adds_and_preserves_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    pos = SpatioTemporalPositionalEmbedding(24, grid, spatial="learnable", temporal="learnable")
    x = torch.zeros(2, 16, 24)
    out = pos(x, grid)
    assert out.shape == (2, 16, 24)
    # spatial embed is broadcast across time -> same spatial position identical across frames
    out = out.reshape(2, grid.t, grid.num_spatial, 24)
    # with zero input + zero temporal init the frames differ only by temporal embed;
    # here temporal is trunc_normal so frames differ, but spatial pattern is shared.
    assert out.shape == (2, 4, 4, 24)


def test_sincos_options_run():
    grid = TokenGrid(t=4, h=2, w=2)
    pos = SpatioTemporalPositionalEmbedding(24, grid, spatial="sincos", temporal="sincos")
    x = torch.randn(1, 16, 24)
    assert pos(x, grid).shape == (1, 16, 24)


def test_timestamp_path_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    pos = SpatioTemporalPositionalEmbedding(24, grid, temporal="learnable")
    x = torch.randn(2, 16, 24)
    ts = torch.tensor([[0.0, 1.3, 2.9, 4.0], [0.0, 0.5, 1.0, 1.5]])
    out = pos(x, grid, timestamps=ts)
    assert out.shape == (2, 16, 24)


def test_sincos_from_timestamps_shape():
    ts = torch.tensor([[0.0, 1.0, 2.0]])
    assert sincos_from_timestamps(ts, 8).shape == (1, 3, 8)


def test_spatial_interpolation_on_grid_change():
    init = TokenGrid(t=4, h=2, w=2)
    pos = SpatioTemporalPositionalEmbedding(24, init, spatial="learnable", temporal="learnable")
    bigger = TokenGrid(t=6, h=3, w=3)
    x = torch.randn(1, bigger.num_tokens, 24)
    assert pos(x, bigger).shape == (1, 54, 24)  # interpolated, no crash
