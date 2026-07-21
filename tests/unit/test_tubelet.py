"""Tubelet embedding: token counts are mathematically correct."""

from __future__ import annotations

import pytest
import torch

from viora.models.embeddings.tubelet_embedding import TokenGrid, TubeletEmbedding


def test_token_count_matches_formula():
    emb = TubeletEmbedding(in_channels=3, embed_dim=24, tubelet_size=2, patch_size=16)
    x = torch.randn(2, 3, 8, 32, 32)  # T=8,H=W=32
    tokens, grid = emb(x)
    # T'=8/2=4, H'=W'=32/16=2  ->  N = 4*2*2 = 16
    assert grid == TokenGrid(t=4, h=2, w=2)
    assert grid.num_tokens == 16
    assert tokens.shape == (2, 16, 24)


@pytest.mark.parametrize(
    "t,h,w,tt,ps,expected",
    [
        (16, 224, 224, 2, 16, 8 * 14 * 14),
        (16, 224, 224, 4, 16, 4 * 14 * 14),
        (32, 96, 96, 8, 32, 4 * 3 * 3),
    ],
)
def test_grid_for_various_sizes(t, h, w, tt, ps, expected):
    emb = TubeletEmbedding(embed_dim=8, tubelet_size=tt, patch_size=ps)
    assert emb.grid_for(t, h, w).num_tokens == expected


def test_indivisible_dims_raise():
    emb = TubeletEmbedding(embed_dim=8, tubelet_size=2, patch_size=16)
    with pytest.raises(ValueError):
        emb(torch.randn(1, 3, 7, 32, 32))  # 7 not divisible by 2
    with pytest.raises(ValueError):
        emb(torch.randn(1, 3, 8, 30, 32))  # 30 not divisible by 16


def test_wrong_rank_and_channels_raise():
    emb = TubeletEmbedding(in_channels=3, embed_dim=8, tubelet_size=2, patch_size=16)
    with pytest.raises(ValueError):
        emb(torch.randn(3, 8, 32, 32))  # 4D
    with pytest.raises(ValueError):
        emb(torch.randn(1, 1, 8, 32, 32))  # 1 channel
