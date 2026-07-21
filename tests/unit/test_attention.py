"""Attention: shapes, SDPA/manual equivalence, and padding-mask correctness."""

from __future__ import annotations

import torch

from viora.models.attention.attention_utils import MultiHeadAttention
from viora.models.attention.factorized_attention import FactorizedAttention
from viora.models.attention.spatial_attention import SpatialAttention
from viora.models.attention.temporal_attention import TemporalAttention
from viora.models.embeddings.tubelet_embedding import TokenGrid


def test_mha_self_attention_shape():
    mha = MultiHeadAttention(24, 3)
    x = torch.randn(2, 10, 24)
    assert mha(x).shape == (2, 10, 24)


def test_mha_cross_attention_shape():
    mha = MultiHeadAttention(24, 3, kv_dim=16)
    q = torch.randn(2, 5, 24)
    ctx = torch.randn(2, 9, 16)
    assert mha(q, context=ctx).shape == (2, 5, 24)


def test_sdpa_and_manual_match():
    torch.manual_seed(0)
    mha_sdpa = MultiHeadAttention(24, 3, use_sdpa=True).eval()
    mha_manual = MultiHeadAttention(24, 3, use_sdpa=False).eval()
    mha_manual.load_state_dict(mha_sdpa.state_dict())
    x = torch.randn(2, 7, 24)
    with torch.no_grad():
        a = mha_sdpa(x)
        b = mha_manual(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_spatial_attention_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    sa = SpatialAttention(24, 3)
    x = torch.randn(2, grid.num_tokens, 24)
    assert sa(x, grid).shape == (2, 16, 24)


def test_temporal_attention_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    ta = TemporalAttention(24, 3)
    x = torch.randn(2, grid.num_tokens, 24)
    assert ta(x, grid).shape == (2, 16, 24)


def test_factorized_attention_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    fa = FactorizedAttention(24, 3)
    x = torch.randn(2, grid.num_tokens, 24)
    assert fa(x, grid).shape == (2, 16, 24)


def test_temporal_padding_mask_excludes_masked_keys():
    """Valid-position outputs must not depend on the content of masked frames."""
    grid = TokenGrid(t=4, h=1, w=2)  # s=2, N=8; token index = t*2 + s_i
    ta = TemporalAttention(16, 2, use_sdpa=True).eval()
    mask = torch.tensor([[True, True, False, False]])  # frames 2,3 are padding

    x1 = torch.randn(1, grid.num_tokens, 16)
    x2 = x1.clone()
    x2[:, 4:8, :] = torch.randn(1, 4, 16)  # perturb only masked frames (t=2,3)

    with torch.no_grad():
        o1 = ta(x1, grid, temporal_mask=mask)
        o2 = ta(x2, grid, temporal_mask=mask)

    # tokens for valid frames t=0,1 -> indices 0..3 must be identical
    assert torch.allclose(o1[:, :4, :], o2[:, :4, :], atol=1e-6)
    # sanity: masked-frame outputs did change (perturbation took effect somewhere)
    assert not torch.allclose(o1[:, 4:, :], o2[:, 4:, :])
