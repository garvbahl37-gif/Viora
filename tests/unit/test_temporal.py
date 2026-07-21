"""Temporal subsystem: hierarchy, event tokens, pooling, and bounded memory."""

from __future__ import annotations

import torch

from viora.models.embeddings.tubelet_embedding import TokenGrid
from viora.models.temporal.event_tokenizer import EventTokenizer
from viora.models.temporal.hierarchical_encoder import (
    HierarchicalTemporalEncoder,
    local_band_mask,
)
from viora.models.temporal.temporal_memory import TemporalMemory
from viora.models.temporal.temporal_pooling import SpatialPool, masked_mean
from viora.utils.config import EventConfig, MemoryConfig, TemporalConfig


# ------------------------------------------------------------------ pooling
def test_spatial_pool_mean_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    pool = SpatialPool(24, mode="mean")
    x = torch.randn(2, grid.num_tokens, 24)
    assert pool(x, grid).shape == (2, 4, 24)


def test_spatial_pool_attention_shape():
    grid = TokenGrid(t=4, h=2, w=2)
    pool = SpatialPool(24, mode="attention", num_heads=3)
    x = torch.randn(2, grid.num_tokens, 24)
    assert pool(x, grid).shape == (2, 4, 24)


def test_masked_mean_ignores_padding():
    x = torch.ones(1, 4, 3)
    x[:, 2:] = 99.0
    mask = torch.tensor([[True, True, False, False]])
    assert torch.allclose(masked_mean(x, mask), torch.ones(1, 3))


# ------------------------------------------------------------- hierarchical
def test_band_mask_is_local():
    m = local_band_mask(6, radius=2, device=torch.device("cpu"))
    assert bool(m[0, 0]) and bool(m[0, 1]) and not bool(m[0, 2])
    assert bool(m[3, 2]) and bool(m[3, 4]) and not bool(m[3, 5])


def test_hierarchical_encoder_shape_and_mask():
    cfg = TemporalConfig(dim=24, depth=4, num_heads=3, local_window=2, global_layers=2)
    enc = HierarchicalTemporalEncoder(cfg)
    x = torch.randn(2, 10, 24)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 7:] = False
    out = enc(x, temporal_mask=mask)
    assert out.shape == (2, 10, 24)


def test_hierarchical_encoder_gradients():
    cfg = TemporalConfig(dim=16, depth=3, num_heads=2, local_window=3, global_layers=1)
    enc = HierarchicalTemporalEncoder(cfg)
    out = enc(torch.randn(1, 8, 16))
    out.pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


# -------------------------------------------------------------- event tokens
def test_event_tokenizer_shapes():
    cfg = EventConfig(num_queries=6, dim=24, num_heads=3, num_layers=2)
    tok = EventTokenizer(cfg)
    feats = torch.randn(2, 10, 24)
    out = tok(feats)
    assert out.event_tokens.shape == (2, 6, 24)
    assert out.event_attention.shape == (2, 6, 10)
    assert out.temporal_relevance.shape == (2, 10)
    # attention rows are (near) probability distributions over time
    assert torch.allclose(out.event_attention.sum(-1), torch.ones(2, 6), atol=1e-4)


def test_event_tokenizer_respects_mask():
    cfg = EventConfig(num_queries=4, dim=16, num_heads=2, num_layers=1)
    tok = EventTokenizer(cfg)
    feats = torch.randn(1, 8, 16)
    mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    out = tok(feats, temporal_mask=mask)
    # masked frames receive ~zero attention
    assert out.event_attention[:, :, 4:].abs().max() < 1e-5


# ------------------------------------------------------------------- memory
def _mem(**kw):
    cfg = MemoryConfig(dim=8, short_term_size=4, long_term_size=4, max_tokens=8, **kw)
    return TemporalMemory(cfg), cfg


def test_memory_respects_budget():
    mem, cfg = _mem()
    state = mem.new_state()
    for step in range(20):
        events = torch.randn(3, 8)
        ts = torch.arange(3).float() + step * 3
        state = mem.update(state, events, ts)
        assert state.num_tokens <= cfg.max_tokens
    assert state.short_tokens.shape[0] == cfg.short_term_size
    assert state.long_tokens.shape[0] <= cfg.long_term_size


def test_memory_preserves_recent_verbatim():
    mem, cfg = _mem()
    state = mem.new_state()
    marker = None
    for step in range(6):
        events = torch.randn(4, 8)
        marker = events[-1]
        state = mem.update(state, events, torch.arange(4).float() + step * 4)
    # the very last event must survive verbatim in short-term
    assert torch.allclose(state.short_tokens[-1], marker)


def test_memory_importance_eviction_keeps_important():
    mem, cfg = _mem(eviction="importance", compression="drop")
    state = mem.new_state()
    # push 4 low-importance long-term slots
    for i in range(8):
        state = mem.update(
            state, torch.randn(1, 8), torch.tensor([float(i)]),
            importance=torch.tensor([0.1]),
        )
    # now push a high-importance event that overflows into long-term
    for i in range(8, 13):
        imp = torch.tensor([0.99]) if i == 8 else torch.tensor([0.1])
        state = mem.update(state, torch.randn(1, 8), torch.tensor([float(i)]), importance=imp)
    # the high-importance slot should still be present
    assert state.long_imp.max() > 0.9


def test_memory_read_shapes_and_timestamps():
    mem, cfg = _mem()
    state = mem.new_state()
    for step in range(10):
        state = mem.update(state, torch.randn(2, 8), torch.arange(2).float() + step * 2)
    tokens, ts = mem.read(state)
    assert tokens.shape[0] == state.num_tokens
    assert ts.shape[0] == state.num_tokens
