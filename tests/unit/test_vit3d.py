"""3D ViT: forward shapes, both attention modes, masking, and gradient flow."""

from __future__ import annotations

import copy

import torch

from viora.models.vision.vit3d import VioraVisionTransformer3D


def test_forward_shapes(tiny_vision_config, video_batch):
    model = VioraVisionTransformer3D(tiny_vision_config)
    out = model(video_batch)
    # grid: T'=4, H'=W'=2 -> N=16
    assert out.tokens.shape == (2, 16, tiny_vision_config.dim)
    assert out.frame_features.shape == (2, 4, tiny_vision_config.dim)
    assert out.grid.num_tokens == 16


def test_full_attention_mode(tiny_vision_config, video_batch):
    cfg = copy.deepcopy(tiny_vision_config)
    cfg.attention_mode = "full"
    model = VioraVisionTransformer3D(cfg)
    out = model(video_batch)
    assert out.tokens.shape == (2, 16, cfg.dim)


def test_temporal_mask_forward(tiny_vision_config, video_batch):
    model = VioraVisionTransformer3D(tiny_vision_config)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    out = model(video_batch, temporal_mask=mask)
    assert out.tokens.shape == (2, 16, tiny_vision_config.dim)


def test_timestamps_forward(tiny_vision_config, video_batch):
    model = VioraVisionTransformer3D(tiny_vision_config)
    ts = torch.tensor([[0.0, 0.5, 1.0, 1.5], [0.0, 1.0, 2.0, 3.0]])
    out = model(video_batch, timestamps=ts)
    assert out.tokens.shape == (2, 16, tiny_vision_config.dim)


def test_per_frame_timestamps_are_pooled(tiny_vision_config, video_batch):
    """Callers may pass raw per-frame timestamps (T=8); they pool to per-token (T'=4)."""
    model = VioraVisionTransformer3D(tiny_vision_config)
    per_frame = torch.arange(8).float().repeat(2, 1)  # [2, 8]
    out = model(video_batch, timestamps=per_frame)
    assert out.tokens.shape == (2, 16, tiny_vision_config.dim)


def test_bad_timestamp_length_raises(tiny_vision_config, video_batch):
    import pytest

    model = VioraVisionTransformer3D(tiny_vision_config)
    with pytest.raises(ValueError):
        model(video_batch, timestamps=torch.zeros(2, 5))  # neither T'=4 nor T=8


def test_gradients_flow(tiny_vision_config, video_batch):
    model = VioraVisionTransformer3D(tiny_vision_config)
    out = model(video_batch)
    loss = out.tokens.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_gradient_checkpointing_matches(tiny_vision_config, video_batch):
    model = VioraVisionTransformer3D(tiny_vision_config).train()
    out_ref = model(video_batch).tokens
    model.set_gradient_checkpointing(True)
    out_ckpt = model(video_batch).tokens
    assert torch.allclose(out_ref, out_ckpt, atol=1e-5)


def test_rmsnorm_and_swiglu_variant(tiny_vision_config, video_batch):
    cfg = copy.deepcopy(tiny_vision_config)
    cfg.norm_type = "rmsnorm"
    cfg.mlp_type = "swiglu"
    model = VioraVisionTransformer3D(cfg)
    assert model(video_batch).tokens.shape == (2, 16, cfg.dim)


def test_num_parameters_positive(tiny_vision_config):
    model = VioraVisionTransformer3D(tiny_vision_config)
    n = model.num_parameters()
    assert n == sum(p.numel() for p in model.parameters())
    assert n > 0
