"""End-to-end: a video tensor traverses the whole architecture and the language
loss backpropagates — the V1 "technically functional" definition of done."""

from __future__ import annotations

import torch

from viora.models.viora import VioraForVideoUnderstanding
from viora.utils.config import (
    EventConfig,
    GroundingConfig,
    LLMConfig,
    MemoryConfig,
    ProjectorConfig,
    ResamplerConfig,
    TemporalConfig,
    VioraConfig,
    VisionConfig,
)


def tiny_viora_config() -> VioraConfig:
    d = 32
    return VioraConfig(
        name="viora_micro",
        vision=VisionConfig(image_size=32, num_frames=8, tubelet_size=2, patch_size=16,
                            dim=d, depth=2, num_heads=4, attention_mode="factorized"),
        temporal=TemporalConfig(dim=d, depth=2, num_heads=4, local_window=2, global_layers=1),
        event=EventConfig(num_queries=4, dim=d, num_heads=4, num_layers=1),
        memory=MemoryConfig(dim=d, short_term_size=4, long_term_size=4, max_tokens=8),
        resampler=ResamplerConfig(num_queries=6, dim=d, num_heads=4, num_layers=1),
        projector=ProjectorConfig(type="mlp", input_dim=d, output_dim=0, depth=2),
        llm=LLMConfig(dummy=True, hidden_size=48, vocab_size=64, max_length=128),
        grounding=GroundingConfig(num_bins=16, hidden_dim=d),
        tasks=["video_qa", "temporal_grounding"],
    )


def _batch(model):
    video = torch.randn(2, 3, 8, 32, 32)
    vid = model.llm.video_token_id
    input_ids = torch.tensor([[10, vid, 11, 12, 13], [5, vid, 6, 7, 8]])
    labels = input_ids.clone()
    attn = torch.ones_like(input_ids)
    timestamps = torch.linspace(0, 4, 8).repeat(2, 1)  # per-frame
    return video, input_ids, attn, labels, timestamps


def test_end_to_end_forward_and_backward():
    model = VioraForVideoUnderstanding(tiny_viora_config())
    video, input_ids, attn, labels, ts = _batch(model)
    sb = torch.tensor([2, 5])
    eb = torch.tensor([8, 11])

    out = model(video, input_ids=input_ids, attention_mask=attn, labels=labels,
                timestamps=ts, grounding_targets=(sb, eb))

    assert out.loss is not None and torch.isfinite(out.loss)
    assert "lm" in out.losses and "grounding" in out.losses
    # grounding evidence has valid shapes
    assert out.temporal_predictions.start_logits.shape == (2, 16)
    assert out.temporal_predictions.confidence.shape == (2,)
    # event attention is exposed for interpretability: [B, E, T'] = [2, 4, 4]
    assert out.event_attention.shape == (2, 4, 4)

    out.loss.backward()
    # gradients reached every major subsystem
    for name, module in [
        ("vision", model.vision), ("temporal", model.temporal),
        ("resampler", model.resampler), ("projector", model.projector), ("llm", model.llm),
    ]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and any(g is not None and g.abs().sum() > 0 for g in grads), f"no grad in {name}"


def test_contrastive_path():
    model = VioraForVideoUnderstanding(tiny_viora_config())
    video, input_ids, attn, labels, ts = _batch(model)
    out = model(video, input_ids=input_ids, attention_mask=attn, labels=labels,
                timestamps=ts, compute_contrastive=True)
    assert "contrastive" in out.losses
    assert out.video_embed is not None and out.text_embed is not None


def test_video_only_forward_runs():
    """No text -> no LM loss, but grounding still produces evidence."""
    model = VioraForVideoUnderstanding(tiny_viora_config())
    out = model(torch.randn(1, 3, 8, 32, 32))
    assert out.temporal_predictions is not None
    assert out.resampled_tokens.shape == (1, 6, 32)


def test_parameter_accounting():
    model = VioraForVideoUnderstanding(tiny_viora_config())
    counts = model.count_parameters()
    for key in ("total", "vision", "temporal", "multimodal", "llm", "heads"):
        assert counts[key] > 0
    # subsystems sum to <= total (heads/embeddings may overlap-tie, so <=)
    assert counts["total"] >= counts["vision"] + counts["llm"]


def test_state_dict_round_trip():
    cfg = tiny_viora_config()
    a = VioraForVideoUnderstanding(cfg)
    b = VioraForVideoUnderstanding(cfg)
    b.load_state_dict(a.state_dict())
    video = torch.randn(1, 3, 8, 32, 32)
    a.eval()
    b.eval()
    with torch.no_grad():
        ra = a(video).resampled_tokens
        rb = b(video).resampled_tokens
    assert torch.allclose(ra, rb, atol=1e-5)
