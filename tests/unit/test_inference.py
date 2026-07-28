"""Offline indexing/QA and streaming: index round-trip, evidence, bounded memory."""

from __future__ import annotations

import shutil
import subprocess

import pytest
import torch

from tests.unit.test_viora_e2e import tiny_viora_config
from viora.inference.pipeline import VideoIndex, VioraInferencePipeline
from viora.inference.streaming import StreamingVioraEngine
from viora.models.viora import VioraForVideoUnderstanding

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg")


def _model():
    return VioraForVideoUnderstanding(tiny_viora_config())


def test_video_index_save_load_round_trip(tmp_path):
    idx = VideoIndex(
        resampled=torch.randn(6, 32), event_tokens=torch.randn(4, 32),
        temporal=torch.randn(5, 32), timestamps=torch.linspace(0, 4, 5),
        duration=4.0, metadata={"path": "x.mp4"},
    )
    path = tmp_path / "v.viora"
    idx.save(path)
    back = VideoIndex.load(path)
    assert torch.allclose(back.temporal, idx.temporal)
    assert back.duration == 4.0 and back.metadata["path"] == "x.mp4"


def test_ask_returns_evidence_and_score_type():
    model = _model()
    pipe = VioraInferencePipeline(model, device="cpu")
    idx = VideoIndex(
        resampled=torch.randn(6, 32), event_tokens=torch.randn(4, 32),
        temporal=torch.randn(6, 32), timestamps=torch.linspace(0, 10, 6),
        duration=10.0,
    )
    ans = pipe.ask(idx, "When does the person appear?", top_k=2)
    assert ans.score_type == "uncalibrated_model_score"
    assert 0.0 <= ans.score <= 1.0
    assert len(ans.evidence) >= 1
    for ev in ans.evidence:
        assert 0.0 <= ev.start <= ev.end <= 10.0 + 1e-3
    assert ans.diagnostics["model_trained"] is False  # honest labeling


def test_index_timestamps_match_temporal_tokens_for_pretrained_backbone(monkeypatch):
    """Regression: the pretrained frame encoder emits ONE temporal token per frame,
    but index() still divided the frame timestamps by tubelet_size -- yielding half as
    many timestamps as tokens. ask()'s topk over tokens then indexed past the end of
    `timestamps` and raised IndexError, breaking /video/ask for every trained
    (pragmatic) deployment. The stride must be 1 unless the backbone is 'scratch'.
    """
    cfg = tiny_viora_config()
    cfg.vision.backbone = "pretrained"
    model = _model()                       # scratch model is fine; only cfg drives stride
    model.cfg.vision.backbone = "pretrained"
    model.cfg.vision.tubelet_size = 2      # would halve the timestamps if wrongly applied

    pipe = VioraInferencePipeline(model, device="cpu")
    assert pipe.tubelet == 1, "pretrained backbone must use stride 1, not tubelet_size"

    model.cfg.vision.backbone = "scratch"
    assert VioraInferencePipeline(model, device="cpu").tubelet == 2  # 3D ViT still strides


def test_caption_requires_real_tokenizer():
    # the tiny config uses a dummy LLM (no tokenizer) -> caption() must fail clearly,
    # not produce garbage; real captioning needs a trained non-dummy model.
    pipe = VioraInferencePipeline(_model(), device="cpu")
    idx = VideoIndex(
        resampled=torch.randn(6, 32), event_tokens=torch.randn(4, 32),
        temporal=torch.randn(6, 32), timestamps=torch.linspace(0, 4, 6), duration=4.0,
    )
    with pytest.raises(RuntimeError, match="tokenizer"):
        pipe.caption(idx)


@requires_ffmpeg
def test_index_real_video(tmp_path):
    video = tmp_path / "t.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=64x64:rate=8", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    pipe = VioraInferencePipeline(_model(), device="cpu")
    idx = pipe.index(video)
    assert idx.temporal.shape[0] == idx.timestamps.shape[0]
    assert idx.duration > 1.0
    ans = pipe.ask(idx, "what happens")
    assert len(ans.evidence) >= 1


def test_streaming_accumulates_bounded_memory():
    model = _model()
    engine = StreamingVioraEngine(model, device="cpu")
    for _ in range(6):
        res = engine.add_video_chunk(torch.rand(3, 8, 32, 32), fps=4.0)
        assert res.num_events > 0
        # memory never exceeds its budget across chunks
        assert res.memory["total"] <= res.memory["budget"]
    summary = engine.get_memory_summary()
    assert summary["stream_time_s"] > 0
    ans = engine.ask("what happened at the start?")
    assert len(ans.evidence) >= 1


def test_streaming_reset():
    engine = StreamingVioraEngine(_model(), device="cpu")
    engine.add_video_chunk(torch.rand(3, 8, 32, 32))
    engine.reset_memory()
    assert engine.get_memory_summary()["total"] == 0
    assert engine.ask("q").diagnostics.get("memory_empty") is True
