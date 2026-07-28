"""FastAPI serving: health, studio, index/ask/events, validation, honesty."""

from __future__ import annotations

import shutil
import subprocess

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from viora.serving.api import app  # noqa: E402

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_trained"] is False  # untrained by default, reported honestly


def test_studio_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.text.lstrip().lower().startswith("<!doctype html")
    assert "Viora" in r.text


def test_ask_unknown_index_is_404():
    r = client.post("/video/ask", json={"index_id": "does-not-exist", "question": "hi"})
    assert r.status_code == 404


def test_index_rejects_bad_extension():
    r = client.post("/video/index", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_ask_validates_empty_question():
    r = client.post("/video/ask", json={"index_id": "x", "question": ""})
    assert r.status_code == 422  # pydantic min_length


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg")
def test_full_index_ask_flow(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=64x64:rate=8", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    with open(video, "rb") as f:
        r = client.post("/video/index", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200
    iid = r.json()["index_id"]

    r = client.post("/video/ask", json={"index_id": iid, "question": "what changes?"})
    assert r.status_code == 200
    body = r.json()
    assert body["score_type"] == "uncalibrated_model_score"
    assert body["model_trained"] is False
    assert len(body["evidence"]) >= 1

    r = client.post("/video/events", json={"index_id": iid})
    assert r.status_code == 200 and len(r.json()["moments"]) > 0

    # captioning needs a real tokenizer; the default (dummy-LLM) deployment must say
    # so clearly rather than emitting gibberish
    r = client.post("/video/caption", json={"index_id": iid})
    assert r.status_code == 400
    assert "viora_pragmatic" in r.json()["detail"]  # tells the operator how to fix it


def test_caption_unknown_index_is_404():
    r = client.post("/video/caption", json={"index_id": "does-not-exist"})
    assert r.status_code == 404


def test_health_reports_configured_model_not_a_hardcoded_default(monkeypatch):
    """Regression: /health hardcoded model='viora_tiny', device='cpu', so a
    pragmatic-model deployment silently misreported what it was serving."""
    import viora.serving.api as api

    monkeypatch.setenv("VIORA_MODEL_CONFIG", "configs/model/viora_pragmatic.yaml")
    monkeypatch.setenv("VIORA_DEVICE", "cuda")
    body = client.get("/health").json()
    assert body["model"] == "viora_pragmatic"
    assert body["device"] == "cuda"
    assert api  # module imported for clarity about what is under test


def test_trainable_only_checkpoint_loads_non_strict(tmp_path, monkeypatch):
    """A checkpoint from export_trained_weights holds ONLY trainable params, so
    strict=True would raise on every missing frozen key. The serving loader must
    detect the exporter's flag and load non-strict."""
    import torch

    import viora.serving.api as api
    from viora.models.viora import VioraForVideoUnderstanding
    from viora.training.checkpointing import export_trained_weights, save_checkpoint
    from viora.utils.config import load_config

    # MUST be the same config the API will build, or the mismatch is a shape error
    # rather than the missing-frozen-keys case under test
    cfg_path = "configs/model/viora_tiny.yaml"
    model = VioraForVideoUnderstanding(load_config(cfg_path))
    for p in model.vision.parameters():
        p.requires_grad_(False)
    full = tmp_path / "full.pt"
    save_checkpoint(full, model, step=1)
    small = tmp_path / "trained.pt"
    export_trained_weights(model, full, small)

    # the probe must read the flag without unpickling arbitrary objects
    assert torch.load(small, map_location="cpu", weights_only=True)["trainable_only"] is True

    monkeypatch.setattr(api, "_pipeline", None)
    monkeypatch.setenv("VIORA_MODEL_CONFIG", cfg_path)
    monkeypatch.setenv("VIORA_CHECKPOINT", str(small))
    monkeypatch.setenv("VIORA_DEVICE", "cpu")
    api._get_pipeline()          # would raise RuntimeError(missing keys) if strict
    monkeypatch.setattr(api, "_pipeline", None)   # don't leak the built pipeline
