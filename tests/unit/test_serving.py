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
