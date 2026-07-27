"""HF checkpoint relay: push a checkpoint after a session, pull it to resume the next."""

from __future__ import annotations

from pathlib import Path

import pytest

from viora.utils.hf_relay import (
    _NOT_FOUND_ERRORS,
    CheckpointRegressionError,
    pull_checkpoint_from_hf,
    push_checkpoint_to_hf,
)


class _FakeApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, *, repo_type, exist_ok, private, token=None):
        self.created.append((repo_id, repo_type, exist_ok, private))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, token=None):
        self.uploaded.append((path_or_fileobj, path_in_repo, repo_id, repo_type))


def _raise_not_found(**kw):
    raise _NOT_FOUND_ERRORS[0]("no remote file yet")


def test_push_checkpoint_creates_repo_and_uploads(tmp_path, monkeypatch):
    ckpt = tmp_path / "final.pt"
    ckpt.write_bytes(b"fake weights")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _raise_not_found)  # no remote yet

    url = push_checkpoint_to_hf(ckpt, "someuser/viora-msrvtt", token="hf_fake")

    assert fake.created == [("someuser/viora-msrvtt", "model", True, True)]
    assert fake.uploaded == [(str(ckpt), "final.pt", "someuser/viora-msrvtt", "model")]
    assert url == "https://huggingface.co/someuser/viora-msrvtt"


def test_push_checkpoint_custom_path_in_repo(tmp_path, monkeypatch):
    ckpt = tmp_path / "step_1000.pt"
    ckpt.write_bytes(b"x")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _raise_not_found)

    push_checkpoint_to_hf(ckpt, "u/r", path_in_repo="checkpoints/step_1000.pt", token="t")

    assert fake.uploaded[0][1] == "checkpoints/step_1000.pt"


def test_push_refuses_to_regress_a_more_advanced_remote_checkpoint(tmp_path, monkeypatch):
    """This happened for real: a session trained to a LOWER step than what was
    already on HF, then pushed, silently discarding real progress. Must refuse."""
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", lambda **kw: "remote.pt")
    monkeypatch.setattr(
        "viora.utils.hf_relay._peek_step",
        lambda p: {"final.pt": 11400, "remote.pt": 15000}[Path(p).name],
    )

    with pytest.raises(CheckpointRegressionError, match="step 11400.*step 15000"):
        push_checkpoint_to_hf(local, "u/r", token="t")

    assert fake.uploaded == []  # refused BEFORE uploading


def test_push_proceeds_when_local_step_is_ahead(tmp_path, monkeypatch):
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", lambda **kw: "remote.pt")
    monkeypatch.setattr(
        "viora.utils.hf_relay._peek_step",
        lambda p: {"final.pt": 20000, "remote.pt": 15000}[Path(p).name],
    )

    push_checkpoint_to_hf(local, "u/r", token="t")

    assert len(fake.uploaded) == 1


def test_push_force_bypasses_the_regression_check(tmp_path, monkeypatch):
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", lambda **kw: "remote.pt")
    monkeypatch.setattr(
        "viora.utils.hf_relay._peek_step",
        lambda p: {"final.pt": 100, "remote.pt": 15000}[Path(p).name],
    )

    push_checkpoint_to_hf(local, "u/r", token="t", force=True)

    assert len(fake.uploaded) == 1  # pushed anyway; hf_hub_download never even needed calling


def test_pull_checkpoint_returns_local_path(monkeypatch, tmp_path):
    downloaded = tmp_path / "final.pt"
    downloaded.write_bytes(b"x")
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", lambda **kw: str(downloaded))

    out = pull_checkpoint_from_hf("u/r", "final.pt", token="t")

    assert out == downloaded


def test_pull_checkpoint_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _raise_not_found)

    assert pull_checkpoint_from_hf("u/r", "final.pt", token="t") is None
