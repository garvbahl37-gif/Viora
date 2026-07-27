"""HF checkpoint relay: push a checkpoint after a session, pull it to resume the next."""

from __future__ import annotations

import json as jsonlib
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


def _no_remote_anything(**kw):
    """Simulates a repo with nothing pushed yet: both the sidecar and the full
    checkpoint 404. Used by tests where the local checkpoint has no readable step
    (fake byte content) and the remote-step lookup should never even matter."""
    raise _NOT_FOUND_ERRORS[0]("nothing here yet")


def test_push_checkpoint_creates_repo_and_uploads(tmp_path, monkeypatch):
    ckpt = tmp_path / "final.pt"
    ckpt.write_bytes(b"fake weights")  # not a real checkpoint -> _peek_step returns None
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _no_remote_anything)

    url = push_checkpoint_to_hf(ckpt, "someuser/viora-msrvtt", token="hf_fake")

    assert fake.created == [("someuser/viora-msrvtt", "model", True, True)]
    # local_step is None (unreadable fake bytes) -> no sidecar uploaded, just the checkpoint
    assert fake.uploaded == [(str(ckpt), "final.pt", "someuser/viora-msrvtt", "model")]
    assert url == "https://huggingface.co/someuser/viora-msrvtt"


def test_push_checkpoint_custom_path_in_repo(tmp_path, monkeypatch):
    ckpt = tmp_path / "step_1000.pt"
    ckpt.write_bytes(b"x")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _no_remote_anything)

    push_checkpoint_to_hf(ckpt, "u/r", path_in_repo="checkpoints/step_1000.pt", token="t")

    assert fake.uploaded[0][1] == "checkpoints/step_1000.pt"


def _filename_aware_download(pt_step: int | None, sidecar_missing: bool = True):
    """Mock hf_hub_download that distinguishes the sidecar request from the full-
    checkpoint request by filename, rather than relying on an incidental
    FileNotFoundError from a bogus path string."""
    def _download(**kw):
        filename = kw["filename"]
        if filename.endswith(".step.json"):
            if sidecar_missing:
                raise _NOT_FOUND_ERRORS[0]("no sidecar")
            raise AssertionError("this test expects the sidecar path, not this fallback")
        if pt_step is None:
            raise _NOT_FOUND_ERRORS[0]("no remote checkpoint")
        return "remote.pt"
    return _download


def test_push_refuses_to_regress_a_more_advanced_remote_checkpoint(tmp_path, monkeypatch):
    """This happened for real: a session trained to a LOWER step than what was
    already on HF, then pushed, silently discarding real progress. Must refuse."""
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _filename_aware_download(15000))
    monkeypatch.setattr(
        "viora.utils.hf_relay._peek_step",
        lambda p: {"final.pt": 11400, "remote.pt": 15000}[Path(p).name],
    )

    with pytest.raises(CheckpointRegressionError, match="step 11400.*step 15000"):
        push_checkpoint_to_hf(local, "u/r", token="t")

    assert fake.uploaded == []  # refused BEFORE uploading anything


def test_push_proceeds_when_local_step_is_ahead(tmp_path, monkeypatch):
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _filename_aware_download(15000))
    monkeypatch.setattr(
        "viora.utils.hf_relay._peek_step",
        lambda p: {"final.pt": 20000, "remote.pt": 15000}[Path(p).name],
    )

    push_checkpoint_to_hf(local, "u/r", token="t")

    assert len(fake.uploaded) == 2  # checkpoint + fresh step sidecar
    assert fake.uploaded[0][1] == "final.pt"
    assert fake.uploaded[1][1] == "final.step.json"
    assert jsonlib.loads(fake.uploaded[1][0]) == {"step": 20000}


def test_push_force_bypasses_the_regression_check(tmp_path, monkeypatch):
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)

    def _must_not_be_called(**kw):
        raise AssertionError("force=True must skip the remote-step lookup entirely")

    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _must_not_be_called)
    monkeypatch.setattr("viora.utils.hf_relay._peek_step", lambda p: 100)

    push_checkpoint_to_hf(local, "u/r", token="t", force=True)

    assert len(fake.uploaded) == 2  # pushed anyway (checkpoint + sidecar), no remote check made


def test_push_uses_sidecar_without_downloading_full_remote_checkpoint(tmp_path, monkeypatch):
    """The fast/safe path: a JSON sidecar (uploaded by every prior push through this
    function) lets the regression check read the remote step from a few bytes of
    JSON -- no pickle deserialization of a multi-GB remote file at all."""
    sidecar_file = tmp_path / "remote_sidecar.json"
    sidecar_file.write_text(jsonlib.dumps({"step": 15000}))
    local = tmp_path / "final.pt"
    local.write_bytes(b"local")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)

    requested = []

    def _download(**kw):
        requested.append(kw["filename"])
        if kw["filename"].endswith(".step.json"):
            return str(sidecar_file)
        raise AssertionError("must not fetch the full checkpoint when a sidecar is present")

    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _download)
    monkeypatch.setattr("viora.utils.hf_relay._peek_step", lambda p: 20000)  # local step only

    push_checkpoint_to_hf(local, "u/r", token="t")

    assert requested == ["final.step.json"]  # only the tiny sidecar was ever fetched
    assert len(fake.uploaded) == 2


def test_pull_checkpoint_returns_local_path(monkeypatch, tmp_path):
    downloaded = tmp_path / "final.pt"
    downloaded.write_bytes(b"x")
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", lambda **kw: str(downloaded))

    out = pull_checkpoint_from_hf("u/r", "final.pt", token="t")

    assert out == downloaded


def test_pull_checkpoint_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _raise_not_found)

    assert pull_checkpoint_from_hf("u/r", "final.pt", token="t") is None
