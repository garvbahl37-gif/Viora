"""Cross-session checkpoint relay through a HuggingFace model repo.

Free Kaggle/Colab sessions don't persist ``/kaggle/working`` across restarts, so a
multi-session training run needs somewhere to hand off the latest checkpoint. This
wraps the exact push/pull calls into two small, tested functions instead of hand-
typed notebook cells: push at the end of a session, pull at the start of the next.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

try:
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    _NOT_FOUND_ERRORS: tuple[type[Exception], ...] = (
        EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError,
    )
except ImportError:  # pragma: no cover - older huggingface_hub without these classes
    _NOT_FOUND_ERRORS = (FileNotFoundError,)


def _peek_step(path: str | Path) -> int | None:
    """Read a checkpoint's own recorded training step, without loading tensors."""
    try:
        step = torch.load(path, map_location="cpu", weights_only=False).get("step")  # noqa: S614 - trusted, self-produced
    except Exception:  # noqa: BLE001 - unreadable/corrupt file -> treat as unknown, not fatal here
        return None
    return int(step) if step is not None else None


class CheckpointRegressionError(ValueError):
    """Raised when a push would silently overwrite a MORE advanced remote checkpoint."""


def push_checkpoint_to_hf(
    local_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    private: bool = True,
    token: str | None = None,
    force: bool = False,
) -> str:
    """Upload ``local_path`` to a HF model repo (created if missing). Returns its URL.

    Refuses (raises :class:`CheckpointRegressionError`) if the local checkpoint's
    recorded step is LOWER than the step already at ``repo_id/path_in_repo`` — this
    happened for real once: a session that accidentally trained from an older resume
    point pushed its (worse) result over a better one already on the Hub, silently
    discarding real training progress. Pass ``force=True`` to push anyway (e.g. you
    are intentionally rolling back).
    """
    local_path = Path(local_path)
    repo_path = path_in_repo or local_path.name
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)

    if not force:
        local_step = _peek_step(local_path)
        try:
            remote_file = hf_hub_download(
                repo_id=repo_id, filename=repo_path, repo_type="model", token=token,
            )
            remote_step = _peek_step(remote_file)
        except _NOT_FOUND_ERRORS:
            remote_step = None  # nothing pushed yet -- no regression possible
        if local_step is not None and remote_step is not None and local_step < remote_step:
            raise CheckpointRegressionError(
                f"refusing to push: local checkpoint is at step {local_step}, but "
                f"{repo_id}/{repo_path} is already at step {remote_step} (this would "
                f"silently discard {remote_step - local_step} steps of progress). "
                "Pass force=True to push anyway if this is intentional."
            )

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=repo_id, repo_type="model", token=token,
    )
    return f"https://huggingface.co/{repo_id}"


def pull_checkpoint_from_hf(
    repo_id: str,
    filename: str,
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> Path | None:
    """Download ``filename`` from a HF model repo. Returns ``None`` if it doesn't
    exist yet (e.g. the first session, before anything has been pushed) instead of
    raising, so callers can fall back to starting fresh."""
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="model",
            local_dir=str(local_dir) if local_dir else None, token=token,
        )
    except _NOT_FOUND_ERRORS:
        return None
    return Path(path)
