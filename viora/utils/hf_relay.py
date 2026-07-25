"""Cross-session checkpoint relay through a HuggingFace model repo.

Free Kaggle/Colab sessions don't persist ``/kaggle/working`` across restarts, so a
multi-session training run needs somewhere to hand off the latest checkpoint. This
wraps the exact push/pull calls into two small, tested functions instead of hand-
typed notebook cells: push at the end of a session, pull at the start of the next.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    _NOT_FOUND_ERRORS: tuple[type[Exception], ...] = (
        EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError,
    )
except ImportError:  # pragma: no cover - older huggingface_hub without these classes
    _NOT_FOUND_ERRORS = (FileNotFoundError,)


def push_checkpoint_to_hf(
    local_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload ``local_path`` to a HF model repo (created if missing). Returns its URL."""
    local_path = Path(local_path)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo or local_path.name,
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
