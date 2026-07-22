"""Checkpoint save/resume.

A checkpoint bundles everything needed to resume exactly: model, optimizer,
scheduler, AMP scaler, step/epoch, resolved config, RNG states, and a version
tag. Architecture mismatches on load raise an understandable error rather than a
cryptic ``size mismatch`` dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from viora import __version__
from viora.utils.config import config_to_dict
from viora.utils.logging import get_logger
from viora.utils.seed import get_rng_states, set_rng_states

logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    step: int = 0,
    epoch: int = 0,
    config: Any = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "viora_version": __version__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config_to_dict(config) if config is not None else None,
        "rng": get_rng_states(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic swap so a crash mid-write can't corrupt the file
    logger.info("saved checkpoint -> %s (step %d)", path, step)


def export_trained_weights(
    model: nn.Module, checkpoint_path: str | Path, out_path: str | Path,
    *, map_location: str = "cpu",
) -> tuple[int, int]:
    """Write a small checkpoint containing only the model's **trainable** params.

    A full checkpoint also stores the frozen backbones (SigLIP + LLM base) and
    optimizer state — GBs. For inference/sharing you only need the trained params
    (LoRA + bridge + heads); the frozen base reloads from its pretrained source.
    Reload with ``load_checkpoint(out_path, model, strict=False)``. Returns
    ``(kept, total)`` tensor counts.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)  # noqa: S614 - self-produced
    sd = ckpt.get("model", ckpt)
    small = {k: v for k, v in sd.items() if k in trainable}
    torch.save(
        {"model": small, "trainable_only": True, "config": ckpt.get("config")}, out_path
    )
    return len(small), len(sd)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
    trusted: bool = True,
) -> dict:
    """Load a checkpoint into the given objects; returns the metadata dict.

    Security: full resume (optimizer / scheduler / RNG state) requires unpickling
    arbitrary objects, so it uses ``weights_only=False`` and MUST only be used on
    checkpoints **you produced/trust**. For an untrusted file, pass
    ``trusted=False`` to load model weights only under ``weights_only=True`` (safe,
    no code execution); optimizer/scheduler/RNG restore is then skipped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if not trusted:
        # Safe path: refuses to execute arbitrary pickled code.
        state = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(state.get("model", state), strict=strict)
        return {"step": 0, "epoch": 0, "config": None, "viora_version": None, "extra": {}}
    ckpt = torch.load(path, map_location=map_location, weights_only=False)  # noqa: S614 - trusted, self-produced

    try:
        model.load_state_dict(ckpt["model"], strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint at {path} does not match the current architecture "
            f"(config mismatch?). Original error: {e}"
        ) from e

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng") is not None:
        set_rng_states(ckpt["rng"])

    return {
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "config": ckpt.get("config"),
        "viora_version": ckpt.get("viora_version"),
        "extra": ckpt.get("extra", {}),
    }
