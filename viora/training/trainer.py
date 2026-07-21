"""Training engine.

Native-PyTorch trainer with mixed precision, gradient accumulation + clipping,
LR warmup/schedule, checkpoint save/resume, periodic validation, structured
per-component loss logging, and NaN/Inf detection that surfaces (never hides)
systemic failures. Distributed via DDP is optional and pays nothing when unused.

The trainer is decoupled from the batch format through a ``step_fn(model, batch)
-> VioraOutput``; a default handles :class:`VideoTextBatch` and plain dicts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from viora.data.collators.video_text_collator import VideoTextBatch
from viora.training.checkpointing import load_checkpoint, save_checkpoint
from viora.training.metrics import MetricTracker
from viora.training.optimizer import build_optimizer
from viora.training.scheduler import build_scheduler
from viora.utils.config import TrainingConfig
from viora.utils.device import autocast_context, get_device_info, resolve_amp_dtype
from viora.utils.logging import get_logger
from viora.utils.seed import set_seed

logger = get_logger(__name__)


def default_step_fn(model: nn.Module, batch: Any):
    core = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if isinstance(batch, VideoTextBatch):
        # The 3D-ViT downsamples time by tubelet_size (T -> T'); the pretrained frame
        # encoder keeps one token per frame (stride 1). Match the mask to the tokens.
        stride = core.cfg.vision.tubelet_size if core.cfg.vision.backbone == "scratch" else 1
        tok_mask = (
            batch.token_temporal_mask(stride) if batch.frame_mask is not None else None
        )
        # Honour a configured contrastive weight: without this the retrieval head never
        # trains, so served video-text/retrieval evidence stays at random init.
        lm = getattr(core, "loss_manager", None)
        want_contrastive = bool(lm) and lm.weights.get("contrastive", 0.0) > 0 and batch.input_ids is not None
        return core(
            batch.video, input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            labels=batch.labels, timestamps=batch.timestamps, temporal_mask=tok_mask,
            compute_contrastive=want_contrastive,
        )
    if isinstance(batch, dict):
        return core(**batch)
    raise TypeError(f"cannot interpret batch of type {type(batch)}; pass a custom step_fn")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        *,
        device: str | None = None,
        step_fn: Callable = default_step_fn,
        full_config: Any = None,
    ) -> None:
        self.cfg = cfg
        self.step_fn = step_fn
        self.full_config = full_config
        set_seed(cfg.seed, deterministic=cfg.deterministic)

        self.dev_info = get_device_info(device)
        self.device = self.dev_info.device
        self._apply_freezing(model)
        self.model = model.to(self.device)
        self._wrap_distributed()

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.amp_dtype = resolve_amp_dtype(self.dev_info, cfg.precision)
        self.use_scaler = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(enabled=self.use_scaler)

        self.step = 0
        self.epoch = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricTracker()
        self._last_summary: dict = {}
        self._write_run_metadata()

        if cfg.resume:
            self._resume(cfg.resume)

    # ------------------------------------------------------------- setup
    def _apply_freezing(self, model: nn.Module) -> None:
        if self.cfg.freeze_vision and hasattr(model, "vision"):
            for p in model.vision.parameters():
                p.requires_grad_(False)
        if self.cfg.freeze_llm and hasattr(model, "llm"):
            for p in model.llm.parameters():
                p.requires_grad_(False)

    def _wrap_distributed(self) -> None:
        """Wrap in FSDP (large models) or DDP under torchrun; no-op single-process."""
        from viora.training.distributed import (
            is_distributed,
            viora_transformer_layers,
            wrap_ddp,
            wrap_fsdp,
        )

        if not is_distributed():
            return
        if self.cfg.fsdp:
            self.model = wrap_fsdp(
                self.model,
                transformer_layer_classes=viora_transformer_layers(),
                param_dtype=self.amp_dtype,
            )
            logger.info("wrapped model in FSDP")
        elif self.cfg.ddp:
            self.model = wrap_ddp(self.model)
            logger.info("wrapped model in DDP")

    def _write_run_metadata(self) -> None:
        counts = self.model.count_parameters() if hasattr(self.model, "count_parameters") else {}
        meta = {
            "device": self.dev_info.name,
            "precision": self.cfg.precision,
            "amp_dtype": str(self.amp_dtype),
            "seed": self.cfg.seed,
            "stage": self.cfg.stage,
            "parameters": counts,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    def _resume(self, path: str) -> None:
        meta = load_checkpoint(
            path, self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, map_location=str(self.device),
        )
        self.step, self.epoch = meta["step"], meta["epoch"]
        logger.info("resumed from %s at step %d", path, self.step)

    # --------------------------------------------------------- train loop
    def _to_device(self, batch: Any) -> Any:
        """Move a batch onto the training device (collators produce CPU tensors)."""
        if isinstance(batch, VideoTextBatch):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def train_step(self, batch: Any) -> dict:
        self.model.train()
        batch = self._to_device(batch)
        with autocast_context(self.device.type, enabled=self.amp_dtype is not None, dtype=self.amp_dtype):
            out = self.step_fn(self.model, batch)
        if out.loss is None:
            raise ValueError("step_fn returned no loss")

        if not torch.isfinite(out.loss):
            logger.error(
                "non-finite loss at step %d: components=%s nonfinite=%s",
                self.step, out.losses, out.diagnostics.get("nonfinite_losses"),
            )
            return {"loss": float("nan"), "skipped": True, **out.losses}

        loss = out.loss / self.cfg.grad_accum
        self.scaler.scale(loss).backward()
        return {"loss": float(out.loss.detach()), "skipped": False, **out.losses}

    def _optimizer_step(self) -> float:
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        # With the fp16 GradScaler, an overflowed step is SKIPPED: optimizer.step() is a no-op
        # and update() lowers the scale. Advance the LR schedule only on a real optimizer step,
        # so the schedule stays aligned and PyTorch doesn't warn about step() ordering.
        scale_before = self.scaler.get_scale() if self.use_scaler else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if scale_before is None or self.scaler.get_scale() >= scale_before:
            self.scheduler.step()
        return float(grad_norm)

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict:
        max_steps = self.cfg.max_steps
        logger.info("training: up to %d steps on %s (%s)", max_steps, self.dev_info.name, self.cfg.precision)
        t0 = time.time()
        micro = 0
        done = False
        while not done:
            for batch in train_loader:
                stats = self.train_step(batch)
                micro += 1
                if micro % self.cfg.grad_accum == 0:
                    grad_norm = self._optimizer_step()
                    self.step += 1
                    self.metrics.update({**{k: v for k, v in stats.items() if isinstance(v, (int, float))},
                                         "grad_norm": grad_norm, "lr": self.scheduler.get_last_lr()[0]})
                    if self.step % self.cfg.log_every == 0:
                        self._log(t0)
                    if self.cfg.save_every and self.step % self.cfg.save_every == 0:
                        self.save(f"step_{self.step}")
                    if val_loader is not None and self.cfg.eval_every and self.step % self.cfg.eval_every == 0:
                        self.evaluate(val_loader)
                    if self.step >= max_steps:
                        done = True
                        break
            self.epoch += 1
            if self.cfg.epochs and self.epoch >= self.cfg.epochs:
                done = True
        self.save("final")
        return self.metrics.summary() or self._last_summary

    def _log(self, t0: float) -> None:
        s = self.metrics.summary()
        self._last_summary = s
        rate = self.step / max(1e-6, time.time() - t0)
        logger.info(
            "step %d | loss %.4f | lr %.2e | grad %.2f | %.2f it/s",
            self.step, s.get("loss", 0.0), s.get("lr", 0.0), s.get("grad_norm", 0.0), rate,
        )
        self.metrics.reset()

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        tracker = MetricTracker()
        for batch in val_loader:
            batch = self._to_device(batch)
            with autocast_context(self.device.type, enabled=self.amp_dtype is not None, dtype=self.amp_dtype):
                out = self.step_fn(self.model, batch)
            if out.loss is not None and torch.isfinite(out.loss):
                tracker.update({"val_loss": float(out.loss), **out.losses})
        summary = tracker.summary()
        logger.info("eval @ step %d: %s", self.step, {k: round(v, 4) for k, v in summary.items()})
        return summary

    # ------------------------------------------------------------- io
    def save(self, tag: str) -> Path:
        path = self.output_dir / f"{tag}.pt"
        save_checkpoint(
            path, self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, step=self.step, epoch=self.epoch, config=self.full_config,
        )
        self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        """Keep only the newest ``keep_last_checkpoints`` ``step_*.pt`` files.

        Whole-model checkpoints are large (~GBs); on a capped disk (e.g. Kaggle's
        20 GB) saving every N steps otherwise fills the disk and aborts the run.
        ``final.pt`` and any non ``step_`` tag are never pruned.
        """
        keep = getattr(self.cfg, "keep_last_checkpoints", 0)
        if not keep or keep < 1:
            return
        ckpts = sorted(
            self.output_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1,
        )
        for old in ckpts[:-keep]:
            try:
                old.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                logger.warning("could not remove old checkpoint %s", old)
