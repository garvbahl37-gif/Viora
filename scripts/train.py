#!/usr/bin/env python3
"""Train Viora (generic entrypoint for every curriculum stage).

    # a few optimization steps on synthetic data — no dataset needed:
    python scripts/train.py --smoke

    # a real stage (requires prepared datasets; see scripts/prepare_dataset.py):
    python scripts/train.py --model configs/model/viora_tiny.yaml \
                            --train configs/training/stage2_alignment.yaml

This single script covers all stages (the stage is set in the training config),
which is cleaner than five near-identical ``train_stageN.py`` files.
"""
from __future__ import annotations

import argparse
import os

# MPS lacks Conv3d (used by tubelet embedding); fall back to CPU for those ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from viora.data.collators.video_text_collator import VideoTextCollator
from viora.data.datasets.base import SyntheticVideoDataset
from viora.models.viora import VioraForVideoUnderstanding
from viora.training.trainer import Trainer
from viora.utils.config import TrainingConfig, load_config
from viora.utils.logging import configure_logging


# Tokenizers are classes (not closures) so the collator is picklable for DataLoader
# workers (num_workers > 0). A local closure raises "Can't get local object" on spawn.
class smoke_tokenize_fn:
    def __init__(self, video_token_id: int, vocab: int) -> None:
        self.video_token_id = video_token_id
        self.vocab = vocab

    def __call__(self, texts: list[str]) -> dict:
        rows = [[self.video_token_id] + [abs(hash(w)) % (self.vocab - 1) for w in t.split()][:14]
                for t in texts]
        length = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), length, dtype=torch.long)
        attn = torch.zeros(len(rows), length, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            attn[i, : len(r)] = 1
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class hf_tokenize_fn:
    """Real-LLM tokenizer for sharded data. Texts already carry the <video> token."""

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100  # ignore pads; injection ignores visual tokens
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def _transform_for(model_cfg):
    from viora.data.preprocessing.transforms import VideoTransform
    size = model_cfg.vision.image_size
    if model_cfg.vision.backbone == "pretrained":  # SigLIP expects [-1, 1]
        return VideoTransform(size=size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return VideoTransform(size=size)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Viora-1.")
    ap.add_argument("--model", default="configs/model/viora_tiny.yaml")
    ap.add_argument("--train", default="configs/training/dev_smoke.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic-data smoke run")
    ap.add_argument("--shards", help="WebDataset shard pattern for REAL training, "
                                     "e.g. 'data/shards/train-{000000..000099}.tar'")
    ap.add_argument("--device", default=None,
                    help="cpu|cuda|mps|auto (default: auto; cpu for --smoke). Use cpu on Macs "
                         "where the pretrained-vision path hits MPS conv issues.")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. training.max_steps=10")
    args = ap.parse_args()
    configure_logging("INFO")

    # Route CLI overrides: "training.*" -> training config (prefix stripped), rest -> model config.
    # e.g. training.max_steps=3000 training.batch_size=4 llm.name_or_path=Qwen/Qwen2.5-1.5B-Instruct
    model_overrides = [o for o in args.overrides if not o.startswith("training.")]
    train_overrides = [o[len("training."):] for o in args.overrides if o.startswith("training.")]

    model_cfg = load_config(args.model, overrides=model_overrides)
    if args.smoke:
        model_cfg.vision.image_size = 32
        model_cfg.vision.num_frames = 8
        for sub in (model_cfg.vision, model_cfg.temporal, model_cfg.event,
                    model_cfg.resampler, model_cfg.memory, model_cfg.grounding):
            if hasattr(sub, "dim"):
                sub.dim = 64
            if hasattr(sub, "num_heads"):
                sub.num_heads = 4  # 64 must be divisible by heads
        model_cfg.vision.depth = 2
        model_cfg.temporal.depth = 2
        model_cfg.llm.dummy = True
        model_cfg.llm.hidden_size = 64
        model_cfg.llm.vocab_size = 128
        model_cfg.projector.output_dim = 0

    train_cfg: TrainingConfig = load_config(  # type: ignore
        args.train, schema=TrainingConfig, overrides=train_overrides
    )

    model = VioraForVideoUnderstanding(model_cfg)
    collator = VideoTextCollator()

    if args.shards:
        # REAL training from sharded video-text data with the real LLM tokenizer.
        from viora.data.webdataset_pipeline import build_video_text_webdataset

        if model.llm.tokenizer is None:
            raise SystemExit("--shards needs a real LLM (set llm.dummy=false in the model config)")
        collator.tokenize_fn = hf_tokenize_fn(model.llm.tokenizer, model_cfg.llm.max_length)
        ds = build_video_text_webdataset(
            args.shards, num_frames=model_cfg.vision.num_frames, transform=_transform_for(model_cfg)
        )
        loader = DataLoader(ds, batch_size=train_cfg.batch_size, collate_fn=collator,
                            num_workers=train_cfg.num_workers)
    else:
        ds = SyntheticVideoDataset(
            size=32, num_frames=model_cfg.vision.num_frames, image_size=model_cfg.vision.image_size
        )
        collator.tokenize_fn = smoke_tokenize_fn(model.llm.video_token_id, model.llm.vocab_size)
        loader = DataLoader(ds, batch_size=train_cfg.batch_size, collate_fn=collator,
                            num_workers=train_cfg.num_workers)

    # tiny smoke runs on CPU (fast, and avoids MPS Conv3d gaps); real runs auto-select
    device = "cpu" if args.smoke else args.device
    trainer = Trainer(model, train_cfg, device=device, full_config=model_cfg)
    summary = trainer.train(loader)
    print("final metrics:", {k: round(v, 4) for k, v in summary.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
