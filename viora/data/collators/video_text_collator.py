"""Collate variable-length video/text items into padded model batches.

Videos are padded on the temporal axis to the batch max and a **frame-level**
validity mask is produced; it is downsampled to the token grid (``T'`` after
tubelet embedding) so the model masks padded temporal tokens. Text is optionally
tokenized (a ``tokenize_fn`` is injected so the collator stays independent of any
specific tokenizer) with a ``<video>`` placeholder for visual-token injection.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class VideoTextBatch:
    video: torch.Tensor                 # [B, C, Tmax, H, W]
    frame_mask: torch.Tensor            # [B, Tmax] True = real frame
    timestamps: torch.Tensor            # [B, Tmax]
    input_ids: torch.Tensor | None      # [B, L]
    attention_mask: torch.Tensor | None
    labels: torch.Tensor | None
    meta: list[dict]

    def token_temporal_mask(self, tubelet_size: int) -> torch.Tensor:
        """Downsample the frame mask to token temporal positions ``T' = Tmax/tubelet``.

        A token position is valid iff **all** frames in its tubelet are valid.
        """
        b, tmax = self.frame_mask.shape
        usable = (tmax // tubelet_size) * tubelet_size
        m = self.frame_mask[:, :usable].reshape(b, usable // tubelet_size, tubelet_size)
        return m.all(dim=2)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> VideoTextBatch:
        """Move every tensor field to ``device`` (``meta`` is left untouched).

        The collator builds CPU tensors; training on CUDA/MPS needs the batch on the
        model's device before the forward pass, or the first op mismatches CPU input
        against device weights.
        """
        def mv(x: torch.Tensor | None) -> torch.Tensor | None:
            return x.to(device, non_blocking=non_blocking) if x is not None else None

        return VideoTextBatch(
            video=mv(self.video),
            frame_mask=mv(self.frame_mask),
            timestamps=mv(self.timestamps),
            input_ids=mv(self.input_ids),
            attention_mask=mv(self.attention_mask),
            labels=mv(self.labels),
            meta=self.meta,
        )


class VideoTextCollator:
    def __init__(
        self,
        tokenize_fn: Callable[[list[str]], dict] | None = None,
        *,
        pad_value: float = 0.0,
        qa_prob: float = 0.5,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.pad_value = pad_value
        self.qa_prob = qa_prob  # when a video has BOTH captions and qa, chance of rendering QA

    def __call__(self, items: list[dict]) -> VideoTextBatch:
        b = len(items)
        c, _, h, w = items[0]["video"].shape
        tmax = max(it["video"].shape[1] for it in items)

        video = torch.full((b, c, tmax, h, w), self.pad_value)
        frame_mask = torch.zeros(b, tmax, dtype=torch.bool)
        timestamps = torch.zeros(b, tmax)
        for i, it in enumerate(items):
            t = it["video"].shape[1]
            video[i, :, :t] = it["video"]
            frame_mask[i, :t] = True
            ts = it.get("timestamps")
            if ts is not None:
                timestamps[i, :t] = ts

        input_ids = attention_mask = labels = None
        if self.tokenize_fn is not None:
            texts = [self._format_text(it) for it in items]
            tok = self.tokenize_fn(texts)
            input_ids = tok.get("input_ids")
            attention_mask = tok.get("attention_mask")
            labels = tok.get("labels")

        meta = [{k: v for k, v in it.items() if k not in ("video", "timestamps")} for it in items]
        return VideoTextBatch(video, frame_mask, timestamps, input_ids, attention_mask, labels, meta)

    def _format_text(self, it: dict) -> str:
        q, a = it.get("question"), it.get("answer")
        if q and a:
            return f"<video>\nQuestion: {q}\nAnswer: {a}"
        caps = it.get("captions")          # multiple reference captions (e.g. MSR-VTT)
        qa_pairs = it.get("qa")            # multiple (question, answer) pairs (e.g. MSRVTT-QA)
        if caps and qa_pairs:
            if random.random() < self.qa_prob:
                qq, aa = random.choice(qa_pairs)
                return f"<video>\nQuestion: {qq}\nAnswer: {aa}"
            return f"<video>\n{random.choice(caps)}"
        if qa_pairs:
            qq, aa = random.choice(qa_pairs)
            return f"<video>\nQuestion: {qq}\nAnswer: {aa}"
        if caps:
            return f"<video>\n{random.choice(caps)}"  # sample one per view (augmentation)
        # Single-caption datasets, in preference order. PE-Video ships BOTH a
        # human-verified `human_caption` and a machine-generated `model_caption`;
        # prefer the human one. Without these aliases such a dataset silently
        # trains on an empty "<video>\n" target.
        for key in ("caption", "human_caption", "model_caption", "text"):
            val = it.get(key)
            if val:
                return f"<video>\n{val}"
        return "<video>\n"
