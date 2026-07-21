"""Dataset base classes.

* :class:`AnnotationVideoDataset` — loads a dataset's annotation file(s) into
  :class:`VideoSample` objects via a registered parser, decoding pixels lazily in
  the collator. Reports availability instead of failing silently.
* :class:`SyntheticVideoDataset` — deterministic synthetic samples (random video
  tensors + text) so the data pipeline, collator, and mixture sampler are testable
  with **no downloads**.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset

from viora.data.registry import DEFAULT_REGISTRY, Availability, VioraDatasetRegistry
from viora.data.schema import Task, TextData, VideoReference, VideoSample
from viora.utils.logging import get_logger

logger = get_logger(__name__)


class AnnotationVideoDataset(Dataset):
    """Loads normalized :class:`VideoSample`s from a dataset root; decode happens later."""

    def __init__(
        self,
        name: str,
        root: str | Path,
        split: str = "train",
        *,
        parser: Callable[[dict, str], VideoSample] | None = None,
        registry: VioraDatasetRegistry = DEFAULT_REGISTRY,
    ) -> None:
        self.name = name
        self.root = Path(root)
        self.split = split
        self.spec = registry.get(name)
        self.availability = registry.availability(name, root)
        if self.availability in (Availability.MISSING_FILES,):
            raise FileNotFoundError(
                f"dataset '{name}' unavailable at {root}: {self.availability.value}. "
                f"Expected {self.spec.annotation_files}. {self.spec.license_note}"
            )
        if self.availability in (Availability.REQUIRES_AUTH, Availability.REQUIRES_MANUAL_DOWNLOAD):
            logger.warning("dataset '%s' status=%s: %s", name, self.availability.value, self.spec.license_note)
        self.parser = parser
        self.samples: list[VideoSample] = []  # populated by subclasses / load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]


class SyntheticVideoDataset(Dataset):
    """Deterministic synthetic dataset for tests / smoke runs (no I/O)."""

    def __init__(
        self,
        size: int = 16,
        *,
        num_frames: int = 8,
        image_size: int = 32,
        channels: int = 3,
        fps: float = 4.0,
        task: Task = Task.VIDEO_QA,
        dataset_name: str = "synthetic",
        seed: int = 0,
    ) -> None:
        self.size = size
        self.num_frames = num_frames
        self.image_size = image_size
        self.channels = channels
        self.fps = fps
        self.task = task
        self.dataset_name = dataset_name
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def sample_meta(self, idx: int) -> VideoSample:
        dur = self.num_frames / self.fps
        return VideoSample(
            sample_id=f"{self.dataset_name}-{idx}",
            dataset_name=self.dataset_name,
            task=self.task,
            video=VideoReference(path=f"<synthetic:{idx}>"),
            duration=dur,
            fps=self.fps,
            text=TextData(question="What happens in the video?", answer=f"event {idx}"),
        )

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        t, h, w = self.num_frames, self.image_size, self.image_size
        video = torch.rand(self.channels, t, h, w, generator=g)
        timestamps = torch.arange(t, dtype=torch.float32) / self.fps
        meta = self.sample_meta(idx)
        return {
            "video": video,
            "timestamps": timestamps,
            "sample_id": meta.sample_id,
            "dataset_name": self.dataset_name,
            "task": self.task,
            "question": meta.text.question,
            "answer": meta.text.answer,
        }
