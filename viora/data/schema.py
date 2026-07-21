"""Canonical internal data representation.

Video datasets have incompatible schemas; every adapter normalizes into these
typed structures so the rest of the pipeline (collator, model, losses) sees one
shape. Fields are optional where a task does not use them — we never fabricate
absent data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Task(str, Enum):
    """Task types, driving which heads/losses/collation a sample activates."""

    VIDEO_TEXT_ALIGNMENT = "video_text_alignment"
    VIDEO_CAPTIONING = "video_captioning"
    VIDEO_QA = "video_qa"
    TEMPORAL_QA = "temporal_qa"
    CAUSAL_QA = "causal_qa"
    ACTION_RECOGNITION = "action_recognition"
    TEMPORAL_GROUNDING = "temporal_grounding"
    EVENT_ORDERING = "event_ordering"
    DENSE_CAPTIONING = "dense_captioning"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class VideoReference:
    """How to locate the actual pixels — a local path, a URL, or an external id."""

    path: str | None = None
    url: str | None = None
    external_id: str | None = None  # e.g. YouTube id (HowTo100M, Ego4D)

    def is_local(self) -> bool:
        return bool(self.path)


@dataclass
class TextData:
    question: str | None = None
    answer: str | None = None
    caption: str | None = None
    instructions: str | None = None
    options: list[str] = field(default_factory=list)  # multiple-choice QA


@dataclass
class EventSegment:
    start: float  # seconds
    end: float
    text: str | None = None  # dense-caption text for this segment


@dataclass
class TemporalData:
    start_time: float | None = None
    end_time: float | None = None
    event_segments: list[EventSegment] = field(default_factory=list)


@dataclass
class Labels:
    action: str | None = None
    class_id: int | None = None
    retrieval_ids: list[str] = field(default_factory=list)


@dataclass
class VideoSample:
    """One normalized training/eval example."""

    sample_id: str
    dataset_name: str
    task: Task
    video: VideoReference
    duration: float | None = None
    fps: float | None = None
    text: TextData = field(default_factory=TextData)
    temporal: TemporalData = field(default_factory=TemporalData)
    labels: Labels = field(default_factory=Labels)
    metadata: dict = field(default_factory=dict)


class SchemaError(ValueError):
    """Raised by :func:`validate_sample` for structurally invalid samples."""


def validate_sample(s: VideoSample, *, strict: bool = False) -> list[str]:
    """Return a list of problem strings (empty = valid). ``strict`` also flags soft issues."""
    problems: list[str] = []
    if not s.sample_id:
        problems.append("empty sample_id")
    if not (s.video.path or s.video.url or s.video.external_id):
        problems.append("video has no path/url/external_id")
    if s.duration is not None and s.duration <= 0:
        problems.append(f"non-positive duration {s.duration}")
    t = s.temporal
    if t.start_time is not None and t.end_time is not None and t.start_time > t.end_time:
        problems.append(f"start_time {t.start_time} > end_time {t.end_time}")
    for seg in t.event_segments:
        if seg.start > seg.end:
            problems.append(f"segment start {seg.start} > end {seg.end}")

    if s.task == Task.VIDEO_QA and not s.text.question:
        problems.append("VIDEO_QA sample missing question")
    if s.task == Task.TEMPORAL_GROUNDING and (t.start_time is None or t.end_time is None):
        problems.append("TEMPORAL_GROUNDING sample missing start/end time")
    if strict and s.task == Task.VIDEO_CAPTIONING and not s.text.caption:
        problems.append("captioning sample missing caption")
    return problems
