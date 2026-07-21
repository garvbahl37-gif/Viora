"""Curriculum: staged training configuration.

Each stage names its datasets, active tasks/objectives, and a suggested mixture.
These are *scaffolding and example ratios*, not scientifically-optimal defaults —
they mirror the milestone plan and are overridable in YAML. Not every stage must
run to exercise the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from viora.data.schema import Task


@dataclass
class CurriculumStage:
    index: int
    name: str
    datasets: list[str]
    tasks: list[Task]
    objectives: list[str]          # loss component names active this stage
    mixture_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""


STAGES: list[CurriculumStage] = [
    CurriculumStage(
        1, "visual_temporal_pretraining", ["something_something"],
        [Task.ACTION_RECOGNITION],
        ["classification", "masked_video"],
        {"something_something": 1.0},
        "Motion / temporal representation; optional self-supervision.",
    ),
    CurriculumStage(
        2, "video_language_alignment", ["webvid", "internvid", "msrvtt"],
        [Task.VIDEO_TEXT_ALIGNMENT, Task.VIDEO_CAPTIONING],
        ["contrastive", "matching", "lm"],
        {"webvid": 0.4, "internvid": 0.4, "msrvtt": 0.2},
        "Contrastive alignment, video-text matching, captioning.",
    ),
    CurriculumStage(
        3, "long_temporal_understanding", ["activitynet", "charades_sta", "ego4d", "howto100m"],
        [Task.TEMPORAL_GROUNDING, Task.DENSE_CAPTIONING],
        ["grounding", "lm", "order"],
        {"activitynet": 0.35, "charades_sta": 0.25, "ego4d": 0.2, "howto100m": 0.2},
        "Temporal grounding, event representation, long-context memory.",
    ),
    CurriculumStage(
        4, "video_reasoning", ["nextqa", "tgifqa", "msvdqa"],
        [Task.VIDEO_QA, Task.CAUSAL_QA, Task.TEMPORAL_QA],
        ["lm"],
        {"nextqa": 0.5, "tgifqa": 0.25, "msvdqa": 0.25},
        "VideoQA, temporal & causal reasoning.",
    ),
    CurriculumStage(
        5, "instruction_tuning", ["videoinstruct"],
        [Task.INSTRUCTION_FOLLOWING],
        ["lm"],
        {"videoinstruct": 1.0},
        "Instruction tuning on curated conversational data.",
    ),
]


def get_stage(index: int) -> CurriculumStage:
    for s in STAGES:
        if s.index == index:
            return s
    raise KeyError(f"no curriculum stage {index} (have {[s.index for s in STAGES]})")
