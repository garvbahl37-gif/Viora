"""Dataset registry.

Registers a declarative :class:`DatasetSpec` per target dataset and reports, per
dataset, whether it can actually be used from a given data root. This is a
deliberate engineering choice over 12 near-identical adapter files: the datasets
differ almost entirely in *metadata and access rules*, not code, so one
data-driven registry + a shared annotation loader (``datasets/base.py``) is
clearer and less error-prone.

Crucially: presence on Hugging Face does NOT mean raw videos are downloadable.
Many of these ship annotations only, external video ids, or require a signed
licence. HF repository ids are intentionally NOT hardcoded here — set them in
config so they are easy to replace and never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from viora.data.schema import Task


class Availability(str, Enum):
    AVAILABLE = "available"                       # annotations + resolvable videos present
    ANNOTATIONS_ONLY = "annotations_only"         # labels present, pixels must be fetched
    REQUIRES_MANUAL_DOWNLOAD = "requires_manual_download"
    REQUIRES_AUTH = "requires_auth"               # licence / login gated
    MISSING_FILES = "missing_files"               # nothing found at the given root


class VideoSource(str, Enum):
    LOCAL = "local"          # video files expected on disk
    URL = "url"              # download URLs in annotations (may be dead)
    EXTERNAL_ID = "external_id"  # YouTube/Ego id — fetch yourself


@dataclass
class DatasetSpec:
    name: str
    tasks: list[Task]
    video_source: VideoSource
    default_access: Availability          # inherent access rule, before checking disk
    annotation_files: list[str]           # expected relative to the dataset root
    fmt: str                              # human description of the annotation format
    approx_scale: str                     # approximate size (verify — not authoritative)
    license_note: str
    splits: list[str] = field(default_factory=lambda: ["train", "val"])
    hf_repo: str | None = None            # set via config; never fabricated here
    notes: str = ""


# --------------------------------------------------------------------------- #
# Target datasets. Descriptions are informational; verify scale/licence before  #
# use, and configure hf_repo / paths yourself.                                  #
# --------------------------------------------------------------------------- #
_SPECS: dict[str, DatasetSpec] = {
    "webvid": DatasetSpec(
        "webvid", [Task.VIDEO_TEXT_ALIGNMENT, Task.VIDEO_CAPTIONING], VideoSource.URL,
        Availability.REQUIRES_MANUAL_DOWNLOAD, ["webvid_{split}.csv"],
        "CSV: videoid, url (mp4), caption, duration.",
        "~2.5M (WebVid-2M) / ~10M (WebVid-10M) clip-caption pairs",
        "Research-only; the original distribution was withdrawn — verify licence and source.",
    ),
    "internvid": DatasetSpec(
        "internvid", [Task.VIDEO_TEXT_ALIGNMENT, Task.VIDEO_CAPTIONING], VideoSource.EXTERNAL_ID,
        Availability.ANNOTATIONS_ONLY, ["internvid_{split}.jsonl"],
        "JSONL: youtube_id, start, end, caption.",
        "~7M videos / ~234M clips (InternVid-10M/full)",
        "Annotations released; videos are YouTube ids you must fetch. Respect YouTube ToS.",
    ),
    "howto100m": DatasetSpec(
        "howto100m", [Task.VIDEO_TEXT_ALIGNMENT, Task.VIDEO_CAPTIONING], VideoSource.EXTERNAL_ID,
        Availability.ANNOTATIONS_ONLY, ["howto100m_captions.csv"],
        "CSV/JSON: youtube_id, ASR caption segments with start/end.",
        "~1.2M YouTube videos, ~136M clips",
        "ASR captions provided; videos are YouTube ids. Noisy narration alignment.",
    ),
    "something_something": DatasetSpec(
        "something_something", [Task.ACTION_RECOGNITION], VideoSource.LOCAL,
        Availability.REQUIRES_AUTH, ["something-something-v2-{split}.json", "labels.json"],
        "JSON: id, template/label; webm videos on disk.",
        "~220k clips, 174 action classes (V2)",
        "Requires registration/login to download the videos. Non-commercial licence.",
    ),
    "msrvtt": DatasetSpec(
        "msrvtt", [Task.VIDEO_CAPTIONING, Task.VIDEO_TEXT_ALIGNMENT, Task.VIDEO_QA], VideoSource.LOCAL,
        Availability.REQUIRES_MANUAL_DOWNLOAD, ["msrvtt_{split}.json"],
        "JSON: video_id, captions[], (QA variants add question/answer).",
        "~10k clips, ~200k captions",
        "Videos distributed as a zip; commonly used research benchmark.",
    ),
    "activitynet": DatasetSpec(
        "activitynet", [Task.DENSE_CAPTIONING, Task.TEMPORAL_GROUNDING], VideoSource.EXTERNAL_ID,
        Availability.ANNOTATIONS_ONLY, ["activitynet_captions_{split}.json"],
        "JSON: video_id -> {duration, timestamps[[s,e]], sentences[]}.",
        "~20k videos, ~100k temporally-localised sentences",
        "Captions/timestamps provided; videos are YouTube ids (some unavailable).",
    ),
    "charades_sta": DatasetSpec(
        "charades_sta", [Task.TEMPORAL_GROUNDING], VideoSource.LOCAL,
        Availability.REQUIRES_MANUAL_DOWNLOAD, ["charades_sta_{split}.txt", "Charades_v1_480/"],
        "TXT: 'vid start end ## sentence' moment-localisation triples.",
        "~10k videos, ~16k moment annotations",
        "Charades videos require separate download; non-commercial licence.",
    ),
    "ego4d": DatasetSpec(
        "ego4d", [Task.TEMPORAL_QA, Task.TEMPORAL_GROUNDING, Task.DENSE_CAPTIONING], VideoSource.LOCAL,
        Availability.REQUIRES_AUTH, ["ego4d_{split}.json"],
        "JSON: canonical Ego4D annotation schema (NLQ/MQ/narrations).",
        "~3,670 hours egocentric video",
        "Signed licence agreement required; download via the official Ego4D CLI.",
    ),
    "nextqa": DatasetSpec(
        "nextqa", [Task.VIDEO_QA, Task.CAUSAL_QA, Task.TEMPORAL_QA], VideoSource.LOCAL,
        Availability.REQUIRES_MANUAL_DOWNLOAD, ["nextqa_{split}.csv"],
        "CSV: video, question, a0..a4, answer (idx); causal/temporal/descriptive types.",
        "~5.4k videos, ~52k multiple-choice QA",
        "Videos sourced from VidOR; download separately.",
    ),
    "tgifqa": DatasetSpec(
        "tgifqa", [Task.VIDEO_QA], VideoSource.LOCAL,
        Availability.REQUIRES_MANUAL_DOWNLOAD, ["tgif_qa_{split}.json"],
        "JSON: gif_name, question, answer; action/transition/count/frameqa tasks.",
        "~72k GIFs, ~165k QA",
        "TGIF GIFs distributed separately; convert GIF->mp4 in preprocessing.",
    ),
    "msvdqa": DatasetSpec(
        "msvdqa", [Task.VIDEO_QA], VideoSource.LOCAL,
        Availability.REQUIRES_MANUAL_DOWNLOAD, ["msvd_qa_{split}.json", "msvd_mapping.json"],
        "JSON: video_id, question, answer (open-ended).",
        "~1.9k videos, ~50k QA",
        "Built on MSVD clips; download the MSVD videos separately.",
    ),
    "videoinstruct": DatasetSpec(
        "videoinstruct", [Task.INSTRUCTION_FOLLOWING, Task.VIDEO_QA], VideoSource.LOCAL,
        Availability.ANNOTATIONS_ONLY, ["video_instruct_{split}.json"],
        "JSON: video, conversations[{from, value}] instruction-tuning turns.",
        "~100k instruction pairs (VideoInstruct-100K-style)",
        "Instruction annotations reference source videos (often ActivityNet) — fetch those.",
    ),
}


class VioraDatasetRegistry:
    """Registry of dataset specs with lazy, path-aware availability checks."""

    def __init__(self) -> None:
        self._specs: dict[str, DatasetSpec] = dict(_SPECS)

    def register(self, spec: DatasetSpec, *, overwrite: bool = False) -> None:
        if spec.name in self._specs and not overwrite:
            raise ValueError(f"dataset '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> DatasetSpec:
        if name not in self._specs:
            raise KeyError(f"unknown dataset '{name}' (have: {sorted(self._specs)})")
        return self._specs[name]

    def list_datasets(self) -> list[str]:
        return sorted(self._specs)

    def tasks_for(self, name: str) -> list[Task]:
        return self.get(name).tasks

    def availability(self, name: str, root: str | Path | None = None) -> Availability:
        """Check disk under ``root`` and combine with the dataset's inherent access rule."""
        spec = self.get(name)
        if root is None:
            return spec.default_access
        root = Path(root)
        if not root.exists():
            return Availability.MISSING_FILES
        found_any = False
        for split in spec.splits:
            for pattern in spec.annotation_files:
                if list(root.glob(pattern.replace("{split}", split))) or (root / pattern.replace("{split}", split)).exists():
                    found_any = True
        if not found_any:
            return Availability.MISSING_FILES
        # annotations present -> success depends on whether pixels are local
        if spec.video_source is VideoSource.LOCAL:
            return Availability.AVAILABLE
        return Availability.ANNOTATIONS_ONLY


DEFAULT_REGISTRY = VioraDatasetRegistry()
