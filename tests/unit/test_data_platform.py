"""Data platform: schema, registry, mixture, collator, preprocessing, synthetic e2e."""

from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader

from viora.data.collators.video_text_collator import VideoTextCollator
from viora.data.datasets.base import SyntheticVideoDataset
from viora.data.mixture.curriculum import STAGES, get_stage
from viora.data.mixture.sampler import TaskAwareMixtureSampler
from viora.data.preprocessing.deduplication import dedup_samples, find_leakage
from viora.data.preprocessing.transforms import VideoTransform
from viora.data.preprocessing.validation import validate_dataset
from viora.data.registry import DEFAULT_REGISTRY, Availability
from viora.data.schema import (
    Task,
    TemporalData,
    TextData,
    VideoReference,
    VideoSample,
    validate_sample,
)


# --------------------------------------------------------------------- schema
def test_schema_validation_flags_problems():
    good = VideoSample("a", "d", Task.VIDEO_QA, VideoReference(path="x.mp4"),
                       text=TextData(question="q?", answer="a"))
    assert validate_sample(good) == []

    bad = VideoSample("", "d", Task.TEMPORAL_GROUNDING, VideoReference(),
                      temporal=TemporalData(start_time=5, end_time=2))
    problems = validate_sample(bad)
    assert any("sample_id" in p for p in problems)
    assert any("no path" in p for p in problems)
    assert any("start_time" in p for p in problems)


# ------------------------------------------------------------------- registry
def test_registry_lists_all_targets():
    names = DEFAULT_REGISTRY.list_datasets()
    for expected in ["webvid", "internvid", "howto100m", "something_something", "msrvtt",
                     "activitynet", "charades_sta", "ego4d", "nextqa", "tgifqa",
                     "msvdqa", "videoinstruct"]:
        assert expected in names


def test_registry_no_fabricated_hf_repos():
    for name in DEFAULT_REGISTRY.list_datasets():
        assert DEFAULT_REGISTRY.get(name).hf_repo is None  # configured, never hardcoded


def test_registry_availability_missing(tmp_path):
    assert DEFAULT_REGISTRY.availability("nextqa", tmp_path) is Availability.MISSING_FILES
    # inherent access rule when no root is given
    assert DEFAULT_REGISTRY.availability("ego4d") is Availability.REQUIRES_AUTH


def test_registry_availability_found(tmp_path):
    (tmp_path / "nextqa_train.csv").write_text("video,question\n")
    assert DEFAULT_REGISTRY.availability("nextqa", tmp_path) is Availability.AVAILABLE


# -------------------------------------------------------------------- mixture
def test_mixture_deterministic_and_distributed():
    sizes = {"a": 100, "b": 50, "c": 20}
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    s1 = TaskAwareMixtureSampler(sizes, weights, seed=7)
    s2 = TaskAwareMixtureSampler(sizes, weights, seed=7)
    assert list(s1) == list(s2)  # deterministic

    # two ranks partition the stream disjointly
    r0 = TaskAwareMixtureSampler(sizes, weights, seed=7, rank=0, world_size=2)
    r1 = TaskAwareMixtureSampler(sizes, weights, seed=7, rank=1, world_size=2)
    assert set(r0) != set(r1) or len(list(r0)) == 0


def test_mixture_temperature_flattens_weights():
    sizes = {"a": 10, "b": 10}
    hot = TaskAwareMixtureSampler(sizes, {"a": 0.9, "b": 0.1}, temperature=1.0).normalized_weights
    warm = TaskAwareMixtureSampler(sizes, {"a": 0.9, "b": 0.1}, temperature=5.0).normalized_weights
    # higher temperature moves the distribution toward uniform
    assert abs(warm["a"] - warm["b"]) < abs(hot["a"] - hot["b"])


def test_mixture_epoch_changes_order():
    s = TaskAwareMixtureSampler({"a": 50}, {"a": 1.0}, seed=1)
    e0 = list(s)
    s.set_epoch(1)
    assert list(s) != e0


# ------------------------------------------------------------------ curriculum
def test_curriculum_stages_present():
    assert len(STAGES) == 5
    assert get_stage(2).name == "video_language_alignment"
    # weights within a stage are positive
    for s in STAGES:
        assert all(w > 0 for w in s.mixture_weights.values())


# -------------------------------------------------------------------- collator
def test_collator_pads_and_masks_variable_lengths():
    items = [
        {"video": torch.randn(3, 8, 32, 32), "timestamps": torch.arange(8).float()},
        {"video": torch.randn(3, 6, 32, 32), "timestamps": torch.arange(6).float()},
    ]
    batch = VideoTextCollator()(items)
    assert batch.video.shape == (2, 3, 8, 32, 32)      # padded to max T=8
    assert batch.frame_mask[0].all() and not batch.frame_mask[1, 6:].any()
    tok_mask = batch.token_temporal_mask(tubelet_size=2)  # T'=4
    assert tok_mask.shape == (2, 4)
    assert tok_mask[1, 3].item() is False  # last tubelet of the short clip is padded


def test_collator_formats_text_variants():
    fmt = VideoTextCollator()._format_text
    assert fmt({"question": "what?", "answer": "a cat"}) == "<video>\nQuestion: what?\nAnswer: a cat"
    assert fmt({"caption": "a dog runs"}) == "<video>\na dog runs"
    assert fmt({}) == "<video>\n"
    # multi-reference captions (MSR-VTT): one is sampled and wrapped
    caps = ["a", "b", "c"]
    out = fmt({"captions": caps})
    assert out.startswith("<video>\n") and out.split("\n", 1)[1] in caps


def test_collator_formats_qa_pairs():
    fmt = VideoTextCollator()._format_text
    out = fmt({"qa": [["what color", "red"], ["how many", "two"]]})
    assert out.startswith("<video>\nQuestion: ") and "\nAnswer: " in out
    # the rendered pair must be one of the two supplied, not a mismatch
    assert out in (
        "<video>\nQuestion: what color\nAnswer: red",
        "<video>\nQuestion: how many\nAnswer: two",
    )


def test_collator_qa_prob_extremes_with_both_present():
    item = {"captions": ["a cat plays"], "qa": [["what animal", "a cat"]]}

    always_qa = VideoTextCollator(qa_prob=1.0)
    for _ in range(20):
        assert always_qa._format_text(item).startswith("<video>\nQuestion: ")

    always_caption = VideoTextCollator(qa_prob=0.0)
    for _ in range(20):
        assert always_caption._format_text(item) == "<video>\na cat plays"


def test_collator_qa_only_ignores_qa_prob():
    # only `qa` present (no captions) -> must always render as QA regardless of qa_prob
    item = {"qa": [["q", "a"]]}
    for prob in (0.0, 0.5, 1.0):
        assert VideoTextCollator(qa_prob=prob)._format_text(item) == "<video>\nQuestion: q\nAnswer: a"


# ------------------------------------------------------------- preprocessing
def test_transform_and_dedup_and_leakage():
    v = VideoTransform(size=16)(torch.rand(3, 4, 40, 30))
    assert v.shape == (3, 4, 16, 16)

    s = lambda i, p: VideoSample(f"s{i}", "d", Task.VIDEO_QA, VideoReference(path=p))  # noqa: E731
    kept, removed = dedup_samples([s(0, "a.mp4"), s(1, "a.mp4"), s(2, "b.mp4")])
    assert len(kept) == 2 and removed == 1

    leak = find_leakage([s(0, "a.mp4")], [s(1, "a.mp4")])
    assert "a.mp4" in leak


def test_validate_dataset_reports_reasons():
    good = VideoSample("a", "d", Task.VIDEO_QA, VideoReference(path="x"), text=TextData(question="q"))
    bad = VideoSample("", "d", Task.VIDEO_QA, VideoReference())
    kept, report = validate_dataset([good, bad])
    assert report.kept == 1 and report.rejected == 1
    assert sum(report.reasons.values()) >= 1


# --------------------------------------------------------- synthetic e2e feed
def test_synthetic_dataset_feeds_dataloader_and_model():
    from tests.unit.test_viora_e2e import tiny_viora_config
    from viora.models.viora import VioraForVideoUnderstanding

    ds = SyntheticVideoDataset(size=6, num_frames=8, image_size=32)
    concat = ConcatDataset([ds])
    sampler = TaskAwareMixtureSampler({"synthetic": len(ds)}, {"synthetic": 1.0}, num_samples=4)
    loader = DataLoader(concat, sampler=sampler, batch_size=2, collate_fn=VideoTextCollator())

    model = VioraForVideoUnderstanding(tiny_viora_config()).eval()
    batch = next(iter(loader))
    tok_mask = batch.token_temporal_mask(tubelet_size=model.cfg.vision.tubelet_size)
    with torch.no_grad():
        out = model(batch.video, temporal_mask=tok_mask, timestamps=batch.timestamps)
    assert out.resampled_tokens.shape[0] == 2


def test_collator_recognizes_single_caption_aliases():
    """PE-Video (and similar HF video datasets) name the caption field
    `human_caption`/`model_caption`, not `caption`. Without these aliases the
    collator silently renders an empty '<video>\\n' target and training learns
    nothing. Human-verified captions win over machine-generated ones."""
    fmt = VideoTextCollator()._format_text
    assert fmt({"human_caption": "a woman blows out candles"}) == "<video>\na woman blows out candles"
    assert fmt({"model_caption": "a machine guess"}) == "<video>\na machine guess"
    # both present -> prefer the human one
    assert fmt({"human_caption": "human", "model_caption": "machine"}) == "<video>\nhuman"
    # explicit `caption` still wins over both
    assert fmt({"caption": "explicit", "human_caption": "human"}) == "<video>\nexplicit"
