"""Real video-caption dataset -> shards adapter (MSR-VTT + folder-sidecar)."""

from __future__ import annotations

import json
import shutil

import pytest
import torch

wds = pytest.importorskip("webdataset")
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg for mp4 encode")

from viora.data.adapters.video_caption import (  # noqa: E402
    build_caption_shards,
    caption_shard_samples,
    index_video_files,
    load_captions_auto,
    parse_caption_records,
    parse_folder_sidecar,
    parse_msrvtt,
)
from viora.data.webdataset_pipeline import (  # noqa: E402
    build_video_text_webdataset,
    tensor_to_mp4_bytes,
)


def _write_videos(dir_, ids):
    dir_.mkdir(parents=True, exist_ok=True)
    for vid in ids:
        (dir_ / f"{vid}.mp4").write_bytes(tensor_to_mp4_bytes(torch.rand(3, 8, 32, 32), fps=4.0))


def test_index_video_files_maps_name_and_stem(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1"])
    idx = index_video_files(tmp_path / "vids")
    assert idx["video0"] == idx["video0.mp4"]          # both forms resolve
    assert len(set(idx.values())) == 2                 # two distinct files


def test_parse_msrvtt_groups_and_filters_split(tmp_path):
    ann = tmp_path / "videodatainfo.json"
    ann.write_text(json.dumps({
        "videos": [{"video_id": "video0", "split": "train"},
                   {"video_id": "video1", "split": "validate"}],
        "sentences": [{"video_id": "video0", "caption": "a cat"},
                      {"video_id": "video0", "caption": "a small cat"},
                      {"video_id": "video1", "caption": "a dog"}],
    }))
    train = parse_msrvtt(ann, split="train")
    assert set(train) == {"video0"} and len(train["video0"]) == 2   # split filter + grouping
    both = parse_msrvtt(ann, split=None)
    assert set(both) == {"video0", "video1"}


def test_parse_folder_sidecar_accepts_str_or_list(tmp_path):
    ann = tmp_path / "caps.json"
    ann.write_text(json.dumps({"a": "one caption", "b": ["two", "captions"]}))
    caps = parse_folder_sidecar(ann)
    assert caps["a"] == ["one caption"] and caps["b"] == ["two", "captions"]


def test_caption_samples_skip_missing_videos(tmp_path):
    _write_videos(tmp_path / "vids", ["video0"])          # only video0 exists
    idx = index_video_files(tmp_path / "vids")
    caps = {"video0": ["c0"], "video99": ["missing"]}
    out = list(caption_shard_samples(caps, idx))
    assert len(out) == 1 and out[0]["meta"]["video_id"] == "video0"
    assert out[0]["meta"]["captions"] == ["c0"] and out[0]["video_bytes"]


def test_build_caption_shards_end_to_end_msrvtt(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1"])
    ann = tmp_path / "videodatainfo.json"
    ann.write_text(json.dumps({
        "videos": [{"video_id": "video0", "split": "train"},
                   {"video_id": "video1", "split": "train"}],
        "sentences": [{"video_id": "video0", "caption": "a cat plays"},
                      {"video_id": "video1", "caption": "a dog runs"}],
    }))
    pattern = str(tmp_path / "msrvtt-train-%06d.tar")
    n = build_caption_shards(tmp_path / "vids", ann, pattern, fmt="msrvtt", split="train")
    assert n == 2

    ds = build_video_text_webdataset(
        str(tmp_path / "msrvtt-train-000000.tar"), num_frames=8, resampled=False, shuffle=0
    )
    items = list(iter(ds))
    assert len(items) == 2
    assert items[0]["video"].shape[:2] == (3, 8)          # decoded [C=3, T=8, ...]
    assert "captions" in items[0]                          # meta carries reference captions


def test_parse_msrvtt_rejects_top_level_list(tmp_path):
    # MSRVTT-QA train_qa.json is a bare list, not a videodatainfo object -> clear error, not AttributeError
    ann = tmp_path / "train_qa.json"
    ann.write_text(json.dumps([{"question": "q", "answer": "a", "video_id": "video0"}]))
    with pytest.raises(ValueError, match="top-level JSON"):
        parse_msrvtt(ann)
    with pytest.raises(ValueError, match="top-level JSON"):
        parse_msrvtt(ann, split="train")


def test_parse_msrvtt_empty_split_lists_present_splits(tmp_path):
    ann = tmp_path / "test_videodatainfo.json"
    ann.write_text(json.dumps({
        "videos": [{"video_id": "video0", "split": "test"}],
        "sentences": [{"video_id": "video0", "caption": "x"}],
    }))
    with pytest.raises(ValueError, match="Splits present"):
        parse_msrvtt(ann, split="train")   # requested split absent -> explicit error


def test_build_caption_shards_bad_ids_write_zero(tmp_path):
    _write_videos(tmp_path / "vids", ["video0"])
    ann = tmp_path / "caps.json"
    ann.write_text(json.dumps({"does_not_exist": "nope"}))   # id has no matching file
    n = build_caption_shards(tmp_path / "vids", ann, str(tmp_path / "t-%06d.tar"), fmt="folder")
    assert n == 0


def test_parse_caption_records_groups_expanded_list(tmp_path):
    # friedrichor/MSR-VTT shape: one row per caption, video_id repeated
    ann = tmp_path / "msrvtt_train_7k.json"
    ann.write_text(json.dumps([
        {"video_id": "video0", "caption": "a cat"},
        {"video_id": "video0", "caption": "a small cat"},
        {"video_id": "video1", "caption": "a dog"},
    ]))
    caps = parse_caption_records(ann)
    assert caps["video0"] == ["a cat", "a small cat"] and caps["video1"] == ["a dog"]


def test_load_captions_auto_dispatches_by_shape(tmp_path):
    lst = tmp_path / "records.json"
    lst.write_text(json.dumps([{"video_id": "v0", "caption": "c"}]))
    assert load_captions_auto(lst) == {"v0": ["c"]}

    vdi = tmp_path / "videodatainfo.json"
    vdi.write_text(json.dumps({"videos": [{"video_id": "v0", "split": "train"}],
                               "sentences": [{"video_id": "v0", "caption": "c"}]}))
    assert load_captions_auto(vdi, split="train") == {"v0": ["c"]}

    side = tmp_path / "sidecar.json"
    side.write_text(json.dumps({"v0": ["c1", "c2"]}))
    assert load_captions_auto(side) == {"v0": ["c1", "c2"]}


def test_resolve_video_handles_numeric_ids(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1"])
    idx = index_video_files(tmp_path / "vids")
    # a records file keyed by bare integers still resolves to videoN.mp4
    caps = {"0": ["c0"], "1": ["c1"]}
    out = list(caption_shard_samples(caps, idx))
    assert {s["meta"]["video_id"] for s in out} == {"0", "1"} and len(out) == 2


def test_build_caption_shards_auto_records_end_to_end(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1"])
    ann = tmp_path / "records.json"
    ann.write_text(json.dumps([
        {"video_id": "video0", "caption": "a cat plays"},
        {"video_id": "video1", "caption": "a dog runs"},
    ]))
    n = build_caption_shards(tmp_path / "vids", ann, str(tmp_path / "r-%06d.tar"), fmt="auto")
    assert n == 2
