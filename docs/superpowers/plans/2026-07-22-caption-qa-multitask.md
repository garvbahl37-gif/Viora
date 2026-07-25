# Caption + Q&A Multi-Task Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Viora's pragmatic model (frozen SigLIP + frozen Qwen2.5-0.5B + LoRA + bridge) to both **caption** clips and **answer questions** about them, by resuming from the existing 5k-step MSR-VTT caption checkpoint and continuing on a combined caption+QA shard set, entirely on a free Kaggle T4.

**Architecture:** Each MSR-VTT video is stored **once** in a shard, carrying both its reference captions and its MSRVTT-QA question/answer pairs in metadata. The collator randomly renders each training view as either a caption or a `Question:/Answer:` example (default 50/50), so one data stream trains both tasks with no change to the model or training loop. Cross-session continuity goes through a HuggingFace model repo: push the latest checkpoint at the end of a session, pull it at the start of the next to resume via `training.resume=`.

**Tech Stack:** PyTorch, `webdataset` (sharded storage), `huggingface_hub` (checkpoint relay + annotation fetch), pytest, ruff.

## Global Constraints

- Free compute only — Kaggle T4, no paid GPU (from spec: "Compute" decision).
- Keep `Qwen2.5-0.5B-Instruct` — no LLM upsize this round (from spec: "Compute" decision).
- No new datasets beyond MSR-VTT captions + MSRVTT-QA — both reuse the **same already-downloaded videos**, no new video downloads (spec Non-goals).
- No grounding/temporal-evidence supervision — MSR-VTT has no timestamp spans (spec Non-goals).
- Resume from `final.pt` (HF repo `bharatverse11/viora-msrvtt`) via `training.resume=` — do not restart from step 0 (spec "Training design").
- Default caption:QA mix ratio is **50/50**, configurable via a `qa_prob` parameter (spec "Collator mix").
- Target **~15–25k additional steps** (aim ~20k) across 2–4 free Kaggle sessions (spec "Training design" / "Multi-session structure").
- Every new function ships with a CPU-only, no-network unit test (spec "Testing").
- Every existing test must keep passing; full suite + ruff must stay green before each commit (repo convention, see `git log`).
- Git identity for all commits: `garvbahl37@gmail.com` / `garvbahl37-gif`, **no** `Co-Authored-By` trailer (durable project instruction).

---

### Task 1: MSRVTT-QA record parser

**Files:**
- Modify: `viora/data/adapters/video_caption.py`
- Test: `tests/unit/test_video_caption_adapter.py`

**Interfaces:**
- Consumes: nothing new (uses existing `_ID_KEYS` tuple already defined in this file at module scope).
- Produces: `parse_qa_records(annotations: str | Path) -> dict[str, list[tuple[str, str]]]` — later tasks (2, 4) import and call this.

MSRVTT-QA ships as a top-level JSON **list** of records like
`{"id": 123, "video_id": "video0", "question": "what is the man doing", "answer": "singing"}`
(key spelling varies by mirror — sometimes `"q"`/`"a"` instead of `"question"`/`"answer"`). This task adds a parser that auto-detects both the id key (reusing the existing `_ID_KEYS` tuple) and the question/answer key pair, and groups multiple QA pairs per video.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_video_caption_adapter.py` (near the other `parse_*` tests):

```python
def test_parse_qa_records_groups_by_video_and_detects_keys(tmp_path):
    ann = tmp_path / "train_qa.json"
    ann.write_text(json.dumps([
        {"id": 1, "video_id": "video0", "question": "what is the man doing", "answer": "singing"},
        {"id": 2, "video_id": "video0", "question": "how many people", "answer": "one"},
        {"id": 3, "video_id": "video1", "question": "what color is the car", "answer": "red"},
    ]))
    qa = parse_qa_records(ann)
    assert qa["video0"] == [("what is the man doing", "singing"), ("how many people", "one")]
    assert qa["video1"] == [("what color is the car", "red")]


def test_parse_qa_records_accepts_q_a_key_spelling(tmp_path):
    ann = tmp_path / "alt_qa.json"
    ann.write_text(json.dumps([{"video_id": "video0", "q": "who is this", "a": "a man"}]))
    assert parse_qa_records(ann) == {"video0": [("who is this", "a man")]}


def test_parse_qa_records_rejects_top_level_object(tmp_path):
    ann = tmp_path / "not_a_list.json"
    ann.write_text(json.dumps({"video0": "not a qa record"}))
    with pytest.raises(ValueError, match="top-level JSON list"):
        parse_qa_records(ann)


def test_parse_qa_records_skips_incomplete_records(tmp_path):
    ann = tmp_path / "qa.json"
    ann.write_text(json.dumps([
        {"video_id": "video0", "question": "q only, no answer"},
        {"question": "no video id", "answer": "a"},
        {"video_id": "video1", "question": "complete", "answer": "yes"},
    ]))
    assert parse_qa_records(ann) == {"video1": [("complete", "yes")]}
```

Add the import at the top of the test file (alongside the existing adapter imports):

```python
from viora.data.adapters.video_caption import (  # noqa: E402
    build_caption_shards,
    caption_shard_samples,
    index_video_files,
    load_captions_auto,
    parse_caption_records,
    parse_folder_sidecar,
    parse_msrvtt,
    parse_qa_records,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_video_caption_adapter.py -k parse_qa_records -v`
Expected: FAIL with `ImportError: cannot import name 'parse_qa_records'`

- [ ] **Step 3: Implement `parse_qa_records`**

In `viora/data/adapters/video_caption.py`, add below `parse_caption_records` (after its closing `return dict(caps)` at the current end of that function, before `load_captions_auto`):

```python
_QA_KEY_PAIRS: tuple[tuple[str, str], ...] = (("question", "answer"), ("q", "a"), ("Q", "A"))


def parse_qa_records(annotations: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Parse a top-level LIST of QA records into ``video_id -> [(question, answer), ...]``.

    Handles the standard MSRVTT-QA shape (one row per QA pair), e.g.
    ``[{"video_id": "video0", "question": "...", "answer": "..."}, ...]``. The id
    key is auto-detected via ``_ID_KEYS``; the question/answer key pair is
    auto-detected via ``_QA_KEY_PAIRS`` (some mirrors use ``q``/``a``). Records
    missing an id, a question, or an answer are skipped.
    """
    data = json.loads(Path(annotations).read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"MSRVTT-QA --qa-annotations expects a top-level JSON list of "
            f"{{video_id, question, answer}} records; got {type(data).__name__}."
        )
    qa: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in data:
        if not isinstance(rec, Mapping):
            continue
        vid = next((str(rec[k]) for k in _ID_KEYS if rec.get(k) is not None), None)
        if vid is None:
            continue
        for qk, ak in _QA_KEY_PAIRS:
            q, a = rec.get(qk), rec.get(ak)
            if q and a:
                qa[vid].append((str(q), str(a)))
                break
    return dict(qa)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_video_caption_adapter.py -k parse_qa_records -v`
Expected: 4 passed

- [ ] **Step 5: Full adapter test file + lint**

Run: `.venv/bin/python -m pytest tests/unit/test_video_caption_adapter.py -q && .venv/bin/ruff check viora/data/adapters/ tests/unit/test_video_caption_adapter.py`
Expected: all pass, "All checks passed!"

- [ ] **Step 6: Commit**

```bash
git add viora/data/adapters/video_caption.py tests/unit/test_video_caption_adapter.py
git commit -m "data: add MSRVTT-QA record parser (parse_qa_records)"
```

---

### Task 2: Merged caption+QA shard builder

**Files:**
- Modify: `viora/data/adapters/video_caption.py`
- Test: `tests/unit/test_video_caption_adapter.py`

**Interfaces:**
- Consumes: `parse_qa_records` (Task 1), `load_captions_auto`/`parse_msrvtt`/`parse_caption_records`/`parse_folder_sidecar` (existing), `index_video_files` and `_resolve_video` (existing), `write_video_text_shards` (existing, in `viora/data/webdataset_pipeline.py`).
- Produces: `merged_shard_samples(captions, qa, video_index, *, limit=None) -> Iterator[dict]` and `build_caption_qa_shards(videos_dir, out_pattern, *, captions_path=None, captions_fmt="auto", qa_path=None, split=None, limit=None, maxcount=500) -> int` — Task 4 (CLI) and Task 7 (notebook) call `build_caption_qa_shards`.

Each video is written **once** to the shard, with `captions` (if any captions were parsed for it) and/or `qa` (if any QA pairs were parsed for it) in its metadata. A video with neither is skipped (nothing to train on). This is the merge that lets one shard set carry both tasks.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_video_caption_adapter.py`:

```python
def test_merged_shard_samples_combines_both_and_skips_neither(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1", "video2"])
    idx = index_video_files(tmp_path / "vids")
    captions = {"video0": ["a cat plays"], "video1": ["a dog runs"]}
    qa = {"video0": [("what animal", "a cat")], "video2": [("what color", "blue")]}
    out = {s["meta"]["video_id"]: s["meta"] for s in merged_shard_samples(captions, qa, idx)}
    assert set(out) == {"video0", "video1", "video2"}
    assert out["video0"]["captions"] == ["a cat plays"] and out["video0"]["qa"] == [["what animal", "a cat"]]
    assert out["video1"]["captions"] == ["a dog runs"] and "qa" not in out["video1"]
    assert out["video2"]["qa"] == [["what color", "blue"]] and "captions" not in out["video2"]


def test_merged_shard_samples_skips_video_with_neither(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1"])
    idx = index_video_files(tmp_path / "vids")
    # video1 has a file but no captions/qa at all -> must not appear
    out = list(merged_shard_samples({"video0": ["cap"]}, {}, idx))
    assert len(out) == 1 and out[0]["meta"]["video_id"] == "video0"


def test_build_caption_qa_shards_end_to_end(tmp_path):
    _write_videos(tmp_path / "vids", ["video0", "video1"])
    cap_ann = tmp_path / "captions.json"
    cap_ann.write_text(json.dumps([
        {"video_id": "video0", "caption": "a cat plays"},
        {"video_id": "video1", "caption": "a dog runs"},
    ]))
    qa_ann = tmp_path / "qa.json"
    qa_ann.write_text(json.dumps([
        {"video_id": "video0", "question": "what animal", "answer": "a cat"},
    ]))
    n = build_caption_qa_shards(
        tmp_path / "vids", str(tmp_path / "combined-%06d.tar"),
        captions_path=cap_ann, captions_fmt="records", qa_path=qa_ann,
    )
    assert n == 2

    ds = build_video_text_webdataset(
        str(tmp_path / "combined-000000.tar"), num_frames=8, resampled=False, shuffle=0
    )
    items = {it["video_id"]: it for it in iter(ds)}
    assert items["video0"]["captions"] == ["a cat plays"]
    assert items["video0"]["qa"] == [["what animal", "a cat"]]
    assert "qa" not in items["video1"]


def test_build_caption_qa_shards_qa_only(tmp_path):
    # captions_path omitted -> QA-only shards (still valid: collator falls back to QA)
    _write_videos(tmp_path / "vids", ["video0"])
    qa_ann = tmp_path / "qa.json"
    qa_ann.write_text(json.dumps([{"video_id": "video0", "question": "q", "answer": "a"}]))
    n = build_caption_qa_shards(tmp_path / "vids", str(tmp_path / "qaonly-%06d.tar"), qa_path=qa_ann)
    assert n == 1
```

Add `merged_shard_samples` and `build_caption_qa_shards` to the test file's import block (extend the same `from viora.data.adapters.video_caption import (...)` statement added in Task 1), and add this import alongside the existing `build_video_text_webdataset` import:

```python
from viora.data.webdataset_pipeline import (  # noqa: E402
    build_video_text_webdataset,
    tensor_to_mp4_bytes,
)
```
(this import already exists in the file from earlier work — just confirm `build_video_text_webdataset` is present, no change needed there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_video_caption_adapter.py -k "merged_shard_samples or build_caption_qa_shards" -v`
Expected: FAIL with `ImportError: cannot import name 'merged_shard_samples'`

- [ ] **Step 3: Implement `merged_shard_samples` and `build_caption_qa_shards`**

In `viora/data/adapters/video_caption.py`, add after `caption_shard_samples` (which currently ends right before `def build_caption_shards`):

```python
def merged_shard_samples(
    captions: dict[str, list[str]],
    qa: dict[str, list[tuple[str, str]]],
    video_index: dict[str, Path],
    *,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield one shard sample per video carrying whichever of captions/QA it has.

    A video with neither captions nor QA pairs is skipped (nothing to supervise).
    Storing each video once — regardless of how many captions/QA pairs it has —
    is what keeps a combined caption+QA shard set the same size as a captions-only
    one; the collator samples one caption or one QA pair per training view.
    """
    missing = 0
    emitted = 0
    for vid in {**captions, **qa}:
        if limit is not None and emitted >= limit:
            break
        caps, pairs = captions.get(vid), qa.get(vid)
        if not caps and not pairs:
            continue
        path = _resolve_video(vid, video_index)
        if path is None:
            missing += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        if not data:
            missing += 1
            continue
        meta: dict = {"video_id": vid}
        if caps:
            meta["captions"] = caps
        if pairs:
            meta["qa"] = [list(p) for p in pairs]
        yield {"video_bytes": data, "meta": meta}
        emitted += 1
    if missing:
        logger.warning("%d annotated clips had no readable video file (skipped)", missing)


def build_caption_qa_shards(
    videos_dir: str | Path,
    out_pattern: str,
    *,
    captions_path: str | Path | None = None,
    captions_fmt: str = "auto",
    qa_path: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    maxcount: int = 500,
) -> int:
    """Build ONE shard set combining captions and MSRVTT-QA (each video stored once).

    At least one of ``captions_path`` / ``qa_path`` must be given. ``captions_fmt``
    matches :func:`build_caption_shards`'s ``fmt`` (``auto``/``msrvtt``/``records``/
    ``folder``); QA is always the MSRVTT-QA list format (:func:`parse_qa_records`).
    Returns the number of videos written.
    """
    if captions_path is None and qa_path is None:
        raise ValueError("build_caption_qa_shards needs captions_path and/or qa_path")

    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"--videos '{videos_dir}' is not a directory")
    video_index = index_video_files(videos_dir)
    if not video_index:
        raise FileNotFoundError(f"no video files ({', '.join(VIDEO_EXTS)}) found under {videos_dir}")

    captions: dict[str, list[str]] = {}
    if captions_path is not None:
        if captions_fmt == "auto":
            captions = load_captions_auto(captions_path, split)
        elif captions_fmt == "msrvtt":
            captions = parse_msrvtt(captions_path, split)
        elif captions_fmt == "records":
            captions = parse_caption_records(captions_path)
        elif captions_fmt == "folder":
            captions = parse_folder_sidecar(captions_path)
        else:
            raise ValueError(f"unknown captions_fmt '{captions_fmt}' (expected auto|msrvtt|records|folder)")

    qa: dict[str, list[tuple[str, str]]] = parse_qa_records(qa_path) if qa_path is not None else {}
    if not captions and not qa:
        raise ValueError("no captions and no QA pairs parsed (check paths/format/split)")

    from viora.data.webdataset_pipeline import write_video_text_shards

    logger.info(
        "%d captioned + %d QA-annotated videos -> writing combined shards",
        len(captions), len(qa),
    )
    samples = merged_shard_samples(captions, qa, video_index, limit=limit)
    return write_video_text_shards(samples, out_pattern, maxcount=maxcount)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_video_caption_adapter.py -q`
Expected: all pass (existing + new)

- [ ] **Step 5: Full suite + lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check viora/ tests/`
Expected: all pass, "All checks passed!"

- [ ] **Step 6: Commit**

```bash
git add viora/data/adapters/video_caption.py tests/unit/test_video_caption_adapter.py
git commit -m "data: merged caption+QA shard builder (build_caption_qa_shards)"
```

---

### Task 3: Collator caption/QA mixing

**Files:**
- Modify: `viora/data/collators/video_text_collator.py`
- Test: `tests/unit/test_data_platform.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `VideoTextCollator(tokenize_fn=None, *, pad_value=0.0, qa_prob=0.5)` — `qa_prob` is a new keyword-only constructor arg; `collator.qa_prob` is a settable instance attribute (Task 6 sets it from a CLI flag).

The collator already special-cases `question`/`answer` singular keys (used by the earlier synthetic-QA smoke path) and `captions` lists (used by MSR-VTT). This task adds a `qa` list (list of `[question, answer]` pairs, as produced by Task 2's shards) and a `qa_prob` coin-flip so a video carrying both `captions` and `qa` renders as either format at random, per view.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_data_platform.py`, right after the existing `test_collator_formats_text_variants`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_data_platform.py -k "qa_pairs or qa_prob or qa_only" -v`
Expected: FAIL — `_format_text` currently ignores the `qa` key and there is no `qa_prob` constructor arg (`TypeError: unexpected keyword argument 'qa_prob'`)

- [ ] **Step 3: Implement the mix**

In `viora/data/collators/video_text_collator.py`, replace the `VideoTextCollator.__init__` and `_format_text`:

```python
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
```

(the `__call__` method is unchanged — it already calls `self._format_text(it)`; only the `@staticmethod`-decorated method below changes)

Replace the existing `_format_text` (currently a `@staticmethod`) with an instance method:

```python
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
        if it.get("caption"):
            return f"<video>\n{it['caption']}"
        return "<video>\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_data_platform.py -q`
Expected: all pass (existing collator tests + new)

- [ ] **Step 5: Full suite + lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check viora/ tests/`
Expected: all pass, "All checks passed!"

- [ ] **Step 6: Commit**

```bash
git add viora/data/collators/video_text_collator.py tests/unit/test_data_platform.py
git commit -m "data: collator mixes captions and QA pairs per view (qa_prob, default 0.5)"
```

---

### Task 4: CLI support for combined shard building

**Files:**
- Modify: `scripts/prepare_video_dataset.py`
- Modify: `docs/PRODUCTION.md`

**Interfaces:**
- Consumes: `build_caption_qa_shards` (Task 2), existing `build_caption_shards`.
- Produces: no new Python interface — this is the CLI entry point Task 7's notebook cell shells out to.

- [ ] **Step 1: Modify the CLI**

Replace the body of `scripts/prepare_video_dataset.py`'s argument list and `main()` (the whole file is short; here is the full replacement):

```python
#!/usr/bin/env python3
"""Build Viora training shards from a real video-caption (+ optional QA) dataset.

MSR-VTT captions only (videos dir + videodatainfo.json)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --annotations /kaggle/input/msrvtt/train_val_videodatainfo.json \
        --format msrvtt --split train \
        --out data/shards/msrvtt-train-%06d.tar

Any dataset (folder of videos + a {id: caption(s)} JSON sidecar; e.g. MSVD)::

    python scripts/prepare_video_dataset.py \
        --videos /path/to/videos --annotations captions.json \
        --format folder --out data/shards/train-%06d.tar

Captions + MSRVTT-QA COMBINED (one shard set, both tasks; add --qa-annotations)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --annotations /kaggle/input/msrvtt/train_val_videodatainfo.json \
        --qa-annotations /path/to/train_qa.json \
        --format msrvtt --split train \
        --out data/shards/msrvtt-mixed-%06d.tar

QA only (no --annotations; the collator falls back to QA-only views)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --qa-annotations /path/to/train_qa.json \
        --out data/shards/msrvtt-qa-%06d.tar

Each clip is stored ONCE (raw mp4 bytes, decoded/sampled at train time) with all
its reference captions and/or QA pairs in metadata. Use --limit for a quick
subset first.
"""
from __future__ import annotations

import argparse

from viora.data.adapters.video_caption import build_caption_qa_shards, build_caption_shards


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare video-caption(+QA) shards for Viora training.")
    ap.add_argument("--videos", required=True, help="directory of video files (searched recursively)")
    ap.add_argument("--annotations", default=None,
                    help="MSR-VTT videodatainfo.json, or a {id: caption(s)} sidecar for --format folder. "
                         "Omit if using --qa-annotations alone (QA-only shards).")
    ap.add_argument("--qa-annotations", default=None,
                    help="MSRVTT-QA json: a top-level list of {video_id, question, answer} records. "
                         "Combine with --annotations for a single mixed caption+QA shard set.")
    ap.add_argument("--format", choices=["auto", "msrvtt", "records", "folder"], default="auto",
                    help="caption annotation format (ignored if --annotations is omitted): "
                         "auto-detect (default); msrvtt=videodatainfo.json; "
                         "records=list of {video_id, caption}; folder={id: caption(s)} object")
    ap.add_argument("--split", default=None, help="MSR-VTT split filter: train | validate | test")
    ap.add_argument("--out", required=True, help="printf pattern, e.g. data/shards/msrvtt-train-%%06d.tar")
    ap.add_argument("--maxcount", type=int, default=500, help="clips per shard")
    ap.add_argument("--limit", type=int, default=None, help="cap clips (quick trial run before the full build)")
    args = ap.parse_args()

    if args.qa_annotations is not None:
        n = build_caption_qa_shards(
            args.videos, args.out,
            captions_path=args.annotations, captions_fmt=args.format, qa_path=args.qa_annotations,
            split=args.split, limit=args.limit, maxcount=args.maxcount,
        )
    else:
        if args.annotations is None:
            raise SystemExit("provide --annotations and/or --qa-annotations")
        n = build_caption_shards(
            args.videos, args.annotations, args.out,
            fmt=args.format, split=args.split, limit=args.limit, maxcount=args.maxcount,
        )
    print(f"wrote {n} clips to shards matching {args.out}")
    if n == 0:
        raise SystemExit(
            "0 clips written — video filenames likely don't match annotation ids. "
            "Check that --videos points at the folder containing the .mp4 files."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the CLI parses and the combined path works end-to-end**

Run:
```bash
.venv/bin/python scripts/prepare_video_dataset.py --help
```
Expected: prints usage with `--qa-annotations` listed, exit 0.

Then, a real combined-build smoke test (uses ffmpeg to synthesize two tiny clips — skip if ffmpeg isn't installed, same convention as the adapter tests):
```bash
.venv/bin/python - <<'PY'
import json, shutil, subprocess, tempfile, os
if not shutil.which("ffmpeg"):
    print("no ffmpeg -- skipping CLI smoke test"); raise SystemExit(0)
d = tempfile.mkdtemp()
vids = os.path.join(d, "vids"); os.makedirs(vids)
for name in ("video0", "video1"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=32x32:rate=8", "-pix_fmt", "yuv420p",
                    os.path.join(vids, f"{name}.mp4")], check=True)
cap = os.path.join(d, "caps.json")
json.dump([{"video_id": "video0", "caption": "a cat plays"}], open(cap, "w"))
qa = os.path.join(d, "qa.json")
json.dump([{"video_id": "video0", "question": "what animal", "answer": "a cat"},
           {"video_id": "video1", "question": "what color", "answer": "blue"}], open(qa, "w"))
out = subprocess.run(
    ["python", "scripts/prepare_video_dataset.py", "--videos", vids, "--annotations", cap,
     "--qa-annotations", qa, "--format", "records", "--out", os.path.join(d, "c-%06d.tar")],
    capture_output=True, text=True)
print(out.stdout, out.stderr)
assert "wrote 2 clips" in out.stdout, out.stdout
print("CLI combined-build smoke test OK ✓")
PY
```
Expected: `CLI combined-build smoke test OK ✓` (or the ffmpeg-skip message on a machine without ffmpeg).

- [ ] **Step 3: Update docs/PRODUCTION.md**

Read `docs/PRODUCTION.md` around its existing "Build shards from your data" section (search for `prepare_video_dataset.py`) and add one more example block directly after the existing MSR-VTT/folder examples:

```bash
# combined captions + MSRVTT-QA (one shard set, both tasks; reuses the SAME videos)
python scripts/prepare_video_dataset.py \
  --videos /data/msrvtt/TrainValVideo --annotations /data/msrvtt/train_val_videodatainfo.json \
  --qa-annotations /data/msrvtt/train_qa.json --format msrvtt --split train \
  --out data/shards/msrvtt-mixed-%06d.tar
```

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_video_dataset.py docs/PRODUCTION.md
git commit -m "cli: support building combined caption+QA shards (--qa-annotations)"
```

---

### Task 5: HuggingFace checkpoint relay helper

**Files:**
- Create: `viora/utils/hf_relay.py`
- Test: `tests/unit/test_hf_relay.py`

**Interfaces:**
- Consumes: `huggingface_hub.HfApi`, `huggingface_hub.hf_hub_download` (already a project dependency, see `pyproject.toml`).
- Produces: `push_checkpoint_to_hf(local_path, repo_id, *, path_in_repo=None, private=True, token=None) -> str` and `pull_checkpoint_from_hf(repo_id, filename, *, local_dir=None, token=None) -> Path | None` — Task 7's notebook cells call both; no other task depends on this module.

This wraps the exact upload/download calls already used manually in the Kaggle session (creating the repo if missing, uploading a file; downloading a file, tolerating "doesn't exist yet" on the very first session) into two tested, reusable functions instead of hand-typed notebook cells.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_hf_relay.py`:

```python
"""HF checkpoint relay: push a checkpoint after a session, pull it to resume the next."""

from __future__ import annotations

from pathlib import Path

import pytest

from viora.utils.hf_relay import pull_checkpoint_from_hf, push_checkpoint_to_hf


class _FakeApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, *, repo_type, exist_ok, private, token=None):
        self.created.append((repo_id, repo_type, exist_ok, private))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, token=None):
        self.uploaded.append((path_or_fileobj, path_in_repo, repo_id, repo_type))


def test_push_checkpoint_creates_repo_and_uploads(tmp_path, monkeypatch):
    ckpt = tmp_path / "final.pt"
    ckpt.write_bytes(b"fake weights")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)

    url = push_checkpoint_to_hf(ckpt, "someuser/viora-msrvtt", token="hf_fake")

    assert fake.created == [("someuser/viora-msrvtt", "model", True, True)]
    assert fake.uploaded == [(str(ckpt), "final.pt", "someuser/viora-msrvtt", "model")]
    assert url == "https://huggingface.co/someuser/viora-msrvtt"


def test_push_checkpoint_custom_path_in_repo(tmp_path, monkeypatch):
    ckpt = tmp_path / "step_1000.pt"
    ckpt.write_bytes(b"x")
    fake = _FakeApi()
    monkeypatch.setattr("viora.utils.hf_relay.HfApi", lambda: fake)

    push_checkpoint_to_hf(ckpt, "u/r", path_in_repo="checkpoints/step_1000.pt", token="t")

    assert fake.uploaded[0][1] == "checkpoints/step_1000.pt"


def test_pull_checkpoint_returns_local_path(monkeypatch, tmp_path):
    downloaded = tmp_path / "final.pt"
    downloaded.write_bytes(b"x")
    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", lambda **kw: str(downloaded))

    out = pull_checkpoint_from_hf("u/r", "final.pt", token="t")

    assert out == downloaded


def test_pull_checkpoint_returns_none_when_missing(monkeypatch):
    from viora.utils.hf_relay import _NOT_FOUND_ERRORS

    def _raise(**kw):
        raise _NOT_FOUND_ERRORS[0]("not found")

    monkeypatch.setattr("viora.utils.hf_relay.hf_hub_download", _raise)

    assert pull_checkpoint_from_hf("u/r", "final.pt", token="t") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_hf_relay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'viora.utils.hf_relay'`

- [ ] **Step 3: Implement `viora/utils/hf_relay.py`**

```python
"""Cross-session checkpoint relay through a HuggingFace model repo.

Free Kaggle/Colab sessions don't persist ``/kaggle/working`` across restarts, so a
multi-session training run needs somewhere to hand off the latest checkpoint. This
wraps the exact push/pull calls into two small, tested functions instead of hand-
typed notebook cells: push at the end of a session, pull at the start of the next.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    _NOT_FOUND_ERRORS: tuple[type[Exception], ...] = (
        EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError,
    )
except ImportError:  # pragma: no cover - older huggingface_hub without these classes
    _NOT_FOUND_ERRORS = (FileNotFoundError,)


def push_checkpoint_to_hf(
    local_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload ``local_path`` to a HF model repo (created if missing). Returns its URL."""
    local_path = Path(local_path)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo or local_path.name,
        repo_id=repo_id, repo_type="model", token=token,
    )
    return f"https://huggingface.co/{repo_id}"


def pull_checkpoint_from_hf(
    repo_id: str,
    filename: str,
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> Path | None:
    """Download ``filename`` from a HF model repo. Returns ``None`` if it doesn't
    exist yet (e.g. the first session, before anything has been pushed) instead of
    raising, so callers can fall back to starting fresh."""
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="model",
            local_dir=str(local_dir) if local_dir else None, token=token,
        )
    except _NOT_FOUND_ERRORS:
        return None
    return Path(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_hf_relay.py -v`
Expected: 4 passed

- [ ] **Step 5: Full suite + lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check viora/ tests/`
Expected: all pass, "All checks passed!"

- [ ] **Step 6: Commit**

```bash
git add viora/utils/hf_relay.py tests/unit/test_hf_relay.py
git commit -m "utils: HF checkpoint relay (push_checkpoint_to_hf / pull_checkpoint_from_hf) for cross-session resume"
```

---

### Task 6: Trainer validation loader + qa_prob wiring in `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

**Interfaces:**
- Consumes: `Trainer.train(train_loader, val_loader=None)` (already exists, unchanged, in `viora/training/trainer.py`), `build_video_text_webdataset` (existing), `VideoTextCollator(qa_prob=...)` (Task 3).
- Produces: `--val-shards` and `--qa-prob` CLI flags — Task 7's notebook training cell passes both.

`Trainer.train` already accepts an optional `val_loader` and, when `cfg.eval_every > 0`, calls `self.evaluate(val_loader)` which logs `val_loss` (see `viora/training/trainer.py:217-218` and `:239`). `scripts/train.py` currently never builds one. This task adds an optional second shard pattern for validation and wires it through, plus exposes the collator's new `qa_prob`.

- [ ] **Step 1: Modify `scripts/train.py`**

In the `hf_tokenize_fn.__call__` docstring/class — no change needed there. Modify `main()`'s argument parser and the `--shards` branch. Full diff, shown as before/after for the relevant regions of `scripts/train.py`:

Add two new arguments right after the existing `--shards` argument:

```python
    ap.add_argument("--shards", help="WebDataset shard pattern for REAL training, "
                                     "e.g. 'data/shards/train-{000000..000099}.tar'")
    ap.add_argument("--val-shards", default=None,
                    help="optional WebDataset shard pattern for periodic validation "
                         "(logs val_loss every training.eval_every steps)")
    ap.add_argument("--qa-prob", type=float, default=0.5,
                    help="chance a video with BOTH captions and QA pairs renders as a "
                         "Question/Answer view instead of a caption (default 0.5)")
```

Replace the `if args.shards:` branch (which currently builds `collator.tokenize_fn` and one `loader`) with a version that also optionally builds `val_loader`:

```python
    val_loader = None
    if args.shards:
        # REAL training from sharded video-text data with the real LLM tokenizer.
        from viora.data.webdataset_pipeline import build_video_text_webdataset

        if model.llm.tokenizer is None:
            raise SystemExit("--shards needs a real LLM (set llm.dummy=false in the model config)")
        collator.tokenize_fn = hf_tokenize_fn(model.llm.tokenizer, model_cfg.llm.max_length)
        collator.qa_prob = args.qa_prob
        ds = build_video_text_webdataset(
            args.shards, num_frames=model_cfg.vision.num_frames, transform=_transform_for(model_cfg)
        )
        loader = DataLoader(ds, batch_size=train_cfg.batch_size, collate_fn=collator,
                            num_workers=train_cfg.num_workers)
        if args.val_shards:
            val_collator = VideoTextCollator(
                tokenize_fn=hf_tokenize_fn(model.llm.tokenizer, model_cfg.llm.max_length),
                qa_prob=args.qa_prob,
            )
            val_ds = build_video_text_webdataset(
                args.val_shards, num_frames=model_cfg.vision.num_frames,
                transform=_transform_for(model_cfg), resampled=False, shuffle=0,
            )
            val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, collate_fn=val_collator,
                                    num_workers=train_cfg.num_workers)
    else:
        ds = SyntheticVideoDataset(
            size=32, num_frames=model_cfg.vision.num_frames, image_size=model_cfg.vision.image_size
        )
        collator.tokenize_fn = smoke_tokenize_fn(model.llm.video_token_id, model.llm.vocab_size)
        loader = DataLoader(ds, batch_size=train_cfg.batch_size, collate_fn=collator,
                            num_workers=train_cfg.num_workers)
```

Update the trainer call at the bottom of `main()`:

```python
    trainer = Trainer(model, train_cfg, device=device, full_config=model_cfg)
    summary = trainer.train(loader, val_loader=val_loader)
```

`val_ds` uses `resampled=False, shuffle=0` (unlike the infinite/shuffled training stream) so validation is a finite, deterministic pass — matching the pattern already used in `tests/unit/test_webdataset.py::test_shard_round_trip`.

- [ ] **Step 2: Smoke-test the new flags parse and don't break `--smoke`**

Run:
```bash
.venv/bin/python scripts/train.py --smoke training.max_steps=4 training.log_every=2 training.save_every=0
```
Expected: exits 0, prints `final metrics: {...}` (identical behavior to before — `--val-shards` defaults to `None`, so `val_loader` stays `None` and nothing new activates on the smoke path).

Run: `.venv/bin/python scripts/train.py --help`
Expected: usage text lists `--val-shards` and `--qa-prob`.

- [ ] **Step 3: Full suite + lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check viora/ scripts/`
Expected: all pass, "All checks passed!" (this task touches no test files; the existing suite must still be green because `scripts/train.py` isn't unit-tested directly — the `--smoke` run above is the functional check)

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "train: optional --val-shards (logs val_loss) and --qa-prob CLI flags"
```

---

### Task 7: Notebook — combined shards, cross-session HF resume, QA inference

**Files:**
- Modify: `notebooks/train_free_gpu.ipynb`

**Interfaces:**
- Consumes: `build_caption_qa_shards` via `scripts/prepare_video_dataset.py --qa-annotations` (Task 4), `push_checkpoint_to_hf` / `pull_checkpoint_from_hf` (Task 5), `--val-shards --qa-prob training.resume=` on `scripts/train.py` (Task 6), existing `VioraInferencePipeline.generate_answer` (already implemented in `viora/inference/pipeline.py`).
- Produces: nothing consumed by later tasks — this is the last task before the wrap-up.

**Important — editing this notebook safely:** earlier work in this project found that editing notebook cells by positional `cell_id` is fragile (an insert shifts every later id, and a stale id silently edits the wrong cell or duplicates content). Every step below uses a small Python script that finds cells **by matching their source text**, not by id, and re-validates the full cell list afterward. Do not use `NotebookEdit` with a hardcoded `cell_id` for this task.

- [ ] **Step 1: Write and run the cell-insertion/edit script**

Create a scratch script (adjust the path to your scratchpad directory) with this content, then run it:

```python
import json
from pathlib import Path

NB = Path("/Users/garvbahl/Documents/Projects/Viora/notebooks/train_free_gpu.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]


def as_source(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def find(pred):
    return next(i for i, c in enumerate(cells) if pred("".join(c["source"])))


def new_code_cell(cell_id, text):
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "execution_count": None, "outputs": [], "source": as_source(text)}


# ---- 1) new cell: combined caption+QA shards + a held-out val split (after cell "5b") ----
CELL_5C = '''\
# 5c) COMBINED captions + MSRVTT-QA -> one shard set for BOTH tasks (multi-task training).
#      Reuses the SAME videos as cell 5b; only fetches the QA annotations. Also builds a
#      small held-out VAL split (captions' "validate" split, + val QA if available) so
#      training can log a real val_loss. Nothing here re-downloads any video.
import glob, os

QA_CANDIDATES = [("morpheushoc/msrvtt-qa", "train_qa.json"), ("morpheushoc/msrvtt-qa", "msrvtt_qa_train.json")]
QA_VAL_CANDIDATES = [("morpheushoc/msrvtt-qa", "val_qa.json"), ("morpheushoc/msrvtt-qa", "msrvtt_qa_val.json")]

if "VIDEOS_DIR" not in globals():
    print("cell 5b did not find real videos -> skipping combined caption+QA build.")
    MIXED_SHARDS = None
    VAL_SHARDS = None
else:
    def _try_fetch(candidates):
        from huggingface_hub import hf_hub_download
        for repo, fname in candidates:
            try:
                path = hf_hub_download(repo, fname, repo_type="dataset")
                print(f"  found QA annotations: {repo}/{fname}")
                return path
            except Exception as e:
                print(f"  {repo}/{fname} unavailable ({type(e).__name__}); trying next...")
        return None

    print("fetching MSRVTT-QA train annotations...")
    QA_TRAIN = _try_fetch(QA_CANDIDATES)

    if QA_TRAIN is None:
        print("!! no MSRVTT-QA source worked -> training will stay caption-only (SHARDS from cell 5b).")
        MIXED_SHARDS = None
        VAL_SHARDS = None
    else:
        !python scripts/prepare_video_dataset.py \\
            --videos "{VIDEOS_DIR}" --annotations "{ANNOT}" --qa-annotations "{QA_TRAIN}" \\
            --format auto --split train \\
            --out data/shards/msrvtt-mixed-%06d.tar --maxcount 500
        _n = len(glob.glob("data/shards/msrvtt-mixed-*.tar"))
        if _n:
            MIXED_SHARDS = "data/shards/msrvtt-mixed-{000000..%06d}.tar" % (_n - 1)
            SHARDS = MIXED_SHARDS   # cell 6 trains on the mixed set from here on
            print(f"OK -> training will use COMBINED caption+QA shards: {MIXED_SHARDS}")
        else:
            print("!! 0 combined shards written -> keeping caption-only SHARDS from cell 5b.")
            MIXED_SHARDS = None

        # held-out validation: only if the local videodatainfo.json actually has a "validate" split
        VAL_SHARDS = None
        try:
            QA_VAL = _try_fetch(QA_VAL_CANDIDATES)
            _qa_val_flag = f'--qa-annotations "{QA_VAL}"' if QA_VAL else ""
            !python scripts/prepare_video_dataset.py \\
                --videos "{VIDEOS_DIR}" --annotations "{ANNOT}" {_qa_val_flag} \\
                --format auto --split validate \\
                --out data/shards/msrvtt-val-%06d.tar --maxcount 500 --limit 300
            _vn = len(glob.glob("data/shards/msrvtt-val-*.tar"))
            if _vn:
                VAL_SHARDS = "data/shards/msrvtt-val-{000000..%06d}.tar" % (_vn - 1)
                print(f"val shards -> {VAL_SHARDS}")
        except Exception as e:
            print(f"no 'validate' split available ({type(e).__name__}) -> skipping val_loss logging.")
'''

if not any("# 5c) COMBINED" in "".join(c["source"]) for c in cells):
    i = find(lambda s: s.startswith("# 5b) REAL MSR-VTT"))
    cells.insert(i + 1, new_code_cell("cell-5c-combined", CELL_5C))
    print("inserted cell 5c")
else:
    print("cell 5c already present -- skipping insert")


# ---- 2) new cell: pull latest checkpoint from HF before training (after cell "4") ----
CELL_RESUME = '''\
# 4b) Resume from HF if a checkpoint was pushed in a PREVIOUS session (else start from
#      the last local one, else start fresh). Set HF_REPO to your model repo.
HF_REPO = "bharatverse11/viora-msrvtt"
HF_TOKEN = None  # paste a HF token here ONLY if HF_REPO is private; never commit a token to git

from viora.utils.hf_relay import pull_checkpoint_from_hf

RESUME_FROM = ""
pulled = pull_checkpoint_from_hf(HF_REPO, "final.pt", local_dir=OUT, token=HF_TOKEN)
if pulled is not None:
    RESUME_FROM = str(pulled)
    print(f"resuming from HF checkpoint -> {RESUME_FROM}")
else:
    import glob
    local = sorted(glob.glob(f"{OUT}/step_*.pt"),
                  key=lambda p: int(p.split("step_")[1].split(".")[0]))
    RESUME_FROM = local[-1] if local else ""
    print(f"no HF checkpoint yet -> {'resuming locally from ' + RESUME_FROM if RESUME_FROM else 'starting fresh'}")
'''

if not any("# 4b) Resume from HF" in "".join(c["source"]) for c in cells):
    i = find(lambda s: s.startswith("# 4) Choose an output dir"))
    cells.insert(i + 1, new_code_cell("cell-4b-hf-resume", CELL_RESUME))
    print("inserted cell 4b")
else:
    print("cell 4b already present -- skipping insert")


# ---- 3) modify the training cell to use RESUME_FROM, VAL_SHARDS, qa_prob, bumped steps ----
NEW_TRAIN_CELL = '''\
# 6) Train: LoRA on Qwen-0.5B + frozen SigLIP + Viora's trainable bridge.
#    T4 supports fp16 (NOT bf16); small num_workers for Kaggle's limited CPUs.
#    Checkpoints to {OUT} every 200 steps -> resumable across free sessions.
#    (If you hit out-of-memory, lower training.batch_size to 2.)
#
#    STEPS this session -- resume across sessions until the TOTAL reaches ~20000:
#      per-session budget : 3000-5000   (fits one free Kaggle session on a T4)
#      overall target      : ~20000 additional steps for a strong caption+QA model
MAX_STEPS_THIS_SESSION = 5000

_resume_flag = f"training.resume={RESUME_FROM}" if RESUME_FROM else ""
_val_flags = f'--val-shards "{VAL_SHARDS}" training.eval_every=500' if VAL_SHARDS else ""

!python scripts/train.py \\
  --model configs/model/viora_pragmatic.yaml \\
  --train configs/training/pragmatic_lora.yaml \\
  --shards "{SHARDS}" {_val_flags} --qa-prob 0.5 \\
  llm.name_or_path=Qwen/Qwen2.5-0.5B-Instruct \\
  training.precision=fp16 training.batch_size=4 training.num_workers=4 \\
  training.gradient_checkpointing=true {_resume_flag} \\
  training.max_steps={MAX_STEPS_THIS_SESSION} training.save_every=200 training.log_every=20 \\
  training.output_dir={OUT}
'''

i = find(lambda s: s.startswith("# 6) Train"))
cells[i]["source"] = as_source(NEW_TRAIN_CELL)
cells[i]["outputs"] = []
cells[i]["execution_count"] = None
print(f"updated training cell at index {i}")


# ---- 4) new cell: push checkpoint to HF at the end of the session (after training cell) ----
CELL_PUSH = '''\
# 6b) Push this session's checkpoint to HF so the NEXT session can resume from it.
from viora.utils.hf_relay import push_checkpoint_to_hf

url = push_checkpoint_to_hf(f"{OUT}/final.pt", HF_REPO, token=HF_TOKEN)
print(f"pushed -> {url}  (next session will auto-resume from this)")
'''

if not any("# 6b) Push this session" in "".join(c["source"]) for c in cells):
    i = find(lambda s: s.startswith("# 6) Train"))
    cells.insert(i + 1, new_code_cell("cell-6b-push-hf", CELL_PUSH))
    print("inserted cell 6b")
else:
    print("cell 6b already present -- skipping insert")


# ---- 5) new cell: ask a question about a clip (after the existing caption cell) ----
CELL_ASK = '''\
# 7b) Ask a question about a real clip (uses the SAME checkpoint loaded in cell 7).
QUESTION = "what is happening in this video"

video_path = glob.glob(f"{INPUT_ROOT}/**/*.mp4", recursive=True)[0]
idx = pipe.index(video_path)
answer, conf = pipe.generate_answer(idx, QUESTION, model.llm.tokenizer)
print(f"{os.path.basename(video_path)}\\nQ: {QUESTION}\\nA: {answer!r}  (conf {conf:.2f})")
'''

if not any("# 7b) Ask a question" in "".join(c["source"]) for c in cells):
    i = find(lambda s: s.startswith("# 7) See what it learned"))
    cells.insert(i + 1, new_code_cell("cell-7b-ask", CELL_ASK))
    print("inserted cell 7b")
else:
    print("cell 7b already present -- skipping insert")


NB.write_text(json.dumps(nb, indent=1) + "\n")

# ---- validate + print final structure ----
nb2 = json.loads(NB.read_text())
print(f"\nFINAL {len(nb2['cells'])} cells:")
for i, c in enumerate(nb2["cells"]):
    head = "".join(c["source"])[:55].replace("\n", " ")
    print(f"  {i} {c['cell_type']:8} | {head}")
```

Run it: `python3 <path-to-script>`

Expected output ends with a numbered list of ~13 cells including new entries for "5c) COMBINED", "4b) Resume from HF", "6b) Push this session", and "7b) Ask a question", with the training cell (`# 6) Train`) still present and updated.

- [ ] **Step 2: Validate the notebook JSON and re-check cell order**

Run:
```bash
python3 -c "import json; nb=json.load(open('notebooks/train_free_gpu.ipynb')); print('valid ✓', len(nb['cells']), 'cells')"
```
Expected: `valid ✓ 13 cells` (or however many resulted — must not error).

Manually re-read the full notebook source to confirm no duplicate/garbled cells (this mirrors the exact verification step used earlier in this project after prior notebook edits):
```bash
python3 -c "
import json
nb = json.load(open('notebooks/train_free_gpu.ipynb'))
for i, c in enumerate(nb['cells']):
    print(f'=== {i} ({c[\"cell_type\"]}) ===')
    print(''.join(c['source'])[:200])
    print()
"
```
Expected: each cell's content is exactly one coherent unit — no cell contains two concatenated snippets, no cell is empty when it shouldn't be.

- [ ] **Step 3: Update the trailing markdown cell's guidance**

Find the final markdown cell (starts with `## Resuming after a session times out`) and replace its "What this training gives you (honest scope)" section to reflect the new capability. Use the same content-matching script pattern:

```python
import json
from pathlib import Path

NB = Path("/Users/garvbahl/Documents/Projects/Viora/notebooks/train_free_gpu.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]


def as_source(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


i = next(i for i, c in enumerate(cells)
        if "".join(c["source"]).startswith("## Resuming after a session"))
text = "".join(cells[i]["source"])
old = """### What this training gives you (honest scope)

MSR-VTT is **caption** supervision, so the model learns to **describe** clips and answer
open-ended *"what is happening / what is in the video"* questions. It will **not** reliably do
counting, yes/no, or precise spatial questions — that needs real **QA** data (e.g. MSRVTT-QA), which
you'd feed through the adapter's `question`/`answer` fields. The **evidence timestamps** are only
meaningful once you also train on a **temporal-grounding** set (e.g. Charades-STA); on caption-only
data the grounding head stays untrained. Bigger LLM (`Qwen/Qwen2.5-1.5B-Instruct`) + more steps +
more data = better quality."""
new = """### What this training gives you (honest scope)

Cell **5c** builds a COMBINED caption+MSRVTT-QA shard set from the SAME videos (no extra
download): each view is randomly rendered as a caption or a `Question:/Answer:` pair
(`--qa-prob`, default 0.5), so the model learns to both **describe** clips and **answer
questions** about them. Cells **4b**/**6b** relay the latest checkpoint through your HF repo
(`HF_REPO`) so a multi-session run (~20k total steps) resumes automatically — no re-download,
no restart from step 0. If cell 5c can't find a working MSRVTT-QA source, training falls back to
caption-only (cell 5b's shards) automatically.

The **evidence timestamps** are still not meaningful — that needs a **temporal-grounding** set
(e.g. Charades-STA), which is out of scope for this run. Bigger LLM
(`Qwen/Qwen2.5-1.5B-Instruct`) + more steps + more data = better quality still."""
assert old in text, "expected markdown section not found verbatim -- read the cell and adjust the match"
cells[i]["source"] = as_source(text.replace(old, new))
NB.write_text(json.dumps(nb, indent=1) + "\n")
print("updated trailing markdown cell")
```

Run it, then re-validate: `python3 -c "import json; json.load(open('notebooks/train_free_gpu.ipynb')); print('valid ✓')"`

- [ ] **Step 4: Full suite + lint (notebook changes touch no Python source, but re-run as a safety net)**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check viora/ tests/ scripts/`
Expected: all pass, "All checks passed!"

- [ ] **Step 5: Commit**

```bash
git add notebooks/train_free_gpu.ipynb
git commit -m "notebook: combined caption+QA shards, HF checkpoint relay for cross-session resume, QA inference cell"
```

---

### Task 8: Final integration check

**Files:** none (verification only)

**Interfaces:** none — this task confirms Tasks 1–7 compose correctly.

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (baseline was 149 before this plan; expect roughly 149 + ~4 (Task 1) + ~4 (Task 2) + ~3 (Task 3) + ~4 (Task 5) = ~164, exact count not load-bearing — the requirement is zero failures)

- [ ] **Step 2: Run lint across everything touched**

Run: `.venv/bin/ruff check viora/ tests/ scripts/`
Expected: "All checks passed!"

- [ ] **Step 3: Validate the notebook one more time**

Run: `python3 -c "import json; nb=json.load(open('notebooks/train_free_gpu.ipynb')); print('valid ✓', len(nb['cells']), 'cells')"`
Expected: no error.

- [ ] **Step 4: Confirm git identity on the commits made in this plan**

Run: `git log --oneline -20 --format='%h %ae %s' | head -20`
Expected: every commit from this plan shows `garvbahl37@gmail.com`; run `git log -8 --format='%B' | grep -ci claude` and confirm it prints `0` (no `Co-Authored-By: Claude` trailer anywhere).

- [ ] **Step 5: Push**

```bash
git push origin main
```
Expected: pushes all commits from Tasks 1–7 cleanly (no force push, no conflicts).

- [ ] **Step 6: Tell the user what to do on Kaggle**

Report back: `git pull` in the Kaggle notebook, run cells 1–5, 5b, **5c** (new), 4b (new, before training — note actual final cell order from Task 7 Step 1's printed listing), 6 (now resumes + trains the mixed set), 6b (new, pushes checkpoint), 7 and 7b (caption + QA answer). Remind the user to paste their own HF token into `HF_TOKEN` in cell 4b only inside Kaggle, never back into chat, and to repeat the cycle (pull → resume → train → push) across sessions until the total step count reaches ~20k.
