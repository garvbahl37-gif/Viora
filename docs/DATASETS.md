# Datasets

Viora **never auto‑downloads** data. The registry (`viora/data/registry.py`)
describes each dataset and reports, per data root, whether it is actually usable.
Hugging Face repository ids are **not hardcoded** — set them in config so they are
easy to replace and never fabricated.

## Availability model

`registry.availability(name, root)` returns one of:

| Status | Meaning |
|--------|---------|
| `available` | annotations present **and** pixels resolvable locally |
| `annotations_only` | labels present; videos are URLs / external ids to fetch |
| `requires_manual_download` | download the assets yourself, then re‑check |
| `requires_auth` | signed licence / login gated |
| `missing_files` | nothing found at the given root |

```bash
python scripts/inspect_dataset.py --list
python scripts/inspect_dataset.py --dataset nextqa --root data/nextqa
python scripts/prepare_dataset.py  --dataset activitynet --root data/activitynet
```

## Target datasets

| Dataset | Tasks | Video source | Default access | Notes |
|---------|-------|--------------|----------------|-------|
| WebVid | alignment, captioning | URL | manual | Original distribution withdrawn — verify source/licence |
| InternVid | alignment, captioning | YouTube id | annotations‑only | Fetch videos yourself; respect YouTube ToS |
| HowTo100M | alignment, captioning | YouTube id | annotations‑only | ASR captions; noisy alignment |
| Something‑Something v2 | action recognition | local | auth | Login required; non‑commercial |
| MSR‑VTT | captioning, alignment, QA | local | manual | Videos as a zip |
| ActivityNet Captions | dense captioning, grounding | YouTube id | annotations‑only | Some videos unavailable |
| Charades‑STA | temporal grounding | local | manual | Charades videos downloaded separately |
| Ego4D | temporal QA, grounding, dense cap | local | auth | Signed licence + official CLI |
| NExT‑QA | video/causal/temporal QA | local | manual | Videos from VidOR |
| TGIF‑QA | video QA | local | manual | Convert GIF→mp4 in preprocessing |
| MSVD‑QA | video QA | local | manual | Built on MSVD clips |
| VideoInstruct | instruction following, QA | local | annotations‑only | References source videos (often ActivityNet) |

Scale figures in the registry are **approximate — verify before use.**

## Canonical schema

Every adapter normalizes into `VideoSample` (`viora/data/schema.py`):
`sample_id, dataset_name, task, video(ref), duration, fps, text(question/answer/
caption/instructions), temporal(start/end/segments), labels(action/class/retrieval),
metadata`. Fields are optional where a task doesn't use them.

`validate_sample` flags empty ids, missing video refs, `start > end`, and
task‑specific gaps; `preprocessing/validation.py` batches this into a reason‑coded
report so a large run logs and skips bad examples (non‑strict) rather than crashing.

## Licence verification checklist

Before training on any dataset:

- [ ] Confirmed the licence permits your use (research/commercial).
- [ ] Set the HF repo id / paths in config (not hardcoded, not fabricated).
- [ ] For URL/id sources, confirmed you may fetch and store the pixels.
- [ ] For auth‑gated sets (Ego4D, SSv2), completed the required agreement.
- [ ] Ran `prepare_dataset.py` and got `available` / `annotations_only`.
- [ ] Ran dedup + train/val leakage check (`preprocessing/deduplication.py`).
