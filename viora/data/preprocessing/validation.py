"""Data-quality validation with machine-readable reason codes.

Wraps :func:`viora.data.schema.validate_sample` and adds pipeline-level checks so
a large preprocessing run can log & skip bad examples (non-strict) instead of
crashing on one corrupt entry, while ``strict`` mode still raises.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from viora.data.schema import VideoSample, validate_sample


@dataclass
class ValidationReport:
    total: int
    kept: int
    rejected: int
    reasons: dict[str, int]


def validate_dataset(
    samples: list[VideoSample], *, strict: bool = False
) -> tuple[list[VideoSample], ValidationReport]:
    """Return the kept samples and a report of rejection reason codes."""
    kept: list[VideoSample] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for s in samples:
        problems = validate_sample(s, strict=strict)
        if s.sample_id in seen_ids:
            problems.append("duplicate sample_id")
        if problems:
            if strict:
                raise ValueError(f"sample {s.sample_id!r} invalid: {problems}")
            for p in problems:
                reasons[p] += 1
            continue
        seen_ids.add(s.sample_id)
        kept.append(s)

    report = ValidationReport(
        total=len(samples), kept=len(kept), rejected=len(samples) - len(kept), reasons=dict(reasons)
    )
    return kept, report
