"""Temporal grounding metrics: 1-D IoU, mIoU, and Recall@IoU thresholds."""

from __future__ import annotations

Segment = tuple[float, float]


def temporal_iou(pred: Segment, gt: Segment) -> float:
    """Intersection-over-union of two ``(start, end)`` time segments."""
    ps, pe = pred
    gs, ge = gt
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def grounding_metrics(
    predictions: list[Segment], references: list[Segment], thresholds=(0.3, 0.5, 0.7)
) -> dict[str, float]:
    """Return mIoU and Recall@IoU for each threshold (fraction with IoU ≥ t)."""
    if not predictions:
        return {"mIoU": 0.0, **{f"R@{t}": 0.0 for t in thresholds}}
    ious = [temporal_iou(p, g) for p, g in zip(predictions, references, strict=True)]
    out = {"mIoU": sum(ious) / len(ious)}
    for t in thresholds:
        out[f"R@{t}"] = sum(i >= t for i in ious) / len(ious)
    return out
