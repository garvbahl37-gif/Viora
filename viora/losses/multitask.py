"""Multi-task loss combination.

Combines active loss components using configured weights, logs every component
independently, and — crucially — **does not hide NaN/Inf**: non-finite components
are reported in the returned diagnostics so the trainer can act (skip step, dump
tensor stats) rather than silently propagating a corrupt gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LossBreakdown:
    total: torch.Tensor
    components: dict[str, float] = field(default_factory=dict)   # detached, for logging
    weights: dict[str, float] = field(default_factory=dict)
    nonfinite: list[str] = field(default_factory=list)


class MultiTaskLossManager:
    """Weighted sum of named losses. Not an ``nn.Module`` — holds no parameters."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = dict(weights)

    def combine(self, losses: dict[str, torch.Tensor | None]) -> LossBreakdown:
        total: torch.Tensor | None = None
        comps: dict[str, float] = {}
        used: dict[str, float] = {}
        nonfinite: list[str] = []

        for name, value in losses.items():
            if value is None:
                continue
            w = self.weights.get(name, 1.0)
            comps[name] = float(value.detach())
            used[name] = w
            if not torch.isfinite(value).all():
                nonfinite.append(name)
                continue  # exclude from the total but surface it
            contribution = w * value
            total = contribution if total is None else total + contribution

        if total is None:  # nothing finite contributed
            any_val = next((v for v in losses.values() if v is not None), None)
            total = (any_val.sum() * 0.0) if any_val is not None else torch.zeros(())

        return LossBreakdown(total=total, components=comps, weights=used, nonfinite=nonfinite)
