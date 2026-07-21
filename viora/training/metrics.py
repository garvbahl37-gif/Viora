"""Lightweight running-metric tracking for training loops."""

from __future__ import annotations

from collections import defaultdict


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0.0


class MetricTracker:
    """Accumulates named scalar metrics; ``summary`` returns windowed means."""

    def __init__(self) -> None:
        self._means: dict[str, RunningMean] = defaultdict(RunningMean)

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._means[k].update(v, n)

    def summary(self) -> dict[str, float]:
        return {k: m.value for k, m in self._means.items()}

    def reset(self) -> None:
        self._means.clear()
