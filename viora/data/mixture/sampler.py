"""Task-aware mixture sampling.

Naive concatenation over-samples large datasets. This sampler draws a dataset per
step from temperature-adjusted weights, then a uniform-random example within it,
mapping to a global index over a ``ConcatDataset``. It is:

* **deterministic** given ``(seed, epoch)`` — call :meth:`set_epoch` each epoch;
* **distributed-aware** — each rank takes a strided, disjoint slice;
* **validated** — weights are normalized and datasets with weight-but-no-data warn.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

from viora.utils.logging import get_logger

logger = get_logger(__name__)


class TaskAwareMixtureSampler(Sampler):
    def __init__(
        self,
        dataset_sizes: dict[str, int],
        weights: dict[str, float],
        *,
        temperature: float = 1.0,
        num_samples: int | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.names = list(dataset_sizes)
        self.sizes = np.array([dataset_sizes[n] for n in self.names], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.temperature = temperature
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        w = np.array([max(0.0, weights.get(n, 0.0)) for n in self.names], dtype=np.float64)
        for name, weight, size in zip(self.names, w, self.sizes, strict=True):
            if weight > 0 and size == 0:
                logger.warning("dataset '%s' has weight %.3f but 0 samples", name, weight)
                # zero it so it cannot be drawn
                w[self.names.index(name)] = 0.0
        if w.sum() == 0:
            raise ValueError("all mixture weights are zero (or their datasets empty)")
        w = w ** (1.0 / temperature)
        self.probs = w / w.sum()
        self.num_samples = int(num_samples or self.sizes.sum())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def normalized_weights(self) -> dict[str, float]:
        return dict(zip(self.names, self.probs.tolist(), strict=True))

    def __len__(self) -> int:
        return len(range(self.rank, self.num_samples, self.world_size))

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        chosen = g.choice(len(self.names), size=self.num_samples, p=self.probs)
        # per chosen dataset, a uniform local index (guard empty datasets: prob is 0 so unchosen)
        local = np.array([g.integers(0, max(1, self.sizes[d])) for d in chosen], dtype=np.int64)
        global_idx = self.offsets[chosen] + local
        return iter(global_idx[self.rank :: self.world_size].tolist())
