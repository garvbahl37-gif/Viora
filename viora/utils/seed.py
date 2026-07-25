"""Reproducibility helpers.

Seeds every RNG Viora touches (Python, NumPy, PyTorch CPU/CUDA) and exposes a
worker-init function for deterministic ``DataLoader`` workers.

Note on determinism: fully deterministic GPU execution requires
``torch.use_deterministic_algorithms(True)`` plus the ``CUBLAS_WORKSPACE_CONFIG``
environment variable, and it disables some fast kernels. We expose that behind
the ``deterministic`` flag so training can opt in without paying the cost during
normal runs. Some cuDNN ops have no deterministic implementation; those will
raise if ``deterministic=True`` — that is intentional (surface, don't hide).
"""

from __future__ import annotations

import os
import random

import numpy as np

try:  # torch is a hard dependency of the library, but seed helpers stay usable
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch always present in practice
    _HAS_TORCH = False


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed all RNGs Viora relies on.

    Args:
        seed: The base seed.
        deterministic: If True, force deterministic algorithms where PyTorch
            supports them. Slower; use for exact reproducibility experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Required for deterministic cuBLAS GEMMs.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by torch
    """``DataLoader`` ``worker_init_fn`` giving each worker a distinct, reproducible seed.

    PyTorch sets ``torch.initial_seed()`` per worker (base_seed + worker_id); we
    derive Python/NumPy seeds from it so all three RNGs are aligned per worker.
    """
    if not _HAS_TORCH:  # pragma: no cover
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_states() -> dict:
    """Snapshot all RNG states for checkpointing."""
    states: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if _HAS_TORCH:
        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    """Restore RNG states captured by :func:`get_rng_states` (best-effort)."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if _HAS_TORCH and "torch" in states:
        # torch.set_rng_state() requires a CPU ByteTensor specifically. Trainer._resume
        # loads checkpoints with map_location=<training device>; torch.load relocates
        # EVERY tensor in the pickle to that device -- including this CPU-only RNG
        # state -- so on a CUDA resume it arrives as a CUDA tensor and set_rng_state
        # raises "RNG state must be a torch.ByteTensor". Force it back to CPU; CUDA RNG
        # state is restored separately via torch_cuda below.
        torch.set_rng_state(states["torch"].cpu())
        if "torch_cuda" in states and torch.cuda.is_available():
            # Same map_location relocation problem, per-GPU: torch.cuda.set_rng_state_all()
            # also requires each per-device state to be a CPU ByteTensor internally.
            torch.cuda.set_rng_state_all([s.cpu() for s in states["torch_cuda"]])
