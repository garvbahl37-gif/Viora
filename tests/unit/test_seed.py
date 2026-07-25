"""RNG seeding and checkpoint-restore of RNG state."""

from __future__ import annotations

import random

import numpy as np
import torch

from viora.utils.seed import get_rng_states, set_rng_states, set_seed


def test_set_seed_makes_python_numpy_torch_reproducible():
    set_seed(7)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(7)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_set_rng_states_round_trip_reproduces_next_draw():
    set_seed(0)
    states = get_rng_states()
    first = torch.rand(3)
    set_rng_states(states)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_set_rng_states_forces_torch_state_to_cpu(monkeypatch):
    """Regression: Trainer._resume loads checkpoints with map_location=<training
    device>. torch.load relocates EVERY tensor in the pickle to that device --
    including the CPU-only RNG state -- so on a CUDA resume it arrives as a CUDA
    tensor and torch.set_rng_state() raises "RNG state must be a torch.ByteTensor".
    set_rng_states must coerce it back to CPU before calling torch.set_rng_state.
    """
    real_cpu_state = torch.get_rng_state()

    class _FakeRelocatedTensor:
        """Stands in for a tensor torch.load moved to another device via map_location."""

        def cpu(self):
            return real_cpu_state

    seen = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda s: seen.__setitem__("state", s))

    set_rng_states({"torch": _FakeRelocatedTensor()})

    assert seen["state"] is real_cpu_state


def test_set_rng_states_forces_torch_cuda_states_to_cpu(monkeypatch):
    """Same class of bug, per-GPU: torch.cuda.set_rng_state_all() also requires each
    device's state to be a CPU ByteTensor internally, and a CUDA-relocated one raises
    the same 'RNG state must be a torch.ByteTensor'. Force each entry back to CPU.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_cpu_state = torch.get_rng_state()

    class _FakeRelocatedTensor:
        def cpu(self):
            return real_cpu_state

    seen = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda s: None)  # not under test here
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: seen.__setitem__("states", states)
    )

    set_rng_states({"torch": real_cpu_state,
                    "torch_cuda": [_FakeRelocatedTensor(), _FakeRelocatedTensor()]})

    assert all(s is real_cpu_state for s in seen["states"]) and len(seen["states"]) == 2
