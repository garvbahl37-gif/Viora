"""Distributed-training helpers (torch.distributed / DDP-ready).

Thin wrappers so the trainer stays clean and single-GPU/CPU runs pay nothing.
Actual process launch is via ``torchrun``; these read the env it sets.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int]:
    """Initialize the process group from env if launched under torchrun.

    Returns ``(rank, world_size)``; ``(0, 1)`` when not distributed.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, get_world_size()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap in DistributedDataParallel (CUDA). No-op when not distributed."""
    if not is_distributed():
        return model
    device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=True
    )


def wrap_fsdp(
    model: torch.nn.Module,
    *,
    transformer_layer_classes: set | None = None,
    param_dtype: torch.dtype | None = None,
    min_num_params: int = 1_000_000,
) -> torch.nn.Module:
    """Wrap in FullyShardedDataParallel for parameter sharding across GPUs.

    Shards by transformer layer when the layer classes are known (best), else by a
    size threshold. Requires CUDA + an initialized process group. Untested on this
    (CPU/MPS) machine — it is a guarded GPU path (see DESIGN_REVIEW.md).
    """
    if not is_distributed():
        return model
    import functools

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    if transformer_layer_classes:
        policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=set(transformer_layer_classes)
        )
    else:
        policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    mp = (
        MixedPrecision(param_dtype=param_dtype, reduce_dtype=param_dtype, buffer_dtype=param_dtype)
        if param_dtype is not None
        else None
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # keeps optimizer param-group construction working
    )


def viora_transformer_layers() -> set:
    """The transformer block classes to shard on (for FSDP transformer policy)."""
    from viora.models.common import TransformerBlock
    from viora.models.vision.blocks import SpatioTemporalBlock

    layers = {TransformerBlock, SpatioTemporalBlock}
    return layers
