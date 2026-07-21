"""Device selection and autocast helpers.

Viora must run on CPU (development / CI), CUDA (training), and Apple MPS
(local Apple-Silicon development) without code changes. All device-dependent
behaviour is funnelled through here so the rest of the codebase never hardcodes
``"cuda"``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved device capabilities used to pick safe defaults."""

    device: torch.device
    type: str  # "cuda" | "mps" | "cpu"
    supports_bf16: bool
    supports_fp16_autocast: bool
    name: str

    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve a concrete device.

    Args:
        preferred: One of ``"cuda"``, ``"mps"``, ``"cpu"``, an explicit index like
            ``"cuda:1"``, or ``None`` / ``"auto"`` to auto-select the best available.
    """
    if preferred and preferred not in ("auto",):
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(preferred: str | None = None) -> DeviceInfo:
    """Return capabilities for the resolved device (used to pick precision)."""
    device = resolve_device(preferred)
    dtype = device.type

    supports_bf16 = False
    supports_fp16_autocast = False
    name = dtype

    if dtype == "cuda":
        name = torch.cuda.get_device_name(device)
        supports_bf16 = torch.cuda.is_bf16_supported()
        supports_fp16_autocast = True
    elif dtype == "mps":
        name = "Apple MPS"
        # MPS autocast support is limited/evolving; bf16 unsupported as of torch 2.x.
        supports_bf16 = False
        supports_fp16_autocast = False
    else:
        name = "CPU"
        # CPU autocast supports bf16 on most modern x86/arm; fp16 autocast is CPU-supported
        # in recent torch but numerically fragile, so we advertise bf16 only.
        supports_bf16 = True
        supports_fp16_autocast = False

    return DeviceInfo(
        device=device,
        type=dtype,
        supports_bf16=supports_bf16,
        supports_fp16_autocast=supports_fp16_autocast,
        name=name,
    )


def autocast_context(
    device_type: str,
    *,
    enabled: bool = True,
    dtype: torch.dtype | None = None,
) -> contextlib.AbstractContextManager:
    """Return an autocast context appropriate for the device.

    Falls back to a no-op context when autocast is disabled or unsupported so
    callers can wrap forward passes unconditionally.
    """
    if not enabled or device_type not in ("cuda", "cpu"):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


def resolve_amp_dtype(info: DeviceInfo, requested: str | None) -> torch.dtype | None:
    """Map a config precision string to a concrete autocast dtype (or None).

    ``requested`` may be ``"bf16"``, ``"fp16"``, ``"fp32"``/``None``. We downgrade
    gracefully (never crash) when the device cannot honour the request, logging is
    left to the caller which has the config context.
    """
    if requested in (None, "fp32", "float32", "no"):
        return None
    if requested in ("bf16", "bfloat16"):
        return torch.bfloat16 if info.supports_bf16 else None
    if requested in ("fp16", "float16", "half"):
        return torch.float16 if info.supports_fp16_autocast else None
    raise ValueError(f"Unknown precision '{requested}' (expected bf16|fp16|fp32)")
