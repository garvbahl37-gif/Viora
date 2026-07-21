#!/usr/bin/env python3
"""Report the runtime environment and recommend a safe Viora configuration.

Prints Python / PyTorch / device capabilities and the status of optional
dependencies, then suggests precision + frame/resolution settings that should fit
the detected hardware. It deliberately allocates **no** large tensors.

    python scripts/validate_environment.py
"""
from __future__ import annotations

import importlib.util
import platform
import sys

from rich.console import Console
from rich.table import Table

console = Console()

OPTIONAL = {
    "av": "PyAV video decoding backend",
    "decord": "decord video decoding backend (fast, optional)",
    "torchvision": "torchvision decoding / transforms",
    "transformers": "Hugging Face language models",
    "peft": "LoRA / parameter-efficient fine-tuning",
    "deepspeed": "DeepSpeed distributed training (optional)",
    "fastapi": "serving API",
    "tensorboard": "training metric logging",
    "safetensors": "safe checkpoint serialization",
    "pytest": "test runner",
}


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def main() -> int:
    console.rule("[bold]Viora environment check")

    # --- core ---
    core = Table(show_header=False, box=None, pad_edge=False)
    core.add_column(style="cyan", no_wrap=True)
    core.add_column()
    core.add_row("Python", platform.python_version())
    core.add_row("Platform", f"{platform.system()} {platform.machine()}")

    try:
        import torch
    except ImportError:
        console.print(core)
        console.print("\n[red]PyTorch is not installed.[/red] Install it, e.g.:")
        console.print("  uv pip install torch")
        return 1

    core.add_row("PyTorch", torch.__version__)

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    device = "cuda" if cuda else ("mps" if mps else "cpu")
    core.add_row("Device", f"[bold green]{device}[/bold green]")

    bf16 = False
    if cuda:
        core.add_row("CUDA", torch.version.cuda or "unknown")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            core.add_row(f"GPU {i}", f"{props.name}  ({_bytes_to_gb(props.total_memory):.1f} GB)")
        bf16 = torch.cuda.is_bf16_supported()
        core.add_row("BF16 supported", "yes" if bf16 else "no")
    elif mps:
        core.add_row("GPU", "Apple MPS (unified memory)")
        core.add_row("BF16 autocast", "not supported on MPS")
    else:
        core.add_row("GPU", "none — CPU only")
        bf16 = True  # CPU bf16 autocast is broadly available on modern hardware
        core.add_row("BF16 autocast (CPU)", "available")

    console.print(core)

    # --- optional deps ---
    console.rule("[bold]Optional dependencies")
    deps = Table(show_header=True, header_style="bold", box=None)
    deps.add_column("package")
    deps.add_column("status")
    deps.add_column("purpose", style="dim")
    for mod, desc in OPTIONAL.items():
        ok = _has(mod)
        deps.add_row(mod, "[green]installed[/green]" if ok else "[yellow]missing[/yellow]", desc)
    console.print(deps)

    # --- recommendation (no allocation, purely heuristic) ---
    console.rule("[bold]Recommended starting configuration")
    if cuda:
        mem = _bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)
        precision = "bf16" if bf16 else "fp16"
        if mem >= 40:
            frames, res, note = 32, 224, "large GPU — room for longer clips / bigger batches"
        elif mem >= 16:
            frames, res, note = 16, 224, "mid GPU — enable gradient checkpointing for depth"
        else:
            frames, res, note = 8, 160, "small GPU — reduce frames/resolution, accumulate gradients"
    elif mps:
        precision, frames, res, note = (
            "fp32", 8, 160,
            "MPS: Conv3d is unsupported — export PYTORCH_ENABLE_MPS_FALLBACK=1 (CPU fallback) or train on CPU",
        )
    else:
        precision, frames, res, note = "fp32", 8, 128, "CPU dev/CI — keep tensors tiny; use viora_tiny"

    rec = Table(show_header=False, box=None)
    rec.add_column(style="cyan", no_wrap=True)
    rec.add_column()
    rec.add_row("precision", precision)
    rec.add_row("num_frames", str(frames))
    rec.add_row("image_size", str(res))
    rec.add_row("config", "configs/model/viora_tiny.yaml")
    rec.add_row("note", note)
    console.print(rec)
    console.print(
        f"\n[dim]Example:[/dim] python scripts/train_stage1.py "
        f"--precision {precision} vision.num_frames={frames} vision.image_size={res}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
