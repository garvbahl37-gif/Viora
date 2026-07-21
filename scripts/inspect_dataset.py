#!/usr/bin/env python3
"""Inspect a dataset's requirements and availability — never downloads anything.

    python scripts/inspect_dataset.py --dataset nextqa [--root data/nextqa]
    python scripts/inspect_dataset.py --list
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from viora.data.registry import DEFAULT_REGISTRY

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect Viora dataset specs.")
    ap.add_argument("--dataset", help="dataset name")
    ap.add_argument("--root", help="dataset root to check availability against")
    ap.add_argument("--list", action="store_true", help="list all registered datasets")
    args = ap.parse_args()

    reg = DEFAULT_REGISTRY
    if args.list or not args.dataset:
        t = Table(title="Registered datasets", header_style="bold")
        t.add_column("name")
        t.add_column("tasks", style="cyan")
        t.add_column("default access")
        for name in reg.list_datasets():
            spec = reg.get(name)
            t.add_row(name, ", ".join(x.value for x in spec.tasks), spec.default_access.value)
        console.print(t)
        return 0

    spec = reg.get(args.dataset)
    avail = reg.availability(args.dataset, args.root)
    t = Table(show_header=False, box=None)
    t.add_column(style="cyan", no_wrap=True)
    t.add_column()
    t.add_row("name", spec.name)
    t.add_row("tasks", ", ".join(x.value for x in spec.tasks))
    t.add_row("video source", spec.video_source.value)
    t.add_row("annotation files", ", ".join(spec.annotation_files))
    t.add_row("format", spec.fmt)
    t.add_row("approx scale", spec.approx_scale)
    t.add_row("splits", ", ".join(spec.splits))
    t.add_row("hf_repo", spec.hf_repo or "[dim](set in config; not hardcoded)[/dim]")
    t.add_row("license", spec.license_note)
    t.add_row("availability", f"[bold]{avail.value}[/bold]" + (f" @ {args.root}" if args.root else ""))
    console.print(t)
    if not args.root:
        console.print("\n[dim]Pass --root to check availability against local files.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
