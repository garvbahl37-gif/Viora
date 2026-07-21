#!/usr/bin/env python3
"""Validate a dataset's requirements before any preprocessing — never auto-downloads.

Reports what is present, what is missing, and the exact next step. Real
preprocessing (transcode, index, tokenize) is only attempted once the required
annotations/pixels are confirmed present.

    python scripts/prepare_dataset.py --dataset activitynet --root data/activitynet
"""
from __future__ import annotations

import argparse

from rich.console import Console

from viora.data.registry import DEFAULT_REGISTRY, Availability, VideoSource

console = Console()

_NEXT_STEP = {
    Availability.MISSING_FILES: "Place the annotation files under --root (see 'annotation files').",
    Availability.REQUIRES_AUTH: "Accept the dataset licence and download via its official tool.",
    Availability.REQUIRES_MANUAL_DOWNLOAD: "Download the videos/annotations manually, then re-run.",
    Availability.ANNOTATIONS_ONLY: "Annotations found; fetch the referenced videos, then re-run.",
    Availability.AVAILABLE: "Ready — run your preprocessing/indexing step.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a dataset before preprocessing.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    reg = DEFAULT_REGISTRY
    spec = reg.get(args.dataset)
    avail = reg.availability(args.dataset, args.root)

    console.rule(f"[bold]prepare {spec.name}")
    console.print(f"video source : {spec.video_source.value}")
    console.print(f"expected     : {', '.join(spec.annotation_files)}")
    console.print(f"license      : {spec.license_note}")
    console.print(f"status       : [bold]{avail.value}[/bold] @ {args.root}")
    console.print(f"next step    : {_NEXT_STEP[avail]}")

    if avail is Availability.AVAILABLE:
        console.print("\n[green]Requirements satisfied.[/green] Wire your preprocessing here.")
        return 0
    if avail is Availability.ANNOTATIONS_ONLY and spec.video_source is not VideoSource.LOCAL:
        console.print("\n[yellow]Annotations only[/yellow] — videos are fetched separately; nothing downloaded.")
        return 0
    console.print("\n[red]Not ready.[/red] Nothing was downloaded (by design).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
