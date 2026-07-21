"""``viora`` command-line entrypoint (self-contained; depends only on the package).

    viora env                        # environment + capability report
    viora inspect --dataset nextqa   # dataset spec & availability
    viora datasets                   # list registered datasets
"""

from __future__ import annotations

import argparse


def _env() -> int:
    from viora.utils.device import get_device_info

    info = get_device_info()
    print(f"device: {info.type} ({info.name})  bf16={info.supports_bf16}")
    print("For the full report, run: python scripts/validate_environment.py")
    return 0


def _inspect(name: str | None, root: str | None, list_all: bool) -> int:
    from viora.data.registry import DEFAULT_REGISTRY

    reg = DEFAULT_REGISTRY
    if list_all or not name:
        for n in reg.list_datasets():
            spec = reg.get(n)
            print(f"{n:20s} {spec.default_access.value:24s} {', '.join(t.value for t in spec.tasks)}")
        return 0
    spec = reg.get(name)
    avail = reg.availability(name, root)
    print(f"name        : {spec.name}")
    print(f"tasks       : {', '.join(t.value for t in spec.tasks)}")
    print(f"video source: {spec.video_source.value}")
    print(f"format      : {spec.fmt}")
    print(f"scale       : {spec.approx_scale}")
    print(f"license     : {spec.license_note}")
    print(f"availability: {avail.value}" + (f" @ {root}" if root else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="viora", description="Viora-1 CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("env", help="environment + capability report")
    sub.add_parser("datasets", help="list registered datasets")
    p = sub.add_parser("inspect", help="inspect a dataset spec/availability")
    p.add_argument("--dataset")
    p.add_argument("--root")

    args = parser.parse_args(argv)
    if args.command == "env":
        return _env()
    if args.command == "datasets":
        return _inspect(None, None, True)
    if args.command == "inspect":
        return _inspect(args.dataset, args.root, False)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
