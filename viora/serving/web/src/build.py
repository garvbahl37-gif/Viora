#!/usr/bin/env python3
"""Build the self-contained Viora Studio page.

Inlines the display/body fonts as base64 ``woff2`` data URIs (the page must work
with zero external requests — it is served under a strict CSP and also opens as a
local file), then writes the standalone document to ``../studio.html``.

    python viora/serving/web/src/build.py

Source of truth is ``studio.template.html`` (markup + CSS + JS with a ``/* @FONTS@ */``
placeholder). The generated ``studio.html`` is committed so the app runs without a
build step, but never edit it by hand — edit the template and rebuild.
"""
from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
TEMPLATE = HERE / "studio.template.html"
OUT = HERE.parent / "studio.html"

# (family, weight, filename) — kept in sync with the @font-face families in the template.
FONTS = [
    ("Sora", 600, "sora-600.woff2"),
    ("Sora", 700, "sora-700.woff2"),
    ("Hanken Grotesk", 400, "hanken-400.woff2"),
    ("Hanken Grotesk", 500, "hanken-500.woff2"),
    ("Hanken Grotesk", 600, "hanken-600.woff2"),
]


def font_faces() -> str:
    rules = []
    for family, weight, fname in FONTS:
        b64 = base64.b64encode((FONT_DIR / fname).read_bytes()).decode("ascii")
        rules.append(
            f'@font-face{{font-family:"{family}";font-style:normal;'
            f'font-weight:{weight};font-display:swap;'
            f'src:url(data:font/woff2;base64,{b64}) format("woff2");}}'
        )
    return "\n".join(rules)


def build() -> str:
    inner = TEMPLATE.read_text(encoding="utf-8").replace("/* @FONTS@ */", font_faces(), 1)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8"/>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        "<title>Viora Studio — it sees time</title>\n"
        '<meta name="description" content="Viora-1: temporal video intelligence. '
        'Ask a video a question, get an answer grounded in the exact moment it came from."/>\n'
        "</head>\n<body>\n" + inner + "\n</body>\n</html>\n"
    )


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"built {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(FONTS)} fonts embedded)")


if __name__ == "__main__":
    main()
