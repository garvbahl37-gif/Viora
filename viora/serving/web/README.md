# Viora Studio — web client

The front door to Viora: load a video, ask it a question, and get an answer
**grounded in the exact moment it came from**. The signature element is the
**Temporal Spectrum** — a timeline where the flow of time reads as a cool
violet→aqua field and the single warm-amber ignition marks the evidence Viora is
pointing at.

## Files

| File | Role |
|------|------|
| `studio.html` | **Generated, committed** self-contained page (fonts inlined). Served as-is; opens as a local file. Do not hand-edit. |
| `src/studio.template.html` | Source of truth — markup + CSS + JS with a `/* @FONTS@ */` placeholder. |
| `src/build.py` | Inlines fonts as base64 `woff2` data URIs and writes `studio.html`. |
| `src/fonts/*.woff2` | Sora (display) + Hanken Grotesk (body), latin subsets, OFL. |

Rebuild after editing the template:

```bash
python viora/serving/web/src/build.py
```

## Design

- **Identity:** a committed dark "night studio" — the right register for video
  tooling. Neutrals are hue-biased toward the iris accent, not flat grey.
- **Type:** Sora (display) + Hanken Grotesk (body) + system `ui-monospace` for
  all instrumentation (timestamps, scores, token counts are *data* → tabular
  mono). Deliberately not Inter / Space Grotesk.
- **Accessibility:** every meaningful text/background pair passes WCAG AA;
  keyboard focus is visible; the timeline exposes `role="slider"` with arrow-key
  seeking; `prefers-reduced-motion` disables the ambient canvas and animations.

## Data / honesty

Everything shown is **illustrative sample data** for `delivery_cam.mp4`. The model
is not yet trained, so scores are surfaced as **uncalibrated model scores**, never
as calibrated probabilities (see the in-page footer). A `VioraClient` abstraction
speaks to a deterministic **mock engine** today; once `viora/serving/api.py` is up
it will call `POST /video/ask` with the same request/response shape, so the UI
does not change when the real pipeline lands.
