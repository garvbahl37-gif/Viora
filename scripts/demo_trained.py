#!/usr/bin/env python3
"""End-to-end demo with the TRAINED model: generate unseen videos, write them as
real mp4 files, decode + index them through the real pipeline, and answer.

Proves the "upload video → analysis → Q&A" loop on the domain the model actually
learned (a coloured square entering from left/right). Not real-world video
understanding — that needs the large datasets + GPUs (see README).

    python scripts/train_synthetic.py --out runs/synth   # first, produce the checkpoint
    python scripts/demo_trained.py --checkpoint runs/synth/best.pt
"""
from __future__ import annotations

import argparse
import os
import tempfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from viora.data.datasets.synthetic_tasks import (  # noqa: E402
    SimpleTokenizer,
    SyntheticQADataset,
    make_video,
)
from viora.inference.pipeline import VioraInferencePipeline  # noqa: E402
from viora.models.viora import VioraForVideoUnderstanding  # noqa: E402
from viora.training.checkpointing import load_checkpoint  # noqa: E402
from viora.utils.config import load_config  # noqa: E402
from viora.utils.logging import configure_logging  # noqa: E402


def write_mp4(video: torch.Tensor, path: str, fps: float) -> None:
    """Encode ``[C, T, H, W]`` float [0,1] to an mp4 (near-lossless x264, mpeg4 fallback)."""
    c, t, h, w = video.shape
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.options = {"crf": "8", "preset": "veryfast"}
    except Exception:  # noqa: BLE001
        stream = container.add_stream("mpeg4", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(t):
        arr = (video[:, i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def main() -> int:
    configure_logging("ERROR")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/synth/best.pt")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    tok = SimpleTokenizer()
    cfg = load_config("configs/model/viora_synth.yaml")
    model = VioraForVideoUnderstanding(cfg)
    meta = load_checkpoint(args.checkpoint, model)
    model.llm.video_token_id = tok.video_token_id
    model.eval()
    pipe = VioraInferencePipeline(model, device="cpu")
    ds = SyntheticQADataset(1, seed=12345, tokenizer=tok)  # for ground-truth params only

    print(f"loaded trained checkpoint (val answer-acc {meta['extra'].get('val_acc', '?')})\n")
    correct = 0
    total = 0
    for k in range(args.n):
        direction, color, t_appear, _ = ds._params(k + 777)  # unseen indices
        vid = make_video(direction, color, t_appear, size=cfg.vision.image_size,
                         num_frames=cfg.vision.num_frames, obj=8)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        write_mp4(vid, path, fps=4.0)
        idx = pipe.index(path)  # real decode + encode
        os.unlink(path)

        qa = [
            ("<video> which side does it enter from ?", direction),
            ("<video> what color is the object ?", color),
        ]
        line = [f"clip {k + 1}: object enters from {direction.upper()}, colour {color}, appears ~{t_appear/4:.2f}s"]
        for q, truth in qa:
            ans, conf = pipe.generate_answer(idx, q, tok)
            ok = ans.strip() == truth
            correct += ok
            total += 1
            mark = "✓" if ok else "✗"
            line.append(f"    Q: {q.replace('<video> ', '')}")
            line.append(f"       A: {ans!r}  (conf {conf:.2f}, uncalibrated)  truth={truth}  {mark}")
        # grounding: when did it appear?
        gout = pipe.model.grounding
        if gout is not None:
            from viora.models.temporal.temporal_pooling import masked_mean
            g = gout(idx.temporal.unsqueeze(0), masked_mean(idx.temporal.unsqueeze(0), None))
            gs, ge = g.to_seconds(torch.tensor([idx.duration]))
            line.append(f"    grounding (appears): {min(float(gs),float(ge)):.2f}s–{max(float(gs),float(ge)):.2f}s "
                        f"(truth ~{t_appear/4:.2f}s)")
        print("\n".join(line) + "\n")

    print(f"trained-model answer accuracy on {total} unseen questions: {correct}/{total} = {correct/total:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
