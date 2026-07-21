#!/usr/bin/env python3
"""Train Viora's OWN production model locally on the Mac (no wrapper, no external API).

Frozen SigLIP + frozen Qwen2.5, with Viora's temporal/resampler/projector bridge +
LoRA trained on a controlled synthetic task (a coloured square entering from the
left/right). Rising held-out accuracy = the real production model learns end-to-end
on this machine. It's the same model the cloud recipe scales to real data.

    python scripts/train_pragmatic_local.py --steps 300
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from viora.data.datasets.synthetic_tasks import DIRECTIONS, make_video  # noqa: E402
from viora.data.preprocessing.transforms import VideoTransform  # noqa: E402
from viora.models.viora import VioraForVideoUnderstanding  # noqa: E402
from viora.training.optimizer import build_optimizer  # noqa: E402
from viora.training.scheduler import build_scheduler  # noqa: E402
from viora.utils.config import TrainingConfig, load_config  # noqa: E402
from viora.utils.seed import set_seed  # noqa: E402

_QUESTION = {
    "direction": "Which side does the object enter from?",   # needs spatial position (hard for mean-pool)
    "color": "What color is the object?",                    # content — mean-pool preserves it
}


def make_dataset(n, tok, ans_ids, *, task, colors, size, num_frames, seed, transform,
                 obj_frac=0.25, max_len=24):
    """Build (video, input_ids, labels, attention_mask) items with the real tokenizer."""
    items = []
    obj_size = max(4, int(size * obj_frac))
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        direction = DIRECTIONS[int(torch.randint(0, 2, (1,), generator=g))]
        color = colors[int(torch.randint(0, len(colors), (1,), generator=g))]
        t_appear = int(torch.randint(0, num_frames - 1, (1,), generator=g))
        video = transform(make_video(direction, color, t_appear, size=size,
                                     num_frames=num_frames, obj=obj_size))
        answer = direction if task == "direction" else color
        prompt = f"<video>\n{_QUESTION[task]}\nAnswer:"
        prefix = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = prefix + [ans_ids[answer]]
        labels = [-100] * len(prefix) + [ans_ids[answer]]
        pad = max_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [tok.pad_token_id] * pad
        labels = labels + [-100] * pad
        items.append({
            "video": video,
            "input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        })
    return items


def collate(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for b in loader:
        out = model(b["video"], input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"])
        lbl, logits = out.injected_labels, out.logits
        pos = (lbl != -100).float().argmax(1)
        idx = torch.arange(lbl.shape[0])
        pred = logits[idx, pos - 1].argmax(-1)
        gold = lbl[idx, pos]
        correct += int((pred == gold).sum())
        total += lbl.shape[0]
    return correct / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="runs/pragmatic_local")
    ap.add_argument("--task", choices=["direction", "color"], default="color",
                    help="color = content (mean-pool learns it); direction = spatial (needs patch tokens)")
    ap.add_argument("--obj-frac", type=float, default=0.25,
                    help="object size as a fraction of the frame (1.0 = full frame = strong signal)")
    args = ap.parse_args()
    set_seed(0)

    from viora.data.datasets.synthetic_tasks import PALETTE

    cfg = load_config("configs/model/viora_pragmatic.yaml")
    cfg.vision.num_frames = args.frames  # fewer frames = faster on the Mac
    print(f"building pragmatic model for the '{args.task}' task "
          f"(frozen SigLIP + frozen Qwen + trainable bridge + LoRA)...")
    model = VioraForVideoUnderstanding(cfg)
    tok = model.llm.tokenizer

    # keep only single-token answer words (with a leading space) for a clean metric
    def single_token(words):
        return [w for w in words if len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]

    colors = single_token(list(PALETTE))
    words = list(DIRECTIONS) if args.task == "direction" else colors
    ans_ids = {w: tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words}
    chance = 1.0 / len(set(words))
    c = model.count_parameters()
    print(f"params: total {c['total']:,} | trainable {c['trainable']:,} "
          f"({100*c['trainable']/c['total']:.1f}%) — bridge + LoRA only | chance {chance:.2f}")

    tf = VideoTransform(size=cfg.vision.image_size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train = make_dataset(400, tok, ans_ids, task=args.task, colors=colors, size=cfg.vision.image_size,
                         num_frames=args.frames, seed=0, transform=tf, obj_frac=args.obj_frac)
    val = make_dataset(80, tok, ans_ids, task=args.task, colors=colors, size=cfg.vision.image_size,
                       num_frames=args.frames, seed=999, transform=tf, obj_frac=args.obj_frac)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=8, collate_fn=collate)

    tcfg = TrainingConfig(max_steps=args.steps, lr=args.lr, warmup_steps=max(10, args.steps // 20),
                          weight_decay=0.0, grad_clip=1.0)
    opt = build_optimizer(model, tcfg)
    sched = build_scheduler(opt, tcfg)

    print(f"val accuracy before training: {accuracy(model, val_loader):.3f}  (chance {chance:.2f})")
    step, t0 = 0, time.time()
    model.train()
    while step < args.steps:
        for b in train_loader:
            out = model(b["video"], input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        labels=b["labels"])
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0 or step == args.steps:
                acc = accuracy(model, val_loader)
                rate = step / (time.time() - t0)
                print(f"  step {step:4d} | loss {float(out.loss.detach()):.3f} | "
                      f"val acc {acc:.3f} | {rate:.2f} steps/s")
                model.train()
            if step >= args.steps:
                break
    print(f"done in {time.time()-t0:.0f}s. This is Viora's OWN model, trained locally — no wrapper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
