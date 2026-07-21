#!/usr/bin/env python3
"""Train Viora *deeply* on a controlled synthetic Video-QA + grounding task.

The answer is genuinely determined by the pixels (a coloured square entering from
the left/right), so rising held-out accuracy = real learning, not text priors.
Saves the best checkpoint so the API/UI can serve the trained model.

    python scripts/train_synthetic.py --steps 1500 --out runs/synth
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from viora.data.datasets.synthetic_tasks import (  # noqa: E402
    SimpleTokenizer,
    SyntheticQADataset,
    collate,
)
from viora.evaluation.temporal_grounding import temporal_iou  # noqa: E402
from viora.losses.temporal_grounding import TemporalGroundingLoss  # noqa: E402
from viora.models.viora import VioraForVideoUnderstanding  # noqa: E402
from viora.training.checkpointing import save_checkpoint  # noqa: E402
from viora.training.optimizer import build_optimizer  # noqa: E402
from viora.training.scheduler import build_scheduler  # noqa: E402
from viora.utils.config import TrainingConfig, load_config  # noqa: E402
from viora.utils.seed import set_seed  # noqa: E402


def _grounding_targets(batch, num_bins):
    s = torch.tensor([g[0] for g in batch["grounding"]])
    e = torch.tensor([g[1] for g in batch["grounding"]])
    return TemporalGroundingLoss.targets_from_seconds(s, e, batch["duration"], num_bins)


@torch.no_grad()
def evaluate(model, loader, num_bins):
    model.eval()
    correct = total = 0
    ious = []
    for b in loader:
        sb, eb = _grounding_targets(b, num_bins)
        out = model(b["video"], input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"], timestamps=b["timestamps"], grounding_targets=(sb, eb))
        lbl, logits = out.injected_labels, out.logits
        pos = (lbl != -100).float().argmax(1)               # answer position (post-injection)
        bsz = lbl.shape[0]
        idx = torch.arange(bsz)
        pred = logits[idx, pos - 1].argmax(-1)              # prediction of the answer token
        gold = lbl[idx, pos]
        correct += int((pred == gold).sum())
        total += bsz
        g = out.temporal_predictions
        ps, pe = g.to_seconds(b["duration"])
        for i in range(bsz):
            pr = (min(float(ps[i]), float(pe[i])), max(float(ps[i]), float(pe[i])))
            ious.append(temporal_iou(pr, b["grounding"][i]))
    return correct / total, sum(ious) / len(ious)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/synth")
    ap.add_argument("--eval-every", type=int, default=150)
    args = ap.parse_args()
    set_seed(0)

    cfg = load_config("configs/model/viora_synth.yaml")
    tok = SimpleTokenizer()
    assert cfg.llm.vocab_size == tok.vocab_size, (cfg.llm.vocab_size, tok.vocab_size)
    model = VioraForVideoUnderstanding(cfg)
    model.llm.video_token_id = tok.video_token_id           # align placeholder id with tokenizer
    num_bins = cfg.grounding.num_bins

    # questions are ~9 tokens; keep sequences short so injection (+Q visual tokens) fits max_length
    train_ds = SyntheticQADataset(4000, seed=0, tokenizer=tok, max_len=10)
    val_ds = SyntheticQADataset(600, seed=99_999, tokenizer=tok, max_len=10)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate)

    tcfg = TrainingConfig(max_steps=args.steps, lr=args.lr, warmup_steps=max(20, args.steps // 20),
                          weight_decay=0.02, grad_clip=1.0)
    opt = build_optimizer(model, tcfg)
    sched = build_scheduler(opt, tcfg)

    print(f"training on the synthetic task: {args.steps} steps, batch {args.batch}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    acc0, iou0 = evaluate(model, val_loader, num_bins)
    print(f"  step {0:5d} | val answer-acc {acc0:.3f} | grounding mIoU {iou0:.3f}  (chance≈0.33)")

    best = 0.0
    step = 0
    model.train()
    while step < args.steps:
        for b in train_loader:
            sb, eb = _grounding_targets(b, num_bins)
            out = model(b["video"], input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        labels=b["labels"], timestamps=b["timestamps"], grounding_targets=(sb, eb))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                acc, iou = evaluate(model, val_loader, num_bins)
                print(f"  step {step:5d} | loss {float(out.loss):.3f} | "
                      f"val answer-acc {acc:.3f} | grounding mIoU {iou:.3f}")
                if acc >= best:
                    best = acc
                    save_checkpoint(f"{args.out}/best.pt", model, optimizer=opt, scheduler=sched,
                                    step=step, config=cfg, extra={"val_acc": acc, "val_miou": iou})
                model.train()
            if step >= args.steps:
                break
    print(f"done. best val answer-acc {best:.3f}. checkpoint -> {args.out}/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
