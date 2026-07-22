"""Training engine: optimizer groups, scheduler, checkpoint resume, tiny loop."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from tests.unit.test_viora_e2e import tiny_viora_config
from viora.data.collators.video_text_collator import VideoTextCollator
from viora.data.datasets.base import SyntheticVideoDataset
from viora.models.viora import VioraForVideoUnderstanding
from viora.training.checkpointing import load_checkpoint, save_checkpoint
from viora.training.optimizer import build_optimizer
from viora.training.scheduler import build_scheduler
from viora.training.trainer import Trainer
from viora.utils.config import TrainingConfig


def _train_cfg(tmp_path, **kw):
    defaults = {"max_steps": 4, "batch_size": 2, "grad_accum": 1, "warmup_steps": 1,
                "log_every": 2, "save_every": 0, "freeze_llm": False, "num_workers": 0,
                "precision": "fp32"}
    defaults.update(kw)
    return TrainingConfig(output_dir=str(tmp_path), **defaults)


def test_optimizer_splits_weight_decay(tmp_path):
    model = VioraForVideoUnderstanding(tiny_viora_config())
    opt = build_optimizer(model, _train_cfg(tmp_path))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] > 0 and no_decay["weight_decay"] == 0.0
    assert len(decay["params"]) > 0 and len(no_decay["params"]) > 0


def test_scheduler_warmup_then_decay(tmp_path):
    model = VioraForVideoUnderstanding(tiny_viora_config())
    cfg = _train_cfg(tmp_path, max_steps=10, warmup_steps=3, lr=1e-3, scheduler="cosine")
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[2]          # warming up
    assert lrs[-1] < lrs[3]         # decaying after peak


def test_checkpoint_round_trip(tmp_path):
    cfg = tiny_viora_config()
    model = VioraForVideoUnderstanding(cfg)
    tcfg = _train_cfg(tmp_path)
    opt = build_optimizer(model, tcfg)
    sched = build_scheduler(opt, tcfg)
    # perturb a weight so state is non-trivial
    with torch.no_grad():
        next(model.parameters()).add_(0.5)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer=opt, scheduler=sched, step=7, epoch=1, config=cfg)

    fresh = VioraForVideoUnderstanding(cfg)
    fopt = build_optimizer(fresh, tcfg)
    fsched = build_scheduler(fopt, tcfg)
    meta = load_checkpoint(path, fresh, optimizer=fopt, scheduler=fsched)
    assert meta["step"] == 7 and meta["epoch"] == 1
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.allclose(p, q)


def test_export_trained_weights_shrinks_and_reloads(tmp_path):
    from viora.training.checkpointing import export_trained_weights

    cfg = tiny_viora_config()
    model = VioraForVideoUnderstanding(cfg)
    for p in model.vision.parameters():   # freeze a chunk -> frozen/trainable split
        p.requires_grad_(False)
    full = tmp_path / "full.pt"
    save_checkpoint(full, model, step=5)

    out = tmp_path / "trained.pt"
    kept, total = export_trained_weights(model, full, out)
    assert kept < total                                    # frozen params dropped
    assert out.stat().st_size < full.stat().st_size        # genuinely smaller

    fresh = VioraForVideoUnderstanding(cfg)
    for p in fresh.vision.parameters():   # same frozen split as training (config does this for real)
        p.requires_grad_(False)
    load_checkpoint(out, fresh, strict=False)              # loads trained subset, frozen from init
    # the exported (trainable) params match the source model
    src = dict(model.named_parameters())
    for n, p in fresh.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p, src[n])


def test_checkpoint_architecture_mismatch_raises(tmp_path):
    a = VioraForVideoUnderstanding(tiny_viora_config())
    path = tmp_path / "a.pt"
    save_checkpoint(path, a)
    big = copy.deepcopy(tiny_viora_config())
    big.resampler.num_queries = 99  # different shape
    b = VioraForVideoUnderstanding(big)
    with pytest.raises(RuntimeError, match="does not match"):
        load_checkpoint(path, b)


def test_tiny_training_loop_runs(tmp_path):
    model = VioraForVideoUnderstanding(tiny_viora_config())
    ds = SyntheticVideoDataset(size=8, num_frames=8, image_size=32)

    def tok(texts):
        n = len(texts)
        ids = torch.randint(1, 60, (n, 6))
        ids[:, 0] = model.llm.video_token_id
        return {"input_ids": ids, "attention_mask": torch.ones(n, 6, dtype=torch.long),
                "labels": ids.clone()}

    loader = DataLoader(ds, batch_size=2, collate_fn=VideoTextCollator(tokenize_fn=tok))
    cfg = _train_cfg(tmp_path, max_steps=4)
    p_before = next(model.parameters()).clone()
    trainer = Trainer(model, cfg, device="cpu")
    summary = trainer.train(loader)

    assert trainer.step == 4
    assert "loss" in summary and summary["loss"] > 0
    assert (tmp_path / "final.pt").exists()          # checkpoint written
    assert (tmp_path / "run_meta.json").exists()     # experiment tracking
    # weights actually moved
    assert not torch.allclose(p_before, next(trainer.model.parameters()))


def _toy_tok(texts):
    n = len(texts)
    return {"input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
            "labels": torch.ones(n, 4, dtype=torch.long)}


def test_batch_to_device_moves_tensors_and_preserves_meta():
    ds = SyntheticVideoDataset(size=2, num_frames=8, image_size=32)
    batch = VideoTextCollator(tokenize_fn=_toy_tok)([ds[0], ds[1]])
    moved = batch.to("cpu")
    for t in (moved.video, moved.frame_mask, moved.timestamps,
              moved.input_ids, moved.attention_mask, moved.labels):
        assert t.device.type == "cpu"
    assert moved.meta == batch.meta

    # untokenized batch: text fields stay None (no crash on .to())
    plain = VideoTextCollator()([ds[0]])
    m2 = plain.to("cpu")
    assert m2.input_ids is None and m2.attention_mask is None and m2.labels is None


def test_trainer_moves_batch_to_device(tmp_path):
    """Regression: the trainer MUST move each batch to its device before the forward
    pass. A CPU batch against CUDA weights was the training-time crash; on CPU CI we
    assert both the dispatch and that train_step applies it."""
    model = VioraForVideoUnderstanding(tiny_viora_config())
    trainer = Trainer(model, _train_cfg(tmp_path, max_steps=1), device="cpu")

    # dispatch: VideoTextBatch / dict / tensor all land on trainer.device
    ds = SyntheticVideoDataset(size=2, num_frames=8, image_size=32)
    batch = VideoTextCollator(tokenize_fn=_toy_tok)([ds[0], ds[1]])
    assert trainer._to_device(batch).video.device == trainer.device
    d = trainer._to_device({"x": torch.ones(2), "n": 5})
    assert d["x"].device == trainer.device and d["n"] == 5
    assert trainer._to_device(torch.ones(2)).device == trainer.device

    # integration: the training loop actually routes batches through _to_device
    seen = {"called": False}
    orig = trainer._to_device
    trainer._to_device = lambda b: (seen.__setitem__("called", True), orig(b))[1]
    loader = DataLoader(ds, batch_size=2, collate_fn=VideoTextCollator(tokenize_fn=_toy_tok))
    trainer.train(loader)
    assert seen["called"] is True


def test_checkpoint_pruning_keeps_last_k(tmp_path):
    import glob

    model = VioraForVideoUnderstanding(tiny_viora_config())
    trainer = Trainer(model, _train_cfg(tmp_path, keep_last_checkpoints=2), device="cpu")
    for s in (100, 200, 300):
        trainer.step = s
        trainer.save(f"step_{s}")
    kept = sorted(p.rsplit("step_", 1)[1] for p in glob.glob(str(tmp_path / "step_*.pt")))
    assert kept == ["200.pt", "300.pt"]                 # oldest pruned to bound disk
    trainer.save("final")
    assert (tmp_path / "final.pt").exists()             # 'final' is never pruned
    assert (tmp_path / "step_300.pt").exists()


def test_checkpoint_pruning_recovers_from_overflow(tmp_path):
    """A session that already piled up checkpoints (disk full): the next save prunes
    BEFORE writing, so it frees space and bounds the total to keep_last."""
    import glob

    model = VioraForVideoUnderstanding(tiny_viora_config())
    trainer = Trainer(model, _train_cfg(tmp_path, keep_last_checkpoints=3), device="cpu")
    for s in range(200, 1401, 200):                     # 7 pre-existing checkpoints
        (tmp_path / f"step_{s}.pt").write_bytes(b"x")
    trainer.step = 1600
    trainer.save("step_1600")
    remaining = sorted(int(p.rsplit("step_", 1)[1].split(".")[0])
                       for p in glob.glob(str(tmp_path / "step_*.pt")))
    assert remaining == [1200, 1400, 1600]              # bounded to keep_last (2 kept + new)


def test_default_step_fn_computes_contrastive_when_weighted(tmp_path):
    from viora.training.trainer import default_step_fn

    model = VioraForVideoUnderstanding(tiny_viora_config())  # default weights include contrastive
    assert model.loss_manager.weights.get("contrastive", 0) > 0
    ds = SyntheticVideoDataset(size=2, num_frames=8, image_size=32)
    batch = VideoTextCollator(tokenize_fn=_toy_tok)([ds[0], ds[1]])
    out = default_step_fn(model, batch)
    assert "contrastive" in out.losses                  # retrieval head actually gets a gradient


def test_hf_tokenizer_appends_eos():
    import sys
    sys.path.insert(0, "scripts")
    from train import hf_tokenize_fn

    class _Tok:
        eos_token = "<eos>"
        def __call__(self, texts, **kw):
            # record what was tokenized; emit 1 id per whitespace token
            _Tok.seen = texts
            rows = [[7] * len(t.split()) for t in texts]
            n = max(len(r) for r in rows)
            ids = torch.zeros(len(rows), n, dtype=torch.long)
            attn = torch.zeros(len(rows), n, dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, :len(r)] = torch.tensor(r)
                attn[i, :len(r)] = 1
            return {"input_ids": ids, "attention_mask": attn}

    hf_tokenize_fn(_Tok(), max_length=64)(["a caption"])
    assert all(t.endswith("<eos>") for t in _Tok.seen)  # EOS appended so the model learns to stop


@pytest.mark.parametrize(
    ("scale_before", "scale_after", "expect_steps"),
    [(65536.0, 65536.0, 1),   # no overflow -> optimizer stepped -> schedule advances
     (65536.0, 32768.0, 0)],  # fp16 overflow -> scaler skipped -> schedule must NOT advance
)
def test_scheduler_advances_only_on_real_optimizer_step(
    tmp_path, monkeypatch, scale_before, scale_after, expect_steps
):
    model = VioraForVideoUnderstanding(tiny_viora_config())
    trainer = Trainer(model, _train_cfg(tmp_path, grad_clip=0.0), device="cpu")
    trainer.use_scaler = True  # force the fp16 skip-detection path

    scales = iter([scale_before, scale_after])
    monkeypatch.setattr(trainer.scaler, "get_scale", lambda: next(scales))
    monkeypatch.setattr(trainer.scaler, "step", lambda opt: None)
    monkeypatch.setattr(trainer.scaler, "update", lambda: None)
    seen = {"n": 0}
    monkeypatch.setattr(trainer.scheduler, "step",
                        lambda *a, **k: seen.__setitem__("n", seen["n"] + 1))

    trainer._optimizer_step()
    assert seen["n"] == expect_steps


def test_nan_loss_is_flagged_not_hidden(tmp_path):
    model = VioraForVideoUnderstanding(tiny_viora_config())

    class _Out:
        loss = torch.tensor(float("nan"))
        losses = {"lm": float("nan")}
        diagnostics = {"nonfinite_losses": ["lm"]}

    trainer = Trainer(model, _train_cfg(tmp_path), device="cpu", step_fn=lambda m, b: _Out())
    stats = trainer.train_step(object())
    assert stats["skipped"] is True
