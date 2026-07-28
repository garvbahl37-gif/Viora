"""Offline video inference: index a video once, then answer questions against it.

Indexing runs the encoder over sampled frames and stores clip/event/temporal
tokens + timestamps to a versioned ``.viora`` file (safetensors — safe to load).
Answering retrieves the temporal regions most relevant to the question (cosine
baseline) and returns **evidence with timestamps** plus an *uncalibrated* score;
answers are honestly marked when the underlying model is untrained.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from viora import __version__
from viora.data.decoding.video_decoder import VideoDecoder
from viora.data.preprocessing.transforms import VideoTransform
from viora.data.sampling.base import build_sampler
from viora.models.multimodal.token_injection import build_multimodal_inputs
from viora.models.temporal.temporal_pooling import masked_mean
from viora.models.viora import VioraForVideoUnderstanding

INDEX_FORMAT_VERSION = 1


@dataclass
class Evidence:
    start: float
    end: float
    score: float


@dataclass
class VioraAnswer:
    answer: str
    evidence: list[Evidence]
    score: float
    score_type: str = "uncalibrated_model_score"
    events_used: list[float] = field(default_factory=list)  # timestamps
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class VideoIndex:
    resampled: torch.Tensor        # [Q, D]
    event_tokens: torch.Tensor     # [E, D]
    temporal: torch.Tensor         # [T', D]
    timestamps: torch.Tensor       # [T'] seconds
    duration: float
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tensors = {
            "resampled": self.resampled.contiguous().cpu(),
            "event_tokens": self.event_tokens.contiguous().cpu(),
            "temporal": self.temporal.contiguous().cpu(),
            "timestamps": self.timestamps.contiguous().cpu(),
        }
        meta = {
            "format_version": str(INDEX_FORMAT_VERSION),
            "viora_version": __version__,
            "duration": str(self.duration),
            "metadata": json.dumps(self.metadata),
        }
        save_file(tensors, str(path), metadata=meta)

    @classmethod
    def load(cls, path: str | Path) -> VideoIndex:
        path = Path(path)
        tensors, meta = {}, {}
        with safe_open(str(path), framework="pt") as f:  # safetensors: no code execution
            meta = f.metadata() or {}
            for k in f.keys():  # noqa: SIM118
                tensors[k] = f.get_tensor(k)
        fmt = int(meta.get("format_version", 0))
        if fmt != INDEX_FORMAT_VERSION:
            raise ValueError(f"incompatible index format v{fmt} (expected v{INDEX_FORMAT_VERSION})")
        return cls(
            resampled=tensors["resampled"], event_tokens=tensors["event_tokens"],
            temporal=tensors["temporal"], timestamps=tensors["timestamps"],
            duration=float(meta.get("duration", 0.0)),
            metadata=json.loads(meta.get("metadata", "{}")),
        )


class VioraInferencePipeline:
    def __init__(
        self,
        model: VioraForVideoUnderstanding,
        *,
        decoder: VideoDecoder | None = None,
        sampler: str = "uniform",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.decoder = decoder or VideoDecoder()
        cfg = model.cfg.vision
        self.num_frames = cfg.num_frames
        self.sampler = build_sampler(sampler, cfg.num_frames)
        self.transform = VideoTransform(size=cfg.image_size)
        # Temporal stride from frames -> temporal tokens. The 3D ViT downsamples time by
        # tubelet_size; the PRETRAINED frame encoder keeps one token per frame (stride 1).
        # Using tubelet_size for the pretrained path produced HALF as many timestamps as
        # temporal tokens, so ask()'s topk over tokens indexed past the end of
        # `timestamps` -> IndexError. Mirrors default_step_fn's stride in the trainer.
        self.tubelet = cfg.tubelet_size if cfg.backbone == "scratch" else 1

    # ------------------------------------------------------------- indexing
    @torch.no_grad()
    def index(self, video_path: str | Path, *, target_fps: float | None = None) -> VideoIndex:
        clip = self.decoder.decode_clip(video_path, target_fps=target_fps)
        sel = self.sampler.select(clip.num_frames, clip.timestamps, clip.frames.numpy())
        frames = clip.frames[sel.indices]  # [T, H, W, C]
        video = (frames.float() / 255.0).permute(3, 0, 1, 2)  # [C, T, H, W]
        video = self.transform(video).unsqueeze(0).to(self.device)  # [1, C, T, H, W]
        frame_ts = torch.as_tensor(sel.timestamps, dtype=torch.float32, device=self.device).unsqueeze(0)

        enc = self.model.encode_video(video, timestamps=frame_ts)
        # per-temporal-token timestamps (mean-pool frame ts over each tubelet)
        tprime = enc["temporal"].shape[1]
        usable = (frame_ts.shape[1] // self.tubelet) * self.tubelet
        tok_ts = frame_ts[:, :usable].reshape(1, -1, self.tubelet).mean(-1).squeeze(0)[:tprime]

        return VideoIndex(
            resampled=enc["resampled"].squeeze(0),
            event_tokens=enc["event_tokens"].squeeze(0),
            temporal=enc["temporal"].squeeze(0),
            timestamps=tok_ts,
            duration=float(clip.timestamps[-1]),
            metadata={"path": str(video_path), "num_frames": int(frames.shape[0])},
        )

    # ------------------------------------------------------------- answering
    def _embed_question(self, question: str) -> torch.Tensor:
        """Question -> [D_llm] embedding (real tokenizer if present, else hashed ids)."""
        tok = getattr(self.model.llm, "tokenizer", None)
        emb = self.model.llm.get_input_embeddings()
        if tok is not None:
            ids = torch.tensor([tok(question)["input_ids"]], device=self.device)
        else:
            vocab = self.model.llm.vocab_size
            ids = torch.tensor(
                [[abs(hash(w)) % (vocab - 1) for w in question.split()] or [1]], device=self.device
            )
        return emb(ids).mean(dim=1).squeeze(0)  # [D_llm]

    @staticmethod
    def _group_segments(indices: list[int], timestamps: torch.Tensor, duration: float) -> list[tuple[float, float]]:
        """Merge consecutive temporal indices into (start, end) time segments."""
        if not indices:
            return []
        indices = sorted(indices)
        segs, run = [], [indices[0]]
        for i in indices[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                segs.append(run)
                run = [i]
        segs.append(run)
        out = []
        span = duration / max(1, len(timestamps))
        for run in segs:
            s = float(timestamps[run[0]])
            e = float(timestamps[run[-1]]) + span
            out.append((s, min(e, duration)))
        return out

    @torch.no_grad()
    def ask(self, index: VideoIndex, question: str, *, top_k: int = 3) -> VioraAnswer:
        temporal = index.temporal.to(self.device)          # [T', D]
        q_llm = self._embed_question(question)             # [D_llm]

        # question-conditioned temporal retrieval (cosine in the shared retrieval space)
        vid = self.model.retrieval.encode_video(temporal)  # [T', P]
        qv = self.model.retrieval.encode_text(q_llm.unsqueeze(0))  # [1, P]
        sims = F.cosine_similarity(vid, qv, dim=-1)         # [T']
        k = min(top_k, temporal.shape[0])
        scores, idx = sims.topk(k)
        evidence_segments = self._group_segments(idx.tolist(), index.timestamps, index.duration)
        evidence = [Evidence(s, e, float(scores.max())) for s, e in evidence_segments]
        used_ts = [round(float(index.timestamps[i]), 2) for i in idx.tolist()]

        # grounding head provides a global evidence span (query = pooled temporal)
        query = masked_mean(temporal.unsqueeze(0), None)
        g = self.model.grounding(temporal.unsqueeze(0), query) if self.model.grounding else None
        if g is not None:
            gs, ge = g.to_seconds(torch.tensor([index.duration]))
            lo, hi = sorted((float(gs), float(ge)))  # order + clamp (untrained heads can invert)
            lo = max(0.0, min(lo, index.duration))
            hi = max(0.0, min(hi, index.duration))
            evidence.append(Evidence(lo, hi, float(g.confidence)))

        score = float(scores.max().clamp(0, 1)) if k else 0.0
        trained = (
            bool(self.model.llm.cfg.name_or_path)
            and not self.model.llm.cfg.dummy
            and self.model.llm.tokenizer is not None
        )
        if trained:
            answer, _ = self.generate_answer(index, question, self.model.llm.tokenizer)
        else:
            best = evidence_segments[0] if evidence_segments else (0.0, index.duration)
            answer = (
                f"The most relevant moment is {best[0]:.1f}s–{best[1]:.1f}s. "
                f"(Model is untrained — this is an evidence-only, retrieval-based response.)"
            )
        return VioraAnswer(
            answer=answer, evidence=evidence, score=score,
            events_used=used_ts,
            diagnostics={"model_trained": trained, "retrieval": "cosine", "top_k": k},
        )

    @torch.no_grad()
    def _greedy_decode(
        self, index: VideoIndex, prompt: str, tokenizer, max_new_tokens: int
    ) -> tuple[str, float]:
        """Inject the video at ``<video>`` and greedily continue ``prompt`` to EOS.

        Returns ``(text, confidence)`` where confidence is the (uncalibrated) mean
        softmax prob of the chosen tokens.
        """
        emb = self.model.llm.get_input_embeddings()
        projected = self.model.projector(index.resampled.unsqueeze(0).to(self.device))  # [1,Q,D_llm]
        ids = torch.tensor([tokenizer.encode(prompt)], device=self.device)
        prompt_len = ids.shape[1]
        eos_id = getattr(tokenizer, "eos_token_id", None)
        confs: list[float] = []
        for _ in range(max_new_tokens):
            mm = build_multimodal_inputs(
                ids, projected, emb, video_token_id=self.model.llm.video_token_id,
                attention_mask=torch.ones_like(ids),
            )
            out = self.model.llm(mm.inputs_embeds, mm.attention_mask)
            probs = out.logits[:, -1].softmax(-1)
            conf, nxt = probs.max(-1)
            confs.append(float(conf))
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            if eos_id is not None and int(nxt) == eos_id:
                break
        text = tokenizer.decode(ids[0, prompt_len:].tolist(), skip_special_tokens=True).strip()
        return text, (sum(confs) / len(confs) if confs else 0.0)

    @torch.no_grad()
    def generate_answer(
        self, index: VideoIndex, question: str, tokenizer, *, max_new_tokens: int = 32
    ) -> tuple[str, float]:
        """Greedily decode an answer from a **trained** model (QA prompt used in training)."""
        prompt = f"{self.model.llm.cfg.video_token}\nQuestion: {question}\nAnswer:"
        return self._greedy_decode(index, prompt, tokenizer, max_new_tokens)

    @torch.no_grad()
    def caption(self, index: VideoIndex, *, max_new_tokens: int = 40) -> tuple[str, float]:
        """Greedily decode a caption, matching the ``<video>\\n{caption}`` training format.

        This is the right entry point for a model trained on captions (e.g. MSR-VTT);
        ``generate_answer`` uses the QA prompt instead.
        """
        tok = getattr(self.model.llm, "tokenizer", None)
        if tok is None:
            raise RuntimeError("caption() needs a real LLM tokenizer (train a non-dummy model)")
        prompt = f"{self.model.llm.cfg.video_token}\n"
        return self._greedy_decode(index, prompt, tok, max_new_tokens)

    def index_and_ask(self, video_path: str | Path, question: str, **kw) -> VioraAnswer:
        return self.ask(self.index(video_path), question, **kw)
