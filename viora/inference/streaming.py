"""Streaming video understanding.

Feed successive chunks; each is encoded once, its event tokens are timestamped
(via the event→time attention) and written to bounded temporal memory. Questions
are answered by retrieving from memory — the whole video is never reprocessed.
Each memory entry keeps its timestamp, embedding, and importance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from viora.data.preprocessing.transforms import VideoTransform
from viora.inference.memory_manager import MemoryManager
from viora.inference.pipeline import Evidence, VioraAnswer, VioraInferencePipeline
from viora.models.temporal.temporal_memory import TemporalMemory
from viora.models.viora import VioraForVideoUnderstanding
from viora.utils.config import MemoryConfig


@dataclass
class ChunkResult:
    num_events: int
    time_range: tuple[float, float]
    memory: dict


class StreamingVioraEngine:
    def __init__(self, model: VioraForVideoUnderstanding, *, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        mem = model.memory or TemporalMemory(MemoryConfig(dim=model.cfg.vision.dim))
        self.mm = MemoryManager(mem, device=device)
        self.transform = VideoTransform(size=model.cfg.vision.image_size)
        self.tubelet = model.cfg.vision.tubelet_size
        self.clock = 0.0  # running end-of-stream time (seconds)

    def reset_memory(self) -> None:
        self.mm.reset()
        self.clock = 0.0

    @torch.no_grad()
    def add_video_chunk(
        self, video: torch.Tensor, *, fps: float = 4.0, chunk_start: float | None = None
    ) -> ChunkResult:
        """``video``: ``[C, T, H, W]`` (a chunk). Encodes it and writes events to memory."""
        start = self.clock if chunk_start is None else chunk_start
        t = video.shape[1]
        x = self.transform(video).unsqueeze(0).to(self.device)
        frame_ts = start + torch.arange(t, dtype=torch.float32, device=self.device) / fps
        enc = self.model.encode_video(x, timestamps=frame_ts.unsqueeze(0))

        # per-token timestamps, then event timestamps via attention over time
        tprime = enc["temporal"].shape[1]
        usable = (t // self.tubelet) * self.tubelet
        tok_ts = frame_ts[:usable].reshape(-1, self.tubelet).mean(-1)[:tprime]  # [T']
        attn = enc["event_attention"].squeeze(0)  # [E, T']
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
        event_ts = attn @ tok_ts  # [E] attention-weighted event time

        events = enc["event_tokens"].squeeze(0)  # [E, D]
        self.mm.add_events(events, event_ts)
        self.clock = float(frame_ts[-1]) + 1.0 / fps
        return ChunkResult(
            num_events=int(events.shape[0]),
            time_range=(float(frame_ts[0]), float(frame_ts[-1])),
            memory=self.mm.summary(),
        )

    @torch.no_grad()
    def ask(self, question: str, *, top_k: int = 3) -> VioraAnswer:
        # reuse the pipeline's question-embedding logic (duck-typed on .model/.device)
        q_llm = VioraInferencePipeline._embed_question(self, question)
        # project into the retrieval space and score memory tokens
        mem_tokens, mem_ts = self.mm.memory.read(self.mm.state)
        if mem_tokens.shape[0] == 0:
            return VioraAnswer("No video has been streamed yet.", [], 0.0,
                               diagnostics={"memory_empty": True})
        vid = self.model.retrieval.encode_video(mem_tokens)
        qv = self.model.retrieval.encode_text(q_llm.unsqueeze(0))
        sims = torch.nn.functional.cosine_similarity(vid, qv, dim=-1)
        k = min(top_k, mem_tokens.shape[0])
        scores, idx = sims.topk(k)
        span = 1.0
        evidence = [Evidence(float(mem_ts[i]), float(mem_ts[i]) + span, float(s))
                    for i, s in zip(idx.tolist(), scores.tolist(), strict=True)]
        best_t = float(mem_ts[idx[0]])
        return VioraAnswer(
            answer=f"Most relevant streamed moment near {best_t:.1f}s. "
                   f"(Untrained model — evidence-only, retrieval-based.)",
            evidence=evidence, score=float(scores.max().clamp(0, 1)),
            events_used=[round(float(mem_ts[i]), 2) for i in idx.tolist()],
            diagnostics={"memory": self.mm.summary()},
        )

    def get_memory_summary(self) -> dict:
        return {**self.mm.summary(), "stream_time_s": round(self.clock, 2)}
