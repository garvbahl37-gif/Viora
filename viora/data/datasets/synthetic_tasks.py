"""Controlled synthetic video-understanding tasks.

Videos where the correct answer is **genuinely determined by the pixels**, so
training measures real learning (not memorization of text priors). A colored
square appears at a random frame and slides in from the left or right; questions
ask the entry side or the object colour, with a grounding target at the moment of
appearance.

This is a *controlled* benchmark — it demonstrates the architecture learns
spatiotemporal + question-conditioned reasoning end-to-end. It is NOT real-world
video understanding (that needs the large datasets in the registry + GPUs).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

# small closed vocabulary — question words + answer words + specials
_VOCAB = [
    "<pad>", "<video>", "which", "side", "does", "it", "enter", "from", "?",
    "what", "color", "is", "the", "object", "answer",
    "left", "right", "red", "green", "blue", "yellow",
]
_ANSWER_WORDS = ["left", "right", "red", "green", "blue", "yellow"]

PALETTE = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.9, 0.3),
    "blue": (0.2, 0.5, 1.0),
    "yellow": (1.0, 0.9, 0.2),
}
DIRECTIONS = ["left", "right"]


class SimpleTokenizer:
    """Fixed-vocab whitespace tokenizer for the synthetic task."""

    def __init__(self) -> None:
        self.itos = list(_VOCAB)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_token_id = self.stoi["<pad>"]
        self.video_token = "<video>"
        self.video_token_id = self.stoi["<video>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i != self.pad_token_id)

    def answer_ids(self) -> dict[str, int]:
        return {w: self.stoi[w] for w in _ANSWER_WORDS}


def make_video(
    direction: str, color: str, t_appear: int, *, size: int = 32, num_frames: int = 8, obj: int = 8
) -> torch.Tensor:
    """Render ``[3, T, H, W]`` in [0,1]: a coloured square entering from ``direction``.

    Empty before ``t_appear``; then it slides in from the given side.
    """
    vid = torch.zeros(3, num_frames, size, size)
    rgb = torch.tensor(PALETTE[color]).view(3, 1, 1)
    y = size // 2 - obj // 2
    active = list(range(t_appear, num_frames))
    n = max(1, len(active))
    for k, t in enumerate(active):
        frac = k / max(1, n - 1)  # 0 at appearance -> 1 at end
        # left = enters at left edge moving right; right = enters at right edge moving left
        x = int(frac * (size - obj)) if direction == "left" else int((1 - frac) * (size - obj))
        vid[:, t, y : y + obj, x : x + obj] = rgb
    vid += 0.03 * torch.rand_like(vid)            # light sensor noise
    return vid.clamp_(0, 1)


@dataclass
class SyntheticQAConfig:
    size: int = 32
    num_frames: int = 8
    fps: float = 4.0
    obj: int = 8


class SyntheticQADataset(Dataset):
    """Deterministic (seed-indexed) synthetic Video-QA + grounding samples."""

    def __init__(self, n: int, *, seed: int = 0, cfg: SyntheticQAConfig | None = None,
                 tokenizer: SimpleTokenizer | None = None, max_len: int = 12) -> None:
        self.n = n
        self.seed = seed
        self.cfg = cfg or SyntheticQAConfig()
        self.tok = tokenizer or SimpleTokenizer()
        self.max_len = max_len

    def __len__(self) -> int:
        return self.n

    def _params(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        direction = DIRECTIONS[int(torch.randint(0, 2, (1,), generator=g))]
        color = list(PALETTE)[int(torch.randint(0, 4, (1,), generator=g))]
        t_appear = int(torch.randint(0, self.cfg.num_frames - 2, (1,), generator=g))
        ask_color = bool(torch.randint(0, 2, (1,), generator=g))
        return direction, color, t_appear, ask_color

    def __getitem__(self, idx: int) -> dict:
        direction, color, t_appear, ask_color = self._params(idx)
        vid = make_video(direction, color, t_appear, size=self.cfg.size,
                         num_frames=self.cfg.num_frames, obj=self.cfg.obj)
        ts = torch.arange(self.cfg.num_frames, dtype=torch.float32) / self.cfg.fps

        if ask_color:
            question, answer = "<video> what color is the object ?", color
        else:
            question, answer = "<video> which side does it enter from ?", direction

        q_ids = self.tok.encode(question)
        a_id = self.tok.stoi[answer]
        ids = q_ids + [a_id]
        # labels: supervise only the answer token (last position)
        labels = [-100] * len(q_ids) + [a_id]
        # right-pad
        pad = self.max_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_token_id] * pad
        labels = labels + [-100] * pad

        appear_s = float(t_appear / self.cfg.fps)
        return {
            "video": vid,
            "timestamps": ts,
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
            "answer_id": a_id,
            "answer_pos": len(q_ids),          # index of the answer token
            "grounding": (appear_s, appear_s + 1.0 / self.cfg.fps),
            "duration": float(self.cfg.num_frames / self.cfg.fps),
        }


def collate(batch: list[dict]) -> dict:
    """Stack same-shape synthetic samples into model kwargs."""
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "timestamps": torch.stack([b["timestamps"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "answer_id": torch.tensor([b["answer_id"] for b in batch]),
        "answer_pos": torch.tensor([b["answer_pos"] for b in batch]),
        "grounding": [b["grounding"] for b in batch],
        "duration": torch.tensor([b["duration"] for b in batch]),
    }
