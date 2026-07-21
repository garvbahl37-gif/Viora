"""Language-model integration.

Two backends behind one interface:

* :class:`DummyLanguageModel` — a tiny in-repo causal decoder so the whole model
  runs end-to-end (and trains) with **no download**; used for tests and smoke.
* Hugging Face ``AutoModelForCausalLM`` — loaded by name from config, with
  ``frozen`` / ``lora`` / ``full`` training modes. LoRA requires ``peft``; if it
  is missing we raise a clear dependency error rather than failing silently.

The adapter exposes exactly what the multimodal path needs: an input-embedding
layer, a ``<video>`` token id, and an ``inputs_embeds`` forward returning a loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from viora.models.common import TransformerBlock, make_norm
from viora.utils.config import LLMConfig
from viora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor          # [B, L, V]
    hidden_states: torch.Tensor   # [B, L, D]


class DummyLanguageModel(nn.Module):
    """A minimal causal transformer LM — real enough to train, tiny enough for CI."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)
        )
        self.norm = make_norm("layernorm", hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> LMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("provide input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)
        b, length, _ = inputs_embeds.shape
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length {self.max_length}")

        x = inputs_embeds + self.pos[:, :length]
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
        kpm = attention_mask.to(torch.bool) if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, self_attn_mask=causal, self_key_padding_mask=kpm)
        h = self.norm(x)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            if (shift_labels != -100).any():
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            else:
                loss = logits.sum() * 0.0  # keep graph, zero loss when nothing supervised
        return LMOutput(loss=loss, logits=logits, hidden_states=h)


class LLMAdapter(nn.Module):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.dummy:
            vocab = cfg.vocab_size or 32000
            hidden = cfg.hidden_size or 512
            self.model = DummyLanguageModel(vocab, hidden, max_length=cfg.max_length)
            self.tokenizer = None
            self.hidden_size = hidden
            self.vocab_size = vocab
            self.video_token_id = vocab - 1  # reserve the last id as <video>
        else:
            self._init_hf(cfg)

    def _init_hf(self, cfg: LLMConfig) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("transformers required for non-dummy LLM; `uv pip install transformers`") from e
        if not cfg.name_or_path:
            raise ValueError("llm.name_or_path must be set when llm.dummy is False")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [cfg.video_token]}
        )
        torch_dtype = getattr(torch, cfg.dtype, torch.float32)
        try:  # transformers >=5 renamed torch_dtype -> dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.name_or_path, dtype=torch_dtype, attn_implementation=cfg.attn_implementation
            )
        except TypeError:  # pragma: no cover - older transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.name_or_path, torch_dtype=torch_dtype, attn_implementation=cfg.attn_implementation
            )
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = len(self.tokenizer)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(cfg.video_token)
        self._apply_train_mode()

    def _apply_train_mode(self) -> None:
        mode = self.cfg.train_mode
        if mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad_(False)
            logger.info("LLM frozen (%d params)", sum(p.numel() for p in self.model.parameters()))
        elif mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:  # pragma: no cover
                raise ImportError("train_mode='lora' requires peft; `uv pip install peft`") from e
            lc = self.cfg.lora
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
                    target_modules=list(lc.target_modules), task_type="CAUSAL_LM",
                ),
            )
        elif mode != "full":
            raise ValueError(f"unknown train_mode '{mode}' (frozen|lora|full)")

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad or not trainable_only
        )
