"""Greedy / sampling decode helpers.

Kept minimal: token-id greedy decoding against any LM exposing
``forward(input_ids=...) -> output.logits``. Full multimodal generation that
continues from ``inputs_embeds`` (visual prefix + prompt) is built on top of this
in the inference milestone.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_generate_ids(
    model,
    input_ids: torch.Tensor,          # [B, L]
    *,
    max_new_tokens: int = 20,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Autoregressively extend ``input_ids``; returns ``[B, L + n]`` including the prompt."""
    model.eval()
    ids = input_ids
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]  # [B, V]
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        else:
            nxt = logits.argmax(dim=-1)
        if eos_token_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
            finished = finished | (nxt == eos_token_id)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
    return ids
