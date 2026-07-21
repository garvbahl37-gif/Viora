"""Multimodal bridge: resampler, Q-Former, projector, and visual-token injection."""

from __future__ import annotations

import torch

from viora.models.language.llm_adapter import DummyLanguageModel
from viora.models.multimodal.projector import MultimodalProjector
from viora.models.multimodal.qformer import QFormer
from viora.models.multimodal.resampler import PerceiverResampler
from viora.models.multimodal.token_injection import build_multimodal_inputs
from viora.utils.config import ProjectorConfig, ResamplerConfig


def test_resampler_fixed_output_regardless_of_input_length():
    cfg = ResamplerConfig(num_queries=8, dim=24, num_heads=3, num_layers=2)
    r = PerceiverResampler(cfg, input_dim=24)
    for m in (5, 50, 200):
        out = r(torch.randn(2, m, 24))
        assert out.shape == (2, 8, 24)


def test_resampler_projects_input_dim():
    cfg = ResamplerConfig(num_queries=4, dim=24, num_heads=3, num_layers=1)
    r = PerceiverResampler(cfg, input_dim=16)  # inputs are 16-d, queries 24-d
    assert r(torch.randn(1, 30, 16)).shape == (1, 4, 24)


def test_resampler_key_padding_mask():
    cfg = ResamplerConfig(num_queries=4, dim=16, num_heads=2, num_layers=1)
    r = PerceiverResampler(cfg, input_dim=16)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, 6:] = False
    assert r(torch.randn(1, 10, 16), key_padding_mask=mask).shape == (1, 4, 16)


def test_qformer_question_conditioned():
    cfg = ResamplerConfig(num_queries=6, dim=24, num_heads=3, num_layers=2)
    qf = QFormer(cfg, input_dim=24, text_dim=32)
    vis = torch.randn(2, 40, 24)
    text = torch.randn(2, 7, 32)
    assert qf(vis, text_embeds=text).shape == (2, 6, 24)
    # also works without text conditioning
    assert qf(vis).shape == (2, 6, 24)


def test_projector_linear_and_mlp():
    lin = MultimodalProjector(ProjectorConfig(type="linear"), input_dim=24, output_dim=48)
    assert lin(torch.randn(2, 8, 24)).shape == (2, 8, 48)
    mlp = MultimodalProjector(ProjectorConfig(type="mlp", depth=3), input_dim=24, output_dim=48)
    assert mlp(torch.randn(2, 8, 24)).shape == (2, 8, 48)


def test_injection_expands_placeholder_and_masks_labels():
    vocab, hidden, q = 50, 16, 5
    lm = DummyLanguageModel(vocab, hidden)
    video_id = vocab - 1
    # prompt: [tok, tok, <video>, tok, tok]
    input_ids = torch.tensor([[3, 4, video_id, 7, 8]])
    labels = input_ids.clone()
    visual = torch.randn(1, q, hidden)

    batch = build_multimodal_inputs(
        input_ids, visual, lm.get_input_embeddings(),
        video_token_id=video_id, labels=labels,
    )
    # length grows by (q - 1): placeholder replaced by q visual tokens
    assert batch.inputs_embeds.shape == (1, 5 - 1 + q, hidden)
    assert batch.attention_mask.shape == (1, 5 - 1 + q)
    # exactly q visual positions are ignored in the LM loss
    assert int((batch.labels == -100).sum()) == q
    # the visual embeddings were spliced at the placeholder position (index 2)
    assert torch.allclose(batch.inputs_embeds[0, 2:2 + q], visual[0])


def test_injection_prepends_when_no_placeholder():
    vocab, hidden, q = 40, 12, 4
    lm = DummyLanguageModel(vocab, hidden)
    input_ids = torch.tensor([[3, 4, 5]])
    batch = build_multimodal_inputs(
        input_ids, torch.randn(1, q, hidden), lm.get_input_embeddings(), video_token_id=99,
    )
    assert batch.inputs_embeds.shape == (1, 3 + q, hidden)


def test_injection_pads_variable_lengths():
    vocab, hidden, q = 40, 12, 3
    lm = DummyLanguageModel(vocab, hidden)
    vid = vocab - 1  # placeholder must be a valid, reserved in-vocab token id
    # one sequence has the placeholder, one does not -> different expanded lengths -> padded
    input_ids = torch.tensor([[1, vid, 2, 3], [1, 2, 3, 4]])
    attn = torch.ones(2, 4, dtype=torch.long)
    batch = build_multimodal_inputs(
        input_ids, torch.randn(2, q, hidden), lm.get_input_embeddings(),
        video_token_id=vid, attention_mask=attn,
    )
    assert batch.inputs_embeds.shape[0] == 2
    assert batch.attention_mask.shape == batch.inputs_embeds.shape[:2]


def test_dummy_lm_trains_a_step():
    lm = DummyLanguageModel(vocab_size=64, hidden_size=32, num_layers=2)
    ids = torch.randint(0, 64, (2, 12))
    out = lm(input_ids=ids, labels=ids)
    assert out.loss is not None and out.loss.item() > 0
    out.loss.backward()
    assert lm.embed_tokens.weight.grad is not None


def test_dummy_lm_forward_with_inputs_embeds():
    lm = DummyLanguageModel(vocab_size=64, hidden_size=32)
    emb = torch.randn(2, 10, 32)
    out = lm(inputs_embeds=emb)
    assert out.logits.shape == (2, 10, 64)
