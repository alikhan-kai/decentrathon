"""Exact storage, computation and fallback checks for tied frozen weights."""
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from solution import frozen_weight as fw


@pytest.fixture(autouse=True)
def restore_forwards(monkeypatch):
    monkeypatch.setattr(torch.nn.Embedding, "forward", torch.nn.Embedding.forward)
    monkeypatch.setattr(torch.nn.Linear, "forward", torch.nn.Linear.forward)
    monkeypatch.setattr(fw, "_patched", False)


def tiny():
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, tie_word_embeddings=True,
                        attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).cuda().float()
    model.lm_head.weight.requires_grad_(False)
    with torch.no_grad():
        model.lm_head.weight.copy_(model.lm_head.weight.half().float())
    return model


@pytest.mark.parametrize("bad", [0.10000001, float("nan"), float("inf"), 65536.0])
def test_reject_lossy_or_nonfinite(bad):
    assert fw._half_if_exact(torch.tensor([bad], dtype=torch.float32)) is None


def test_exact_signed_zero_and_values():
    original = torch.tensor([0.0, -0.0, 0.5, -2.0, 65504.0], dtype=torch.float32)
    packed = fw._half_if_exact(original)
    assert packed.dtype == torch.float16
    assert torch.equal(original.view(torch.int32), packed.float().view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("autocast", [False, True])
def test_model_values_gradients_tie_and_fallback(autocast):
    torch.manual_seed(17)
    reference = tiny()
    candidate = copy.deepcopy(reference)
    parameter = candidate.lm_head.weight
    assert fw.compact_tied_weight(candidate)
    assert candidate.lm_head.weight is parameter is candidate.model.embed_tokens.weight
    assert parameter.dtype == torch.float16 and not parameter.requires_grad
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    assert candidate.model.embed_tokens(ids).dtype == torch.float32
    grads = []
    outputs = []
    for model in (reference, candidate):
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            output = model(input_ids=ids, labels=ids)
        output.loss.backward()
        outputs.append((output.logits.detach(), output.loss.detach()))
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    for expected, actual in zip(outputs[0], outputs[1]):
        assert torch.equal(expected, actual)
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert torch.equal(grads[0][name], grads[1][name]), name
    parameter.requires_grad_(True)
    candidate.model.embed_tokens(ids)
    assert parameter.dtype == torch.float32
    assert candidate.lm_head.weight is candidate.model.embed_tokens.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_guards():
    model = tiny()
    model.lm_head.weight.requires_grad_(True)
    assert not fw.compact_tied_weight(model)
    model.lm_head.weight.requires_grad_(False)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone(), requires_grad=False)
    assert not fw.compact_tied_weight(model)
    model = tiny().cpu()
    assert not fw.compact_tied_weight(model)
