"""Numerical contracts independent of the original vocabulary implementation."""

import pytest
import torch
import torch.nn.functional as F

from solution.head import frozen_head_loss


def dense(hidden, weight, labels, denominator=None):
    logits = F.linear(hidden, weight).float()
    targets = F.pad(labels[:, 1:], (0, 1), value=-100)
    token = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(labels)
    loss = token.sum() / ((targets != -100).sum() if denominator is None else denominator)
    logp = logits.log_softmax(-1)
    return loss, token, -(logp.exp() * logp).sum(-1), logits.argmax(-1)


@pytest.mark.parametrize("chunk", [1, 4, 128])
@pytest.mark.parametrize("denominator", [None, 27])
def test_head_matches_dense_and_gradient(chunk, denominator):
    torch.manual_seed(7)
    hidden = torch.randn(3, 7, 13, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(103, 13) * .2
    labels = torch.randint(103, (3, 7))
    labels[0] = -100
    labels[1, :4] = -100
    result = frozen_head_loss(hidden, weight, labels, denominator, chunk)
    ref = dense(other, weight, labels, denominator)
    for actual, expected in zip((result.loss, result.token_loss, result.token_entropy, result.predictions), ref):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-6)
    (result.loss * 11).backward()
    (ref[0] * 11).backward()
    torch.testing.assert_close(hidden.grad, other.grad, rtol=1e-5, atol=1e-6)
    assert not result.token_loss.requires_grad
    assert not result.token_entropy.requires_grad
    assert torch.count_nonzero(result.token_loss[0]) == 0
    assert torch.all(result.token_entropy[0] > 0)


def test_no_saved_logits_and_external_scale():
    torch.manual_seed(11)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(103, 7)
    labels = torch.randint(103, (2, 5))
    saved = []
    def pack(t):
        saved.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = frozen_head_loss(hidden, weight, labels, chunk_size=3)
    assert (2, 5, 103) not in saved and (10, 103) not in saved
    assert all(103 not in shape or shape == (103, 7) for shape in saved)
    result.loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_all_ignored_and_explicit_denominator():
    h = torch.randn(2, 4, 5, requires_grad=True)
    w = torch.randn(17, 5)
    labels = torch.full((2, 4), -100)
    out = frozen_head_loss(h, w, labels, 8)
    assert out.loss.item() == 0
    out.loss.backward()
    assert torch.count_nonzero(h.grad) == 0
    assert torch.isnan(frozen_head_loss(h, w, labels).loss)


def test_trainable_head_rejected():
    with pytest.raises(ValueError, match="frozen"):
        frozen_head_loss(torch.randn(1, 2, 3), torch.randn(7, 3, requires_grad=True), torch.ones(1, 2, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; CPU is not GPU validation")
@pytest.mark.parametrize("scale", [1., 65536.])
def test_cuda_amp_backward_scale(scale):
    torch.manual_seed(6)
    hidden = torch.randn(2, 9, 64, device="cuda", requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(4099, 64, device="cuda") * .05
    labels = torch.randint(4099, (2, 9), device="cuda")
    labels[0, :4] = -100
    with torch.autocast("cuda", dtype=torch.float16):
        out = frozen_head_loss(hidden, weight, labels, chunk_size=8)
        expected = dense(other, weight, labels)
    for actual, ref in zip((out.loss, out.token_loss, out.token_entropy), expected[:3]):
        torch.testing.assert_close(actual, ref, atol=2e-3, rtol=1e-3)
    (out.loss * scale).backward()
    (expected[0] * scale).backward()
    torch.testing.assert_close(hidden.grad / scale, other.grad / scale, atol=2e-6, rtol=.01)
