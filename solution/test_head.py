"""Numerical contracts independent of the original vocabulary implementation."""

import gc
import inspect
import weakref

import pytest
import torch
import torch.nn.functional as F

from solution.head import frozen_head_loss


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA; CPU is not GPU validation"))]


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


@pytest.mark.parametrize("cache_logits", [False, True])
def test_all_ignored_and_explicit_denominator(cache_logits):
    h = torch.randn(2, 4, 5, requires_grad=True)
    w = torch.randn(17, 5)
    labels = torch.full((2, 4), -100)
    out = frozen_head_loss(h, w, labels, 8, cache_logits=cache_logits)
    assert out.loss.item() == 0
    out.loss.backward()
    assert torch.count_nonzero(h.grad) == 0
    assert torch.isnan(frozen_head_loss(h, w, labels, cache_logits=cache_logits).loss)


def test_trainable_head_rejected():
    with pytest.raises(ValueError, match="frozen"):
        frozen_head_loss(torch.randn(1, 2, 3), torch.randn(7, 3, requires_grad=True), torch.ones(1, 2, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; CPU is not GPU validation")
@pytest.mark.parametrize("scale", [1., 65536.])
@pytest.mark.parametrize("denominator", [None, 37])
@pytest.mark.parametrize("cache_logits", [False, True])
def test_cuda_amp_backward_scale(scale, denominator, cache_logits):
    torch.manual_seed(6)
    hidden = torch.randn(2, 9, 64, device="cuda", requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(4099, 64, device="cuda") * .05
    labels = torch.randint(4099, (2, 9), device="cuda")
    labels[0] = -100
    labels[1, :4] = -100
    with torch.autocast("cuda", dtype=torch.float16):
        out = frozen_head_loss(hidden, weight, labels, denominator, chunk_size=8,
                               cache_logits=cache_logits)
        expected = dense(other, weight, labels, denominator)
    for actual, ref in zip((out.loss, out.token_loss, out.token_entropy), expected[:3]):
        torch.testing.assert_close(actual, ref, atol=2e-3, rtol=1e-3)
    (out.loss * scale).backward()
    (expected[0] * scale).backward()
    torch.testing.assert_close(hidden.grad / scale, other.grad / scale, atol=2e-6, rtol=.01)


def test_cache_logits_is_opt_in_and_keyword_only():
    option = inspect.signature(frozen_head_loss).parameters["cache_logits"]
    assert option.default is False
    assert option.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("chunk", [1, 4, 128])
@pytest.mark.parametrize("denominator", [None, 27])
@pytest.mark.parametrize("device", DEVICES)
def test_cached_head_matches_default_and_dense_strided(dtype, chunk, denominator, device):
    torch.manual_seed(19)
    source = torch.randn(3, 7, 26, dtype=dtype, device=device)[..., ::2]
    hidden = source.detach().requires_grad_()
    recomputed = source.detach().requires_grad_()
    other = source.detach().requires_grad_()
    weight = (torch.randn(13, 103, dtype=dtype, device=device) * .2).t()
    labels = torch.randint(103, (7, 3), device=device).t()
    labels[0] = -100
    labels[1, :4] = -100
    assert not hidden.is_contiguous() and not weight.is_contiguous() and not labels.is_contiguous()
    cached = frozen_head_loss(hidden, weight, labels, denominator, chunk, cache_logits=True)
    default = frozen_head_loss(recomputed, weight, labels, denominator, chunk)
    reference = dense(other, weight, labels, denominator)
    def outputs(result):
        return result.loss, result.token_loss, result.token_entropy, result.predictions

    for actual, original, expected in zip(outputs(cached), outputs(default), reference):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-3 if dtype == torch.float16 else 3e-6,
                                   atol=2e-3 if dtype == torch.float16 else 2e-6)
    (cached.loss * 11).backward()
    (default.loss * 11).backward()
    (reference[0] * 11).backward()
    torch.testing.assert_close(hidden.grad, recomputed.grad, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, other.grad, rtol=2e-3 if dtype == torch.float16 else 1e-5,
                               atol=2e-4 if dtype == torch.float16 else 1e-6)
    assert hidden.grad.dtype == dtype
    assert all(not value.requires_grad for value in outputs(cached)[1:])
    assert torch.count_nonzero(cached.token_loss[0]) == 0
    assert torch.count_nonzero(hidden.grad[0]) == 0
    assert torch.all(cached.token_entropy[0] > 0)
    assert torch.all(cached.token_entropy[:, -1] > 0)


@pytest.mark.parametrize("dtype,amp", [(torch.float32, False), (torch.float16, False),
                                      (torch.float32, True)])
@pytest.mark.parametrize("cache_logits", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_saved_logits_match_bounded_forward_projection(monkeypatch, dtype, amp, cache_logits, device):
    torch.manual_seed(23)
    hidden = torch.randn(2, 5, 7, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(103, 7, dtype=dtype, device=device)
    labels = torch.randint(103, (2, 5), device=device)
    projections, saved, caches = [], [], []
    linear = F.linear

    def project(*args, **kwargs):
        logits = linear(*args, **kwargs)
        projections.append(logits.detach().clone())
        return logits

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        if tensor.shape == (10, 103):
            caches.append(tensor.detach().clone())
        return tensor

    monkeypatch.setattr(F, "linear", project)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            result = frozen_head_loss(hidden, weight, labels, chunk_size=3,
                                      cache_logits=cache_logits)
    assert [chunk.shape[0] for chunk in projections] == [3, 3, 3, 1]
    assert len(caches) == int(cache_logits)
    assert all(103 not in shape or shape in ((103, 7), (10, 103)) for shape, _ in saved)
    assert [saved_dtype for shape, saved_dtype in saved if shape == (103, 7)] == [dtype]
    if cache_logits:
        assert caches[0].dtype == (torch.float16 if amp else dtype)
        assert caches[0].is_contiguous()
        torch.testing.assert_close(caches[0], torch.cat(projections), rtol=0, atol=0)
    projections.clear()
    result.loss.backward()
    assert [chunk.shape[0] for chunk in projections] == ([] if cache_logits else [3, 3, 3, 1])


@pytest.mark.parametrize("finish", ["backward", "abandon", "retain_graph"])
@pytest.mark.parametrize("device", DEVICES)
def test_cached_logits_autograd_lifetime(finish, device):
    torch.manual_seed(29)
    hidden = torch.randn(2, 5, 7, device=device, requires_grad=True)
    weight = torch.randn(103, 7, device=device) * .2
    labels = torch.randint(103, (2, 5), device=device)
    refs = []

    def pack(tensor):
        if tensor.shape == (10, 103):
            refs.append(weakref.ref(tensor))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = frozen_head_loss(hidden, weight, labels, chunk_size=3, cache_logits=True)
    assert len(refs) == 1 and refs[0]() is not None
    if finish == "abandon":
        del result
    elif finish == "backward":
        result.loss.backward()
    else:
        snapshot = refs[0]().clone()
        for scale, retain in [(3., True), (11., False)]:
            other = hidden.detach().clone().requires_grad_()
            expected = dense(other, weight, labels)[0]
            actual_grad, = torch.autograd.grad(result.loss * scale, hidden, retain_graph=retain)
            expected_grad, = torch.autograd.grad(expected * scale, other)
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
            if retain:
                assert refs[0]() is not None
                torch.testing.assert_close(refs[0](), snapshot, rtol=0, atol=0)
    gc.collect()
    assert refs[0]() is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [1., 65536.])
@pytest.mark.parametrize("cache_logits", [False, True])
def test_half_gradient_scale_precedes_narrowing(device, scale, cache_logits):
    # The non-target contribution underflows fp16 unless scaled before casting.
    hidden = torch.zeros(1, 2, 1, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.zeros(257, 1, device=device, dtype=torch.float16)
    weight[1, 0] = 1
    labels = torch.zeros(1, 2, device=device, dtype=torch.long)
    out = frozen_head_loss(hidden, weight, labels, 2**20, chunk_size=1,
                           cache_logits=cache_logits)
    (out.loss * scale).backward()
    expected = torch.zeros_like(hidden)
    expected[0, 0, 0] = scale / (257 * 2**20)
    torch.testing.assert_close(hidden.grad, expected, rtol=0, atol=0)
    assert bool(hidden.grad[0, 0, 0] > 0) == (scale == 65536.)


def test_cached_graphs_are_independent():
    torch.manual_seed(31)
    graphs = []
    for vocab, chunk in [(103, 3), (107, 4)]:
        hidden = torch.randn(2, 5, 7, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        weight = torch.randn(vocab, 7) * .2
        labels = torch.randint(vocab, (2, 5))
        result = frozen_head_loss(hidden, weight, labels, 17, chunk, cache_logits=True)
        graphs.append((hidden, other, result.loss, dense(other, weight, labels, 17)[0]))
    for hidden, other, actual, expected in reversed(graphs):
        (actual * 11).backward()
        (expected * 11).backward()
        torch.testing.assert_close(hidden.grad, other.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("cache_logits", [False, True])
def test_head_remains_first_order_only(cache_logits):
    hidden = torch.randn(1, 3, 7, requires_grad=True)
    result = frozen_head_loss(hidden, torch.randn(103, 7), torch.zeros(1, 3, dtype=torch.long),
                              cache_logits=cache_logits)
    gradient, = torch.autograd.grad(result.loss, hidden, create_graph=True)
    assert not gradient.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(gradient.sum(), hidden)
