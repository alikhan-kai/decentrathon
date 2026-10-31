import contextlib
import pytest
import torch
import triton
from candidate import residual_norm as rn
from test_residual_norm import interpreted, patched, make_layer, layer_arguments, qwen_reference


def test_decoder_validates_once_after_attention(interpreted, patched, monkeypatch):
    layer = make_layer()
    calls = []
    original = rn._supports_triton
    def validate(*args):
        calls.append('validation')
        return original(*args)
    monkeypatch.setattr(rn, '_supports_triton', validate)
    monkeypatch.setattr(rn, 'fused_residual_norm', lambda *args: pytest.fail('repeated public validation'))
    out = layer(torch.randn(1, 3, 16, requires_grad=True), **layer_arguments())
    out.sum().backward()
    assert calls == ['validation']


def test_attention_installs_norm_hook_after_initial_guard(interpreted, patched, monkeypatch):
    layer = make_layer()
    events, handles = [], []
    def attention_hook(module, args, out):
        events.append('attention')
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            lambda *args: events.append('norm')))
    handles.append(layer.self_attn.register_forward_hook(attention_hook))
    monkeypatch.setattr(rn, '_fused_residual_norm_validated', lambda *args: pytest.fail('new hook bypassed'))
    try:
        layer(torch.randn(1, 3, 16), **layer_arguments())
        assert events == ['attention', 'norm']
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('hidden', [17, 1536])
def test_offset_contiguous_inputs_and_gradients(interpreted, hidden):
    # Contiguity does not imply 16-byte alignment; storage offset one is legal.
    r = torch.randn(2 * hidden + 1)[1:].reshape(2, hidden).requires_grad_()
    a = torch.randn(2 * hidden + 1, dtype=torch.float16)[1:].reshape(2, hidden).requires_grad_()
    w = torch.rand(hidden + 1)[1:]
    assert r.is_contiguous() and r.data_ptr() % 16 != 0
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    expected = rn._eager(r, a, w, 1e-6)
    gs = torch.randn(2, hidden * 2)[:, ::2]
    gy = torch.randn(1).expand_as(r)
    got = torch.autograd.grad(actual, (r, a), (gs, gy))
    want = torch.autograd.grad(expected, (r, a), (gs, gy))
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
    for x, y in zip(got, want):
        tol = 2e-3 if x.dtype == torch.float16 else 1e-4
        torch.testing.assert_close(x, y, atol=tol, rtol=tol)


def test_default_disabled():
    assert rn.ENABLED is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires compiled CUDA')
@pytest.mark.parametrize('hidden', [17, 1536])
def test_cuda_offset_contiguous_inputs_and_gradients(hidden):
    # Keep the offset views themselves: cloning would remove the condition tested.
    r = torch.randn(2 * hidden + 1, device='cuda')[1:].reshape(2, hidden).requires_grad_()
    a = torch.randn(2 * hidden + 1, device='cuda', dtype=torch.float16)[1:].reshape(2, hidden).requires_grad_()
    w = torch.rand(hidden + 1, device='cuda')[1:]
    assert r.is_contiguous() and r.data_ptr() % 16 != 0
    assert a.data_ptr() % 16 != 0 and w.data_ptr() % 16 != 0
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    assert type(actual[0].grad_fn).__name__ == '_ResidualNormBackward'
    expected = rn._eager(r, a, w, 1e-6)
    gs = torch.randn(2, hidden * 2, device='cuda')[:, ::2]
    gy = torch.randn(1, device='cuda').expand_as(r)
    got = torch.autograd.grad(actual, (r, a), (gs, gy))
    want = torch.autograd.grad(expected, (r, a), (gs, gy))
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
    for x, y in zip(got, want):
        tol = 2e-3 if x.dtype == torch.float16 else 1e-4
        torch.testing.assert_close(x, y, atol=tol, rtol=tol)
