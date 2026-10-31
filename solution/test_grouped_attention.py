"""Grouped packed attention must preserve independent rows and all Q/K/V grads."""
from itertools import accumulate
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('lengths', [(2, 3, 7, 13), (113, 127, 255, 512), (512,) * 4])
@pytest.mark.parametrize('frozen_key', [False, True])
@pytest.mark.parametrize('transposed_storage', [False, True])
def test_grouped_output_gradients_and_compact_saved_storage(lengths, frozen_key, transposed_storage):
    from solution.varlen_attention import try_grouped_attention
    from transformers.integrations.sdpa_attention import repeat_kv
    torch.manual_seed(7129)
    n = sum(lengths)
    def value(heads):
        if transposed_storage:
            return torch.randn(1, n, heads, 128, device='cuda', dtype=torch.float16).transpose(1, 2)
        return torch.randn(1, heads, n, 128, device='cuda', dtype=torch.float16)
    values = [value(12), value(2), value(2)]
    candidate = [t.detach().requires_grad_(i != 1 or not frozen_key) for i, t in enumerate(values)]
    reference = [t.detach().clone().requires_grad_(i != 1 or not frozen_key) for i, t in enumerate(values)]
    cumulative = torch.tensor([0, *accumulate(lengths * 2)], device='cuda', dtype=torch.int32)
    saved = []
    def observe(t):
        if t.ndim == 4 and t.shape[2] == 6 and t.stride(2) == 0:
            saved.append(t.untyped_storage().nbytes())
        return t
    rng = torch.cuda.get_rng_state()
    with torch.autograd.graph.saved_tensors_hooks(observe, lambda t: t):
        actual = try_grouped_attention(*candidate, cumulative, max(lengths), dropout=0.0, scale=128**-0.5)
    assert actual is not None
    assert len(saved) == 2 and all(size == 2*n*128*2 for size in saved), saved
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    q, k, v = reference
    k, v = repeat_kv(k, 6), repeat_kv(v, 6)
    parts, start = [], 0
    for length in lengths:
        parts.append(F.scaled_dot_product_attention(
            q[:, :, start:start+length], k[:, :, start:start+length], v[:, :, start:start+length],
            dropout_p=0.0, is_causal=True, scale=128**-0.5))
        start += length
    expected = torch.cat(parts, dim=2)
    grad = torch.randn_like(expected)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for got, want in zip(candidate, reference):
        if got.requires_grad:
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-3, atol=2e-3)
    assert try_grouped_attention(*candidate, cumulative, max(lengths), dropout=.1, scale=None) is None
