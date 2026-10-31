"""Frozen-key packed-row attention parity; T4 only."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('n', [113, 512])
def test_frozen_key_output_and_live_gradients(n):
    from solution.attention_gqa import try_t4_packed_attention
    torch.manual_seed(91)
    module = SimpleNamespace(training=True, num_key_value_groups=6, is_causal=True,
                             sliding_window=None, config=SimpleNamespace(output_attentions=False))
    q = torch.randn(1, 12, n, 128, device='cuda', requires_grad=True)
    k = torch.randn(1, 2, n, 128, device='cuda')
    v = torch.randn(1, 2, n, 128, device='cuda', requires_grad=True)
    dq, dv = q.detach().requires_grad_(), v.detach().requires_grad_()
    grad = torch.randn(1, 12, n, 128, device='cuda', dtype=torch.float16)
    with torch.autocast('cuda', dtype=torch.float16):
        reference = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                   is_causal=True, enable_gqa=True)
        result = try_t4_packed_attention(module, dq, k, dv, enabled=True, full_prefill=True)
        assert result is not None
        candidate = result[0].transpose(1, 2)
        rg = torch.autograd.grad(reference, (q, v), grad)
        cg = torch.autograd.grad(candidate, (dq, dv), grad)
    torch.testing.assert_close(candidate, reference, rtol=2e-3, atol=2e-3)
    for got, expected in zip(cg, rg):
        torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)
    assert try_t4_packed_attention(module, dq, k, dv, enabled=True,
                                   full_prefill=True, use_cache=True) is None
