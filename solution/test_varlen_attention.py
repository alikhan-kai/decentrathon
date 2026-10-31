"""Compare the packed native operator with independent causal SDPA calls."""
from itertools import accumulate
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('lengths', [(2, 3, 7, 13), (113, 127, 255, 512), (512,) * 4])
@pytest.mark.parametrize('frozen_key', [False, True])
@pytest.mark.parametrize('strided_grad', [False, True])
def test_independent_rows_and_gradients(lengths, frozen_key, strided_grad):
    from solution.varlen_attention import try_packed_attention
    torch.manual_seed(1027)
    n = sum(lengths)
    values = [torch.randn(1, 12, n, 128, device='cuda', dtype=torch.float16)
              for _ in range(3)]
    candidate = [t.detach().requires_grad_(i != 1 or not frozen_key) for i, t in enumerate(values)]
    reference = [t.detach().clone().requires_grad_(i != 1 or not frozen_key) for i, t in enumerate(values)]
    cumulative = torch.tensor([0, *accumulate(lengths)], device='cuda', dtype=torch.int32)
    rng = torch.cuda.get_rng_state()
    actual = try_packed_attention(*candidate, cumulative, max(lengths), dropout=0.0, scale=128**-0.5)
    assert actual is not None
    assert torch.equal(rng, torch.cuda.get_rng_state())
    parts, start = [], 0
    for length in lengths:
        parts.append(F.scaled_dot_product_attention(
            *(t[:, :, start:start+length] for t in reference), is_causal=True,
            dropout_p=0.0, scale=128**-0.5))
        start += length
    expected = torch.cat(parts, dim=2)
    grad = torch.randn(1, 12, n, 256 if strided_grad else 128,
                       device='cuda', dtype=torch.float16)
    if strided_grad:
        grad = grad[..., ::2]
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for got, want in zip(candidate, reference):
        if got.requires_grad:
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-3, atol=2e-3)
    assert try_packed_attention(*candidate, cumulative, max(lengths), dropout=.1, scale=None) is None
