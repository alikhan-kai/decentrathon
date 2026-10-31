from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def quantized(shape):
    import bitsandbytes.functional as bf
    torch.manual_seed(731)
    return bf.quantize_4bit(torch.randn(*shape, device='cuda', dtype=torch.float16) * 0.02,
                            blocksize=64, compress_statistics=True, quant_type='nf4')


@pytest.mark.parametrize('shape', [(8960, 1536), (1536, 8960)])
@pytest.mark.parametrize('rows', [64, 257])
def test_gradient_approximation_and_forward_unchanged(shape, rows):
    import bitsandbytes as bnb
    import bitsandbytes.functional as bf
    from solution import nf4_dgrad_int8 as candidate
    from bitsandbytes.autograd._functions import MatMul4Bit

    packed, state = quantized(shape)
    grad = torch.randn(1, rows, shape[0], device='cuda', dtype=torch.float16)
    reference = grad @ bf.dequantize_4bit(packed, state)
    actual = candidate.input_gradient(grad, packed, state)
    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.025, relative_l2
    assert actual.dtype == reference.dtype and actual.shape == reference.shape
    original = MatMul4Bit.backward
    before_calls = candidate._calls
    try:
        x = torch.randn(1, rows, shape[1], device='cuda', dtype=torch.float16, requires_grad=True)
        ref_out = bnb.matmul_4bit(x, packed.t(), quant_state=state)
        candidate.patch()
        out = bnb.matmul_4bit(x, packed.t(), quant_state=state)
        assert torch.equal(out, ref_out)
        out.backward(grad)
        assert candidate._calls == before_calls + 1
        assert torch.equal(x.grad, actual)
    finally:
        MatMul4Bit.backward = staticmethod(original)
        candidate._patched = False


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_gradient_remains_nonfinite_for_amp(bad):
    from solution.nf4_dgrad_int8 import input_gradient
    packed, state = quantized((1536, 8960))
    grad = torch.zeros(64, 1536, device='cuda', dtype=torch.float16)
    grad[0, 0] = bad
    actual = input_gradient(grad, packed, state)
    assert not torch.isfinite(actual[0]).all()
    assert torch.equal(actual[1:], torch.zeros_like(actual[1:]))


def test_cpu_dispatch_falls_back():
    from solution.nf4_dgrad_int8 import eligible
    with torch.no_grad():
        assert not eligible(SimpleNamespace(is_empty=False), torch.ones(64, 1536))
