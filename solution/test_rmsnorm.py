"""Unit tests for solution/rmsnorm.py against the original transformers Qwen2RMSNorm (5.16.1).

Tolerances (explicit): the training case is fp32 input and fp32 weight; forward and backward must match
the original within rtol = atol = 1e-4 (the two differ only in the order of the fp32 reduction and the
rsqrt implementation). The fp16-input case is not handled by the kernel and must return exactly what
the original returns (torch.equal). Skipped without CUDA.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOL = 1e-4


@pytest.fixture(autouse=True)
def unpatched():
    from solution import rmsnorm

    rmsnorm.unpatch()
    yield rmsnorm
    rmsnorm.unpatch()


def make_norm(hidden=1536, requires_grad=False):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(0)
    norm = Qwen2RMSNorm(hidden, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden) * 0.5 + 1.0)
    norm.weight.requires_grad_(requires_grad)
    return norm


@pytest.mark.parametrize("shape", [(4, 512, 1536), (4, 37, 1536), (3, 5, 1001)])
@pytest.mark.parametrize("weight_grad", [False, True])
def test_matches_original_forward_and_backward(shape, weight_grad, unpatched):
    norm = make_norm(shape[-1], requires_grad=weight_grad)
    torch.manual_seed(1)
    x = (torch.randn(*shape, device="cuda") * 3).requires_grad_(True)
    grad = torch.randn(*shape, device="cuda")

    ref = norm(x)
    ref.backward(grad)
    ref_dx, ref_dw = x.grad.clone(), (norm.weight.grad.clone() if weight_grad else None)
    x.grad = norm.weight.grad = None

    out = unpatched.rmsnorm(x, norm.weight, norm.variance_epsilon)
    out.backward(grad)
    assert out.dtype == ref.dtype == torch.float32 and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=TOL, atol=TOL)
    torch.testing.assert_close(x.grad, ref_dx, rtol=TOL, atol=TOL)
    if weight_grad:
        torch.testing.assert_close(norm.weight.grad, ref_dw, rtol=1e-3, atol=1e-3)
    else:
        assert norm.weight.grad is None


def test_patched_module_and_fallbacks(unpatched):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    norm = make_norm()
    x16 = torch.randn(2, 16, 1536, device="cuda", dtype=torch.float16)
    ref16 = norm(x16)
    x32 = torch.randn(2, 16, 1536, device="cuda")
    ref32 = norm(x32)

    unpatched.patch()
    unpatched.patch()  # idempotent
    assert Qwen2RMSNorm.forward is not unpatched._original
    assert torch.equal(norm(x16), ref16)                       # fp16 input: original path, identical
    torch.testing.assert_close(norm(x32), ref32, rtol=TOL, atol=TOL)  # fp32 input: kernel
    unpatched.unpatch()
    assert Qwen2RMSNorm.forward is unpatched._original
