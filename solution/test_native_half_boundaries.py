"""Exact comparisons against the expanded FP32 SwiGLU boundaries on real NF4."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("rows", [1, 37, 257])
@pytest.mark.parametrize("scaled", [False, True])
def test_frozen_mlp_outputs_and_input_gradients_are_exact(rows, scaled):
    import bitsandbytes as bnb
    from solution import native_half, swiglu

    torch.manual_seed(819)

    def linear(i, o):
        mod = bnb.nn.Linear4bit(i, o, bias=False, compute_dtype=torch.float16,
                               compress_statistics=True, quant_type="nf4")
        return mod.requires_grad_(False).cuda()

    mlp = SimpleNamespace(gate_proj=linear(1536, 8960), up_proj=linear(1536, 8960),
                          down_proj=linear(8960, 1536))
    x = torch.randn(1, rows, 1536, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        gate = mlp.gate_proj(x_ref.to(torch.float16)).float()
        up = mlp.up_proj(x_ref.to(torch.float16)).float()
        reference = mlp.down_proj(swiglu.swiglu(gate, up))
        actual = native_half._validated_entry(mlp, x, swiglu)
    grad = torch.randn_like(reference) * (128 if scaled else 1)
    reference.backward(grad)
    actual.backward(grad)
    assert actual.dtype == reference.dtype == torch.float32
    assert torch.equal(actual, reference)
    assert torch.equal(x.grad, x_ref.grad)
    assert torch.isfinite(x.grad).all()
