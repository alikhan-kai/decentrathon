"""Recomputation stores only values actually consumed by the frozen projections."""
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('rows', [941, 1427])
@pytest.mark.parametrize('grad_scale', [1.0, 128.0])
def test_half_boundary_save_matches_native_path_exactly(rows, grad_scale):
    import bitsandbytes as bnb
    from solution import nf4, swiglu
    from solution.frozen_remat import frozen_branch_remat
    from solution.native_half import _NativeHalfSwiGLU
    torch.manual_seed(2049)
    nf4.patch()
    module = SimpleNamespace(intermediate_size=8960)
    for name in ('gate_proj', 'up_proj'):
        projection = bnb.nn.Linear4bit(1536, 8960, bias=False,
                                      compute_dtype=torch.float16, quant_type='nf4',
                                      compress_statistics=True)
        projection.requires_grad_(False).cuda().train()
        projection.compute_type_is_set = True
        setattr(module, name, projection)
    x = torch.randn(1, rows, 1536, device='cuda', requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    grad = torch.randn(1, rows, 8960, device='cuda', dtype=torch.float16) * grad_scale
    saved = []

    def observe(t):
        saved.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        return t

    rng = torch.cuda.get_rng_state()
    with torch.autocast('cuda', dtype=torch.float16):
        with torch.autograd.graph.saved_tensors_hooks(observe, lambda t: t):
            actual = frozen_branch_remat(module, x, swiglu)
        assert torch.equal(torch.cuda.get_rng_state(), rng)
        gate = module.gate_proj(reference.to(torch.float16))
        up = module.up_proj(reference.to(torch.float16))
        expected = _NativeHalfSwiGLU.apply(gate, up, swiglu)
    assert saved == [(tuple(x.shape), torch.float16, x.numel() * 2)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward(grad)
    rng_before_backward = torch.cuda.get_rng_state()
    actual.backward(grad)
    assert torch.equal(torch.cuda.get_rng_state(), rng_before_backward)
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
