"""Compare a kernel with plain PyTorch, forward and backward. No kernels here.

    from solution.kcheck import check_fwd_bwd
    check_fwd_bwd(solution.swiglu.swiglu, lambda a, b: torch.nn.functional.silu(a) * b,
                  shapes=[(4, 512, 8960), (4, 512, 8960)], dtype=torch.float16)

The reference runs in fp32 on the same (upcast) inputs; the kernel runs in `dtype`. Raises
AssertionError with the tensor name on a mismatch; returns the max abs errors for printing.
Tests that use it must be skipped without CUDA:
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
"""

import torch

TOLERANCE = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-4}


def check_fwd_bwd(kernel_fn, ref_fn, shapes, dtype=torch.float16, seed=0, tol=None, device="cuda"):
    tol = TOLERANCE[dtype] if tol is None else tol
    torch.manual_seed(seed)
    inputs = [torch.randn(s, device=device, dtype=dtype, requires_grad=True) for s in shapes]
    ref_inputs = [x.detach().float().requires_grad_(True) for x in inputs]
    ref = ref_fn(*ref_inputs)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    out = kernel_fn(*inputs)
    out.backward(grad.to(out.dtype))
    errors = {"out": (out.float() - ref).abs().max().item()}
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol, msg=lambda m: f"out: {m}")
    for i, (x, r) in enumerate(zip(inputs, ref_inputs)):
        errors[f"grad{i}"] = (x.grad.float() - r.grad).abs().max().item()
        torch.testing.assert_close(x.grad.float(), r.grad, rtol=tol, atol=tol,
                                   msg=lambda m, i=i: f"grad{i}: {m}")
    return errors
