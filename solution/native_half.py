"""Private experiment, ONLY after canonical_guards.MeasurementPatch.reason_for.

Same lossless-save hypothesis as packed saves; no original kernel credit.
No installer, registry entry, class patch, or broad public activation API.
"""
import hashlib
import inspect

import torch

ENABLED = False
BNB_HASHES = {
    "Linear4bit.forward": "d52cf63457717cc718ca0a449dad0967171e679be90d5d2adcafe2e8a141093d",
    "MatMul4Bit": "243126867f134209a5c94687f17b90bf20b1b9e5391d6299810b8053f6df070b",
    "matmul_4bit": "a219d40d62264de60a424e8f9f1bab7656136eb7264a575d39d57a7e2f25ed4a",
}


def _validate_sources(starter, canonical_guards):
    """One-time provenance check in addition to, never instead of, guards."""
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit, matmul_4bit
    assert bnb.__version__ == "0.50.2"
    assert canonical_guards._verified_starter(starter)
    objects = (bnb.nn.Linear4bit.forward, MatMul4Bit, matmul_4bit)
    for name, obj in zip(BNB_HASHES, objects):
        assert hashlib.sha256(inspect.getsource(obj).encode()).hexdigest() == BNB_HASHES[name], name


class _NativeHalfSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, starter):
        assert type(gate) is torch.Tensor and type(up) is torch.Tensor
        assert gate.dtype == up.dtype == torch.float16
        assert gate.shape == up.shape and gate.device == up.device
        assert gate.is_contiguous() and up.is_contiguous()
        ctx.starter = starter
        # Calling starter.forward would allocate empty_like(gate) in FP16.
        out = torch.empty(gate.shape, device=gate.device, dtype=torch.float32)
        n = gate.numel()
        starter.example_swiglu_fwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            gate, up, out, n, BLOCK=starter.BLOCK)
        ctx.save_for_backward(gate, up)  # no cast, clone, or expanded save
        return out

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("native-half save is first-order only")
        gate, up = ctx.saved_tensors
        assert grad.dtype == torch.float32
        grad = grad.contiguous()
        dgate = torch.empty(gate.shape, device=gate.device, dtype=torch.float32)
        dup = torch.empty(up.shape, device=up.device, dtype=torch.float32)
        starter = ctx.starter
        n = gate.numel()
        starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=starter.BLOCK)
        # Autograd converts each returned FP32 gradient to its FP16 input dtype.
        # This replaces the baseline output-expansion ToCopyBackward boundary.
        return dgate, dup, None


def _validated_entry(mlp, x, starter):
    """Caller MUST have just passed the EXISTING canonical runtime guards.

    Also requires _validate_sources once and first-order-only caller contract.
    Never hoist a shared x.to(float16) out of these TWO projection calls.
    """
    gate = mlp.gate_proj(x.to(torch.float16))
    up = mlp.up_proj(x.to(torch.float16))
    if not gate.is_contiguous() or not up.is_contiguous():
        # Preserve starter layout behavior without replaying either projection.
        out = starter.swiglu(gate.float(), up.float())
    else:
        out = _NativeHalfSwiGLU.apply(gate, up, starter)
    del gate, up
    assert out.dtype == torch.float32
    return mlp.down_proj(out)
