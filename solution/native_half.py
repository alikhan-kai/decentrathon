"""Private experiment, ONLY after canonical_guards.SavedActivationPatch.reason_for.

Same lossless-save hypothesis as packed saves; no original kernel credit.
No installer, registry entry, class patch, or broad public activation API.
"""
import torch

ENABLED = False
PINNED_BNB = "0.50.2"


def _validate_versions():
    """Reason string if the installed bitsandbytes is not the pinned one, else None. The decision
    uses package versions only; no library source is read."""
    import bitsandbytes as bnb
    if bnb.__version__ != PINNED_BNB:
        return "unpinned-bitsandbytes-version"
    return None


class _NativeHalfSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, starter):
        assert type(gate) is torch.Tensor and type(up) is torch.Tensor
        assert gate.dtype == up.dtype == torch.float16
        assert gate.shape == up.shape and gate.device == up.device
        assert gate.is_contiguous() and up.is_contiguous()
        ctx.starter = starter
        # The downstream canonical BNB input boundary is FP16 already.
        out = torch.empty(gate.shape, device=gate.device, dtype=torch.float16)
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
        assert grad.dtype == torch.float16
        grad = grad.contiguous()
        # Unchanged starter arithmetic; each output rounds independently at its
        # original FP16 projection boundary. Pointer specializations get no credit.
        dgate = torch.empty(gate.shape, device=gate.device, dtype=torch.float16)
        dup = torch.empty(up.shape, device=up.device, dtype=torch.float16)
        starter = ctx.starter
        n = gate.numel()
        starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=starter.BLOCK)
        # Return two distinct half cotangents. Each projection's independent
        # input cast still expands to FP32 before residual accumulation.
        return dgate, dup, None


def _validated_entry(mlp, x, starter):
    """Caller MUST have just passed the EXISTING canonical runtime guards.

    Also requires _validate_versions once and first-order-only caller contract.
    Never hoist a shared x.to(float16) out of these TWO projection calls.
    """
    gate = mlp.gate_proj(x.to(torch.float16))
    up = mlp.up_proj(x.to(torch.float16))
    if not gate.is_contiguous() or not up.is_contiguous():
        # Preserve starter layout behavior without replaying either projection.
        out = starter.swiglu(gate.float(), up.float())
        del gate, up
        return mlp.down_proj(out)
    else:
        out = _NativeHalfSwiGLU.apply(gate, up, starter)
    del gate, up
    assert out.dtype == torch.float16
    # Move only BNB's output widening outside the canonical frozen down call.
    return mlp.down_proj(out).to(torch.float32)
