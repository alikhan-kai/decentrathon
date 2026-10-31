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
        # The next frozen projection immediately rounds its input to FP16.
        # Store that same rounding here instead of expanding then recasting.
        out = torch.empty_like(gate)
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
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        starter = ctx.starter
        n = gate.numel()
        starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=starter.BLOCK)
        # The kernel promotes arithmetic to FP32, then stores the same rounding
        # that autograd previously performed at each FP16 projection boundary.
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
    else:
        out = _NativeHalfSwiGLU.apply(gate, up, starter)
    del gate, up
    return mlp.down_proj(out).to(x.dtype)
