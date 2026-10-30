"""Private post-provenance MLP helper. No installer, patch, or registry edits.

Caller MUST first pass the unchanged canonical direct-pack R2 provenance and
input guards and declare first-order eager training. This function cannot prove
that arbitrary FP32 projections are losslessly representable as FP16. CPU and
odd-N CUDA cases in this directory are explicitly private diagnostic probes,
not additions to production eligibility.

The two private classes below are verbatim R2. The unchanged starter kernels
keep their names and remain excluded from original-kernel credit. No residual
cast is introduced. The fixed row threshold is a working-set policy, never a
step/seed/dataset lookup or a value chosen from measurements at runtime.
"""
import torch
from torch.utils.checkpoint import checkpoint


class _PackSaves:
    def __init__(self, ctx):
        self.ctx = ctx

    def save_for_backward(self, gate, up):
        # Private, reached only after provenance checks in installed forward.
        # No tensors are attached to ctx except these two packed saves.
        assert gate.dtype == up.dtype == torch.float32
        self.ctx.save_for_backward(gate.to(torch.float16), up.to(torch.float16))


class _PackedSwiGLU(torch.autograd.Function):
    """Private primitive; NOT an API for packing arbitrary fp32 inputs."""

    @staticmethod
    def forward(ctx, gate, up, starter):
        ctx.starter = starter
        # Execute exactly the existing forward, intercepting only its saves.
        return starter.SwiGLU.forward(_PackSaves(ctx), gate, up)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            # create_graph is only known at backward, too late to redo forward.
            raise RuntimeError("SwiGLU packing is first-order only; remove installer before forward for higher-order work")
        gate16, up16 = ctx.saved_tensors
        # The unchanged starter kernel already promotes every load to FP32.
        # Feed its packed saves directly, preserving FP32 incoming/output grads.
        grad = grad.to(torch.float32).contiguous()
        dgate = torch.empty(gate16.shape, device=gate16.device, dtype=torch.float32)
        dup = torch.empty(up16.shape, device=up16.device, dtype=torch.float32)
        n = gate16.numel()
        starter = ctx.starter
        starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            grad, gate16, up16, dgate, dup, n, BLOCK=starter.BLOCK)
        return dgate, dup, None



def hybrid_mlp_forward(module, x, starter, threshold=1024):
    """Validated-entry helper; call ONLY after the canonical R2 guards pass.

    N = x.numel() / H. N <= threshold uses the exact R2 direct packed saves;
    larger inputs checkpoint frozen gate/up + starter SwiGLU, with down outside.
    Keep threshold fixed across the run. Higher-order/compile/adapters/hooks and
    noncanonical projection provenance remain excluded by the caller's guards.
    """
    if type(threshold) is not int or threshold < 1:
        raise ValueError("threshold must be a fixed positive integer")
    n = x.numel() // x.shape[-1]
    if n > threshold:
        def branch(hidden):
            return starter.swiglu(module.gate_proj(hidden), module.up_proj(hidden))

        out = checkpoint(branch, x, use_reentrant=False, preserve_rng_state=True)
    else:
        gate, up = module.gate_proj(x), module.up_proj(x)
        # Same post-projection metadata fallback as R2, without replaying either
        # projection. This is not a substitute for the caller's provenance guard.
        if (type(gate) is not torch.Tensor or type(up) is not torch.Tensor
                or gate.dtype != torch.float32 or up.dtype != torch.float32
                or gate.device != x.device or up.device != x.device
                or gate.shape != (*x.shape[:-1], module.intermediate_size)
                or up.shape != gate.shape):
            return module.down_proj(starter.swiglu(gate, up))
        out = _PackedSwiGLU.apply(gate, up, starter)
        del gate, up  # Release expanded operands before down projection.
    return module.down_proj(out)
