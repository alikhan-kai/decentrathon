"""Private frozen-NF4 branch rematerialization, after unchanged PR7 guards.

Only hybrid_mlp's N > 1024 branch calls this helper. Down remains outside;
only the canonical FP16 projection input is saved across forward/backward. This is
first-order eager training only, exactly as declared to PR7's installer.

Uses the unchanged starter kernels. ALL starter SwiGLU credit stays excluded.
The existing team's NF4 implementation and attribution are unchanged.
"""
import torch


def _autocast_state(device_type):
    return dict(enabled=torch.is_autocast_enabled(device_type),
                dtype=torch.get_autocast_dtype(device_type),
                cache_enabled=torch.is_autocast_cache_enabled())


def _project(module, hidden):
    # Two distinct cast nodes are ESSENTIAL: each branch's FP16 input gradient
    # must expand to FP32 before addition. A shared cast changes 2049 to 2048.
    gate = module.gate_proj(hidden.to(torch.float16))
    up = module.up_proj(hidden.to(torch.float16))
    return gate, up


class _FrozenBranchRemat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_up, hidden_gate, module, starter):
        # Baseline creates gate then up; its ready up branch runs first.
        # Duplicate outer edges keep both FP32 contributions in that order.
        assert hidden_up is hidden_gate
        hidden = hidden_gate
        ctx.set_materialize_grads(False)
        ctx.module, ctx.starter = module, starter
        ctx.device_index = hidden.device.index
        ctx.cpu_rng = torch.get_rng_state()
        ctx.cuda_rng = torch.cuda.get_rng_state(hidden.device)
        ctx.cuda_amp, ctx.cpu_amp = _autocast_state("cuda"), _autocast_state("cpu")
        ctx.hidden_dtype = hidden.dtype
        # Both canonical frozen projections consume this exact FP16 rounding.
        # Backward needs those projection inputs, not their pre-cast FP32 values.
        # Keep the TWO outer FP32 edges: half cotangents must never add in FP16.
        ctx.save_for_backward(hidden.to(torch.float16))
        gate, up = _project(module, hidden)
        # Canonical bnb returns contiguous half. Keep the starter's layout
        # normalization even if that implementation detail changes.
        gate, up = gate.contiguous(), up.contiguous()
        assert gate.dtype == up.dtype == torch.float16
        assert gate.shape == up.shape == (*hidden.shape[:-1], module.intermediate_size)
        out = torch.empty(gate.shape, device=gate.device, dtype=torch.float16)
        n = gate.numel()
        starter.example_swiglu_fwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            gate, up, out, n, BLOCK=starter.BLOCK)
        return out

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("frozen branch rematerialization is first-order only; remove installer before higher-order work")
        if grad is None or not any(ctx.needs_input_grad[:2]):
            return None, None, None, None
        (saved_hidden,) = ctx.saved_tensors
        # Match checkpoint's CPU/device RNG restoration and its forward-time
        # autocast replay. RNG restoration cannot escape this context; the
        # unchanged canonical guards exclude stateful projection variants.
        with torch.random.fork_rng(devices=[ctx.device_index], enabled=True):
            torch.set_rng_state(ctx.cpu_rng)
            torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device_index)
            with torch.enable_grad(), torch.autocast("cuda", **ctx.cuda_amp), \
                    torch.autocast("cpu", **ctx.cpu_amp):
                hidden_gate = saved_hidden.detach().requires_grad_(True)
                hidden_up = saved_hidden.detach().requires_grad_(True)
                gate = ctx.module.gate_proj(hidden_gate.to(torch.float16))
                up = ctx.module.up_proj(hidden_up.to(torch.float16))
        # Replay only the projections. The forward starter launch is unnecessary
        # here because its backward depends just on the rematerialized operands.
        # Store each unchanged starter result directly at its independent FP16
        # bnb output boundary. No starter/compiler specialization credit.
        assert grad.dtype == torch.float16
        # Unchanged starter loads this half cotangent into FP32 registers.
        grad = grad.contiguous()
        dgate16 = torch.empty(gate.shape, device=gate.device, dtype=torch.float16)
        dup16 = torch.empty(up.shape, device=up.device, dtype=torch.float16)
        starter, n = ctx.starter, gate.numel()
        starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            grad, gate.contiguous(), up.contiguous(), dgate16, dup16, n, BLOCK=starter.BLOCK)
        del grad
        # BNB backward needs its grad_fn and frozen weights, not output storage.
        gate_edge = torch.autograd.graph.get_gradient_edge(gate)
        up_edge = torch.autograd.graph.get_gradient_edge(up)
        del gate, up
        # Independent leaves prevent nested autograd from summing the branches.
        (dhidden_up,) = torch.autograd.grad(up_edge, hidden_up, dup16,
                                          create_graph=False, allow_unused=False)
        del up_edge, dup16
        (dhidden_gate,) = torch.autograd.grad(gate_edge, hidden_gate, dgate16,
                                            create_graph=False, allow_unused=False)
        del gate_edge, dgate16
        return dhidden_up.to(ctx.hidden_dtype), dhidden_gate.to(ctx.hidden_dtype), None, None

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError("frozen branch rematerialization does not support forward AD")


def frozen_branch_remat(module, hidden, starter):
    """Internal validated entry; never call before MeasurementPatch.reason_for."""
    return _FrozenBranchRemat.apply(hidden, hidden, module, starter)
