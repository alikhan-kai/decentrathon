"""Original row-wise Triton RMSNorm for Qwen2's frozen normalization weights.

Qwen2 rounds normalized activations to the input dtype *before* multiplying
by weight. In particular, fp16 inputs and fp32 weights produce fp32 outputs.
The backward kernel preserves the corresponding fp16 gradient boundary too.
Triton is loaded only when a supported CUDA tensor reaches this module.
"""

import functools
import math

import torch


def _eager_rms_norm(x, weight, eps):
    input_dtype = x.dtype
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def _supports_triton(x, weight, eps):
    """Keep unusual layouts, trainable weights and tensor subclasses eager."""
    return (
        type(x) is torch.Tensor
        and type(weight) in (torch.Tensor, torch.nn.Parameter)
        and x.device.type == "cuda"
        and torch.version.hip is None
        and weight.device == x.device
        and x.dtype in (torch.float16, torch.float32)
        and weight.dtype in (torch.float16, torch.float32)
        and x.layout == torch.strided
        and weight.layout == torch.strided
        and x.ndim >= 1
        and weight.ndim == 1
        and 0 < x.shape[-1] <= 8192
        and weight.shape[0] == x.shape[-1]
        and 0 < x.numel() < 2**31
        and x.is_contiguous()
        and weight.is_contiguous()
        and not weight.requires_grad
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
    )


@functools.lru_cache(maxsize=1)
def _triton_kernels():
    # JIT functions resolve their language helpers from module globals. Keeping
    # the imports here lets CPU use (and importing solution) work without Triton.
    global triton, tl
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as exc:
        if exc.name == "triton":
            return None
        raise

    @triton.jit
    def _rmsnorm_forward(
        X, W, Y, INV_RMS, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        valid = column < H
        raw = tl.load(X + row * H + column, valid, other=0)
        value = raw.to(tl.float32)
        variance = tl.sum(value * value, axis=0) / H
        inv_rms = tl.rsqrt(variance + EPS)
        # This cast is observable when W is fp32. Do not merge it with Y's cast.
        rounded = (value * inv_rms).to(raw.dtype).to(tl.float32)
        weight = tl.load(W + column, valid, other=0).to(tl.float32)
        tl.store(Y + row * H + column, rounded * weight, valid)
        tl.store(INV_RMS + row, inv_rms)

    @triton.jit
    def _rmsnorm_backward(
        X, W, INV_RMS, DY, DX, H: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        valid = column < H
        raw = tl.load(X + row * H + column, valid, other=0)
        value = raw.to(tl.float32)
        weight = tl.load(W + column, valid, other=0).to(tl.float32)
        upstream = tl.load(DY + row * H + column, valid, other=0).to(tl.float32)
        inv_rms = tl.load(INV_RMS + row)
        # MulBackward accumulates into the normalized activation's dtype, then
        # ToCopyBackward converts it back to fp32 for the normalization graph.
        grad_normalized = (upstream * weight).to(raw.dtype).to(tl.float32)
        grad_inv_rms = tl.sum(grad_normalized * value, axis=0)
        cube = (inv_rms * inv_rms) * inv_rms
        grad_variance = (grad_inv_rms * -0.5) * cube
        grad_square = grad_variance / H
        grad_value = grad_normalized * inv_rms + (2.0 * value) * grad_square
        tl.store(DX + row * H + column, grad_value, valid)

    return _rmsnorm_forward, _rmsnorm_backward


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        forward_kernel, _ = _triton_kernels()
        width = x.shape[-1]
        rows = x.numel() // width
        block = triton.next_power_of_2(width)
        warps = 4 if block <= 2048 else 8
        output = torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))
        inv_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
        # The input can live on a CUDA device different from the current one.
        with torch.cuda.device(x.device):
            forward_kernel[(rows,)](
                x, weight, output, inv_rms, width, eps, block,
                num_warps=warps, enable_fp_fusion=False,
            )
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.width, ctx.block, ctx.warps, ctx.eps = width, block, warps, eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        if torch.is_grad_enabled():
            # create_graph=True needs a differentiable backward. Rebuild the
            # original graph rather than silently returning detached gradients.
            output = _eager_rms_norm(x, weight, ctx.eps)
            dx, = torch.autograd.grad(output, x, grad_output, create_graph=True)
            return dx, None, None
        _, backward_kernel = _triton_kernels()
        dx = torch.empty_like(x)
        with torch.cuda.device(x.device):
            backward_kernel[(inv_rms.numel(),)](
                x, weight, inv_rms, grad_output.contiguous(), dx,
                ctx.width, ctx.block, num_warps=ctx.warps, enable_fp_fusion=False,
            )
        return dx, None, None


def rms_norm(x, weight, eps):
    """Qwen2 RMSNorm with a frozen-weight CUDA fast path and eager fallback.

    Supported kernels: contiguous fp16/fp32 CUDA inputs, contiguous frozen
    fp16/fp32 weights, last dimension 1..8192, finite nonnegative scalar eps.
    All other cases retain eager PyTorch's dtype, broadcasting and autograd.
    """
    if _supports_triton(x, weight, eps) and _triton_kernels() is not None:
        return _FrozenRMSNorm.apply(x, weight, eps)
    return _eager_rms_norm(x, weight, eps)


def patch_rmsnorm():
    """Idempotently patch Qwen2RMSNorm.forward; return False if unavailable.

    Unsupported calls use the saved original method. Parameters and model
    configuration are untouched. Importing this module does not apply a patch.
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except ImportError:
        return False
    original = getattr(Qwen2RMSNorm, "forward", None)
    if not callable(original):
        return False
    if getattr(original, "_soup_rmsnorm_original_forward", None) is not None:
        return True

    @functools.wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        weight = getattr(self, "weight", None)
        eps = getattr(self, "variance_epsilon", None)
        if (
            not args and not kwargs
            and _supports_triton(hidden_states, weight, eps)
            and _triton_kernels() is not None
        ):
            return _FrozenRMSNorm.apply(hidden_states, weight, eps)
        return original(self, hidden_states, *args, **kwargs)

    forward._soup_rmsnorm_original_forward = original
    Qwen2RMSNorm.forward = forward
    return True
