"""Fused RMSNorm, forward and backward in Triton, for Qwen2RMSNorm (28 x 2 + 1 = 57 calls per step).

What it replaces (transformers 5.16.1, modeling_qwen2.Qwen2RMSNorm.forward):
    x = hidden_states.to(float32); var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps);  return self.weight * x.to(input_dtype)
About 10 small PyTorch ops forward and about as many backward (fp32 mul, pow, mean, rsqrt, casts: the
largest part of the 285 ms/step elementwise group, 62.1 ms/step for the norms, docs/FINDINGS.md).

dtypes, measured on the T4 (op_profile): the residual stream is fp32 (Embedding and DecoderLayer outputs),
the norm weight is fp32 with requires_grad=False (peft's get_peft_model freezes everything but the LoRA
weights), the output is fp32. This kernel handles exactly that case: fp32 input, fp32 weight, same fp32
output, reductions in fp32. For fp32 input the original's `.to(input_dtype)` is the identity, so the math
is y = w * (x * rstd) with rstd = rsqrt(mean(x^2) + eps), the same order of operations.
Backward (dy = grad of y, xhat = x * rstd, c = mean(dy * w * xhat)):  dx = rstd * (dy * w - xhat * c).
The weight is frozen, so no weight gradient is computed; if a caller ever asks for it (requires_grad=True)
it is computed in plain PyTorch (dw = sum over rows of dy * xhat), so the result stays correct.
Any other dtype, a non-CUDA tensor or a row longer than 8192 goes to the original forward.
"""

import torch
import triton
import triton.language as tl

ENABLED = True

NUM_WARPS = 4        # fixed by hand: no autotune (the compile cache is empty in every run)
MAX_HIDDEN = 8192

_original = None
_patched = False


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, hidden, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / hidden + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * hidden + cols, (w * (x * rstd)).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, hidden, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden
    offs = row * hidden + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    wdy = dy * w
    c = tl.sum(wdy * xhat, 0) / hidden
    tl.store(dx_ptr + offs, ((wdy - xhat * c) * rstd).to(dx_ptr.dtype.element_ty), mask=mask)


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        hidden = x.shape[-1]
        x2 = x.reshape(-1, hidden).contiguous()
        y = torch.empty_like(x2)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        rmsnorm_fwd_kernel[(x2.shape[0],)](x2, weight, y, rstd, hidden, eps,
                                           BLOCK=triton.next_power_of_2(hidden), num_warps=NUM_WARPS)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2, weight, rstd = ctx.saved_tensors
        hidden = x2.shape[-1]
        dy = grad_out.reshape(-1, hidden).contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(x2.shape[0],)](dy, x2, weight, rstd, dx, hidden,
                                           BLOCK=triton.next_power_of_2(hidden), num_warps=NUM_WARPS)
        dw = None
        if ctx.needs_input_grad[1]:  # not the case in this task (frozen norm weight)
            dw = (dy.float() * (x2.float() * rstd[:, None])).sum(0).to(weight.dtype)
        return dx.view(ctx.shape), dw, None


def rmsnorm(x, weight, eps):
    return FusedRMSNorm.apply(x, weight, eps)


def _eligible(hidden_states, weight):
    return (hidden_states.is_cuda and hidden_states.dtype == torch.float32
            and weight.dtype == torch.float32 and weight.is_cuda
            and hidden_states.shape[-1] == weight.numel() <= MAX_HIDDEN and hidden_states.numel() > 0)


def patch():
    """Idempotent: Qwen2RMSNorm.forward -> the fused kernel for the fp32 case, original otherwise."""
    global _original, _patched
    if _patched:
        return
    from transformers.models.qwen2 import modeling_qwen2

    _original = modeling_qwen2.Qwen2RMSNorm.forward
    original = _original

    def forward(self, hidden_states):
        if not _eligible(hidden_states, self.weight):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
    _patched = True


def unpatch():
    """Undo patch() (used by the unit test)."""
    global _patched
    if _patched:
        from transformers.models.qwen2 import modeling_qwen2

        modeling_qwen2.Qwen2RMSNorm.forward = _original
        _patched = False
