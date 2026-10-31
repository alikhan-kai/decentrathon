"""Experimental INT8 input-gradient GEMM for frozen canonical NF4 MLP weights.

Forward and model storage stay unchanged. Temporary per-row INT8 weights are
derived anew from NF4 each backward; no cross-step weight cache. The supplied
challenge discussion explicitly permits this approximation (reply 2039 to 2037).
Rejected for speed: all six T4 microbenchmarks were slower (0.39-0.71x).
Eight focused tests passed; no full training benchmark. Not installed on import.
"""
import torch
import triton
import triton.language as tl

ENABLED = False
_original = None
_patched = False
_calls = 0


@triton.jit
def nf4_dgrad_weight_int8(packed, scales8, scale_code, scale_absmax, offset,
                          nf4_code, quant, row_scales, K: tl.constexpr,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    element = row * K + col
    byte = tl.load(packed + element // 2, col < K, 0).to(tl.int32)
    nibble = tl.where(element % 2 == 0, byte >> 4, byte & 15)
    block = element // 64
    scale_index = tl.load(scales8 + block, col < K, 0).to(tl.int32)
    scale = tl.load(scale_code + scale_index) * tl.load(scale_absmax + block // 256, col < K, 0)
    scale = scale + tl.load(offset)
    value = (tl.load(nf4_code + nibble) * scale).to(tl.float16).to(tl.float32)
    value = tl.where(col < K, value, 0.0)
    row_scale = tl.maximum(tl.max(tl.abs(value), 0) / 127.0, 1.0e-20)
    scaled = tl.div_rn(value, row_scale)
    rounded = tl.extra.cuda.libdevice.nearbyint(scaled)
    q = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(quant + element, q, col < K)
    tl.store(row_scales + row, row_scale)


@triton.jit(do_not_specialize=["M"])
def nf4_dgrad_input_int8(grad, weight_scales, quant, grad_scales,
                         M, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(grad + row * N + col, (row < M) & (col < N), 0).to(tl.float32)
    weight_scale = tl.load(weight_scales + col, col < N, 0)
    value = value * weight_scale
    maximum = tl.max(tl.abs(value), 0)
    # Preserve nonfinite signaling for AMP: do not turn NaN/Inf gradients into
    # finite integers and accidentally execute an optimizer step.
    bad = tl.max(((value != value) | (tl.abs(value) == float('inf'))).to(tl.int32), 0)
    scale = tl.maximum(maximum / 127.0, 1.0e-20)
    scale = tl.where(bad != 0, float('nan'), scale)
    q = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(value, scale))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(quant + row * N + col, q, col < N)
    tl.store(grad_scales + row, scale)


@triton.jit
def nf4_dgrad_output_int8(acc, scales, output, TOTAL, K: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(acc + pos, pos < TOTAL, 0).to(tl.float32)
    scale = tl.load(scales + pos // K, pos < TOTAL, 0)
    tl.store(output + pos, value * scale, pos < TOTAL)


def input_gradient(grad, packed, state):
    n, k = state.shape
    m = grad.numel() // n
    padded_m = triton.cdiv(m, 32) * 32
    grad = grad.contiguous()
    w8 = torch.empty((n, k), device=grad.device, dtype=torch.int8)
    ws = torch.empty(n, device=grad.device, dtype=torch.float32)
    nf4_dgrad_weight_int8[(n,)](
        packed, state.absmax, state.state2.code, state.state2.absmax,
        state.offset, state.code, w8, ws, k, triton.next_power_of_2(k),
        num_warps=8 if k > 4096 else 4, enable_fp_fusion=False)
    g8 = torch.empty((padded_m, n), device=grad.device, dtype=torch.int8)
    gs = torch.empty(padded_m, device=grad.device, dtype=torch.float32)
    nf4_dgrad_input_int8[(padded_m,)](
        grad, ws, g8, gs, m, n, triton.next_power_of_2(n),
        num_warps=8 if n > 4096 else 4, enable_fp_fusion=False)
    # Turing IMMA requires TN in cuBLAS column-major notation. PyTorch's
    # row-major left operand therefore needs a column-major right operand.
    accum = torch._int_mm(g8, w8.t().contiguous().t())
    out = torch.empty((*grad.shape[:-1], k), device=grad.device, dtype=grad.dtype)
    nf4_dgrad_output_int8[(triton.cdiv(out.numel(), 1024),)](
        accum, gs, out, out.numel(), k, 1024, enable_fp_fusion=False)
    return out


def eligible(ctx, grad):
    state = getattr(ctx, 'state', None)
    if (getattr(ctx, 'is_empty', True) or torch.is_grad_enabled()
            or type(grad) is not torch.Tensor or grad.device.type != 'cuda'
            or grad.dtype != torch.float16 or grad.ndim not in (2, 3)
            or not ctx.needs_input_grad[0] or any(ctx.needs_input_grad[1:])
            or ctx.dtype_A != torch.float16 or state is None):
        return False
    if (state.quant_type != 'nf4' or state.blocksize != 64 or not state.nested
            or state.state2.blocksize != 256 or state.dtype != torch.float16
            or tuple(state.shape) not in ((8960, 1536), (1536, 8960))
            or grad.shape[-1] != state.shape[0]
            or not 64 <= grad.numel() // grad.shape[-1] <= 2048):
        return False
    weight = ctx.tensors[1]
    return (type(weight) is torch.Tensor and not weight.requires_grad
            and weight.is_contiguous() and weight.dtype == torch.uint8
            and weight.device == grad.device and weight.numel() * 2 == state.shape.numel()
            and torch.cuda.get_device_capability(grad.device) == (7, 5))


def patch():
    global _original, _patched
    if _patched:
        return
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    if bnb.__version__ != '0.50.2':
        return
    _original = MatMul4Bit.backward

    def backward(ctx, grad):
        global _calls
        if not eligible(ctx, grad):
            return _original(ctx, grad)
        _calls += 1
        return input_gradient(grad, ctx.tensors[1], ctx.state), None, None, None, None

    MatMul4Bit.backward = staticmethod(backward)
    _patched = True
