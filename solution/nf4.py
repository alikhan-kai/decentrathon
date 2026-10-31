"""Triton NF4 dequantization (including fused nested double-quantization) + direct matmul_4bit fast path.

What it replaces (in bitsandbytes 0.50.2):
  1. `_dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out)` -> Triton `nf4_dequant_kernel`
     for pre-decoded fp32 `absmax`.
  2. Nested double-quantized NF4 (`quant_state.nested=True`, `blocksize=64`, `state2.blocksize=256`):
     fuses `F.dequantize_blockwise(state.absmax, state.state2) + state.offset` directly into
     `nf4_dequant_kernel(..., NESTED=True)` using IEEE-754 round-to-nearest-even `mul.rn.f32` +
     `add.rn.f32` so the output matches bitsandbytes bit for bit (`torch.equal`), while eliminating
     two extra GPU kernel launches and two temporary buffer allocations per projection.
  3. Direct `_FastNF4MatMul` for eligible nested NF4 `matmul_4bit` calls that use bitsandbytes'
     dequantize + cuBLAS linear path (`M >= 64` on sm < 8.0 or `M > 1536`), bypassing repeated
     `torch.ops.bitsandbytes` dispatcher round-trips in both forward and backward.
"""

import math

import torch
import triton
import triton.language as tl

ENABLED = True

# NormalFloat4 code book (QLoRA, appendix E); identical to bitsandbytes.functional.get_4bit_type("nf4").
NF4_CODE = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)

BLOCKSIZE = 64
BLOCKSIZE2 = 256
BYTES_PER_PROGRAM = 512      # 1024 weights per program; fixed by hand (no autotune: the cache is empty each run)
NUM_WARPS = 4

_tables = {}
_sm_cache = {}
_original = None
_orig_dequantize_4bit = None
_orig_matmul_4bit = None
_orig_fn_matmul_4bit = None
_orig_mod_matmul_4bit = None
_patched = False


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, code2_ptr, absmax2_ptr, offset_ptr,
                       out_ptr, n, BYTES: tl.constexpr, BLOCKSIZE: tl.constexpr,
                       BLOCKSIZE2: tl.constexpr = 256, NESTED: tl.constexpr = False):
    pid = tl.program_id(0)
    byte_idx = pid * BYTES + tl.arange(0, BYTES)
    byte_mask = byte_idx * 2 < n
    packed = tl.load(packed_ptr + byte_idx, mask=byte_mask, other=0).to(tl.int32)
    # High nibble (2*i) and low nibble (2*i + 1) of byte i always share block i // (BLOCKSIZE // 2).
    blk = byte_idx // (BLOCKSIZE // 2)
    if NESTED:
        qabs = tl.load(absmax_ptr + blk, mask=byte_mask, other=0).to(tl.int32)
        c2 = tl.load(code2_ptr + qabs, mask=byte_mask, other=0.0)
        a2 = tl.load(absmax2_ptr + (blk // BLOCKSIZE2), mask=byte_mask, other=0.0)
        off = tl.load(offset_ptr)
        # Match bitsandbytes' separate fp32 multiply then fp32 add without FMA contraction.
        scale = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;\n\tadd.rn.f32 $0, $0, $3;",
            "=f,f,f,f",
            [c2, a2, off],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        scale = tl.load(absmax_ptr + blk, mask=byte_mask, other=0.0)
    half = tl.arange(0, 2)
    # [BYTES, 2] tile: column 0 is the high nibble (weight 2i), column 1 the low nibble (weight 2i + 1)
    nibble = tl.where(half[None, :] == 0, packed[:, None] >> 4, packed[:, None] & 15)
    elem = byte_idx[:, None] * 2 + half[None, :]
    mask = elem < n
    value = tl.load(code_ptr + nibble) * scale[:, None]
    tl.store(out_ptr + elem, value.to(out_ptr.dtype.element_ty), mask=mask)


def _get_table(device):
    table = _tables.get(device)
    if table is None:
        table = _tables[device] = torch.tensor(NF4_CODE, dtype=torch.float32, device=device)
    return table


def _get_sm(device):
    idx = device.index or 0
    sm = _sm_cache.get(idx)
    if sm is None:
        sm = _sm_cache[idx] = torch.cuda.get_device_capability(idx)
    return sm


def dequantize_nf4(packed, absmax, out):
    """Write the fp16 weights of `packed` (uint8, 2 codes per byte) into `out` (flat or shaped)."""
    n = out.numel()
    table = _get_table(out.device)
    grid = (triton.cdiv(triton.cdiv(n, 2), BYTES_PER_PROGRAM),)
    nf4_dequant_kernel[grid](packed, absmax, table, None, None, None, out, n,
                             BYTES=BYTES_PER_PROGRAM, BLOCKSIZE=BLOCKSIZE,
                             BLOCKSIZE2=BLOCKSIZE2, NESTED=False, num_warps=NUM_WARPS)
    return out


def dequantize_nf4_nested(packed, qabsmax, code2, absmax2, offset, out, blocksize2=BLOCKSIZE2):
    """Write fp16 weights from double-quantized NF4 (`packed` + uint8 `qabsmax`) directly into `out`."""
    n = out.numel()
    table = _get_table(out.device)
    grid = (triton.cdiv(triton.cdiv(n, 2), BYTES_PER_PROGRAM),)
    nf4_dequant_kernel[grid](packed, qabsmax, table, code2, absmax2, offset, out, n,
                             BYTES=BYTES_PER_PROGRAM, BLOCKSIZE=BLOCKSIZE,
                             BLOCKSIZE2=blocksize2, NESTED=True, num_warps=NUM_WARPS)
    return out


def _eligible(A, absmax, blocksize, quant_type, dtype, out):
    return (quant_type == "nf4" and blocksize == BLOCKSIZE and dtype == torch.float16
            and A.is_cuda and A.dtype == torch.uint8 and absmax.dtype == torch.float32
            and out.dtype == torch.float16 and out.is_contiguous() and out.numel() < 2**31
            and A.numel() * 2 >= out.numel() and absmax.numel() * BLOCKSIZE >= out.numel())


def _get_offset_cuda(quant_state, device):
    cached = getattr(quant_state, "_ember_off_dev", None)
    if cached is not None and cached.device == device:
        return cached
    off = getattr(quant_state, "offset", None)
    if isinstance(off, torch.Tensor) and off.numel() == 1:
        if off.is_cuda and off.device == device and off.dtype == torch.float32:
            cached = off
        elif not off.is_cuda:
            cached = torch.tensor(float(off.item()), dtype=torch.float32, device=device)
        else:
            cached = off.to(device=device, dtype=torch.float32)
    elif isinstance(off, (int, float)):
        cached = torch.tensor(float(off), dtype=torch.float32, device=device)
    else:
        return None
    try:
        quant_state._ember_off_dev = cached
    except AttributeError:
        pass
    return cached


def _eligible_nested(A, quant_state, out=None, require_fp16_state=True):
    if (quant_state is None or not getattr(quant_state, "nested", False)
            or quant_state.quant_type != "nf4" or quant_state.blocksize != BLOCKSIZE):
        return False
    if require_fp16_state and quant_state.dtype != torch.float16:
        return False
    s2 = getattr(quant_state, "state2", None)
    off = getattr(quant_state, "offset", None)
    qabs = getattr(quant_state, "absmax", None)
    shape = getattr(quant_state, "shape", None)
    off_ok = isinstance(off, (int, float)) or (isinstance(off, torch.Tensor) and off.numel() == 1)
    if (s2 is None or s2.blocksize != BLOCKSIZE2 or shape is None or not off_ok
            or not isinstance(A, torch.Tensor) or not A.is_cuda or A.dtype != torch.uint8
            or not isinstance(qabs, torch.Tensor) or not qabs.is_cuda
            or qabs.dtype != torch.uint8 or not qabs.is_contiguous()
            or not isinstance(s2.code, torch.Tensor) or not s2.code.is_cuda
            or s2.code.dtype != torch.float32 or not s2.code.is_contiguous() or s2.code.numel() != 256
            or not isinstance(s2.absmax, torch.Tensor) or not s2.absmax.is_cuda
            or s2.absmax.dtype != torch.float32 or not s2.absmax.is_contiguous()):
        return False
    shape_t = tuple(shape)
    n = math.prod(shape_t)
    if (n <= 0 or n >= 2**31 or A.numel() * 2 < n
            or qabs.numel() * BLOCKSIZE < n or s2.absmax.numel() * BLOCKSIZE2 < qabs.numel()):
        return False
    if out is not None:
        if (not isinstance(out, torch.Tensor) or not out.is_cuda or out.dtype != torch.float16
                or not out.is_contiguous() or tuple(out.shape) != shape_t):
            return False
    return True


class _FastNF4MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B, bias, quant_state, off_dev):
        ctx.B = B
        ctx.quant_state = quant_state
        ctx.off_dev = off_dev
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        s2 = quant_state.state2
        w = torch.empty(quant_state.shape, dtype=torch.float16, device=x.device)
        dequantize_nf4_nested(B, quant_state.absmax, s2.code, s2.absmax, off_dev, w)
        return torch._C._nn.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_A = None
        if ctx.needs_input_grad[0]:
            qs = ctx.quant_state
            s2 = qs.state2
            w = torch.empty(qs.shape, dtype=torch.float16, device=grad_output.device)
            dequantize_nf4_nested(ctx.B, qs.absmax, s2.code, s2.absmax, ctx.off_dev, w)
            if grad_output.dtype != torch.float16:
                w = w.to(grad_output.dtype)
            g2d = grad_output.contiguous().view(-1, w.shape[0])
            grad_A = torch.mm(g2d, w).view(ctx.x_shape)
            if grad_A.dtype != ctx.x_dtype:
                grad_A = grad_A.to(ctx.x_dtype)
        return grad_A, None, None, None, None


def patch():
    """Idempotent: route bitsandbytes' NF4 dequantization and eligible matmul_4bit calls through Triton."""
    global _original, _orig_dequantize_4bit, _orig_matmul_4bit, _orig_fn_matmul_4bit, _orig_mod_matmul_4bit, _patched
    if _patched:
        return
    try:
        import bitsandbytes as bnb
        import bitsandbytes.autograd._functions as bnb_fns
        import bitsandbytes.functional as bnb_F
        import bitsandbytes.nn.modules as bnb_mods
        from bitsandbytes.backends.cuda import ops
    except ImportError:  # no CUDA build of bitsandbytes: nothing to replace
        return
    if not hasattr(ops, "_dequantize_4bit_impl"):
        raise RuntimeError("bitsandbytes layout changed: backends.cuda.ops._dequantize_4bit_impl is gone")
    _original = ops._dequantize_4bit_impl
    _orig_dequantize_4bit = bnb_F.dequantize_4bit
    _orig_matmul_4bit = bnb.matmul_4bit
    _orig_fn_matmul_4bit = bnb_fns.matmul_4bit
    _orig_mod_matmul_4bit = getattr(bnb_mods, "matmul_4bit", None)

    def _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out):
        if not _eligible(A, absmax, blocksize, quant_type, dtype, out):
            return _original(A, absmax, blocksize, quant_type, dtype, out=out)
        if not A.is_contiguous():
            A = A.contiguous()
        if not absmax.is_contiguous():
            absmax = absmax.contiguous()
        dequantize_nf4(A, absmax, out)

    def _fast_dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=64, quant_type="fp4"):
        if absmax is None and _eligible_nested(A, quant_state, out, require_fp16_state=True):
            off_dev = _get_offset_cuda(quant_state, A.device)
            if off_dev is not None:
                if not A.is_contiguous():
                    A = A.contiguous()
                if out is None:
                    out = torch.empty(quant_state.shape, dtype=torch.float16, device=A.device)
                s2 = quant_state.state2
                return dequantize_nf4_nested(A, quant_state.absmax, s2.code, s2.absmax, off_dev, out)
        return _orig_dequantize_4bit(A, quant_state=quant_state, absmax=absmax, out=out,
                                     blocksize=blocksize, quant_type=quant_type)

    def _fast_matmul_4bit(A, B, out=None, bias=None, quant_state=None):
        if (out is None and isinstance(A, torch.Tensor) and A.is_cuda and A.dtype == torch.float16
                and A.ndim >= 2 and isinstance(B, torch.Tensor)
                and (B.ndim == 1 or (B.ndim == 2 and (B.shape[0] == 1 or B.shape[1] == 1)))
                and B.is_contiguous() and not B.requires_grad
                and (bias is None or (isinstance(bias, torch.Tensor) and bias.is_cuda
                                      and bias.dtype == torch.float16 and not bias.requires_grad))
                and _eligible_nested(B, quant_state, None, require_fp16_state=False)
                and A.shape[-1] == quant_state.shape[1] and A.shape[-1] > 0
                and not torch.compiler.is_compiling()
                and not torch._C._are_functorch_transforms_active()):
            m = A.numel() // A.shape[-1]
            min_m = 64 if _get_sm(A.device) < (8, 0) else 1537
            if m >= min_m:
                off_dev = _get_offset_cuda(quant_state, A.device)
                if off_dev is not None:
                    if not torch.is_grad_enabled() or not A.requires_grad:
                        s2 = quant_state.state2
                        w = torch.empty(quant_state.shape, dtype=torch.float16, device=A.device)
                        dequantize_nf4_nested(B, quant_state.absmax, s2.code, s2.absmax, off_dev, w)
                        return torch._C._nn.linear(A, w, bias)
                    return _FastNF4MatMul.apply(A, B, bias, quant_state, off_dev)
        return _orig_matmul_4bit(A, B, out=out, bias=bias, quant_state=quant_state)

    ops._dequantize_4bit_impl = _dequantize_4bit_impl
    bnb_F.dequantize_4bit = _fast_dequantize_4bit
    bnb.matmul_4bit = _fast_matmul_4bit
    bnb_fns.matmul_4bit = _fast_matmul_4bit
    if _orig_mod_matmul_4bit is not None:
        bnb_mods.matmul_4bit = _fast_matmul_4bit
    _patched = True


def unpatch():
    """Undo patch() (used by the unit test to get bitsandbytes' own result)."""
    global _patched
    if _patched:
        import bitsandbytes as bnb
        import bitsandbytes.autograd._functions as bnb_fns
        import bitsandbytes.functional as bnb_F
        import bitsandbytes.nn.modules as bnb_mods
        from bitsandbytes.backends.cuda import ops

        ops._dequantize_4bit_impl = _original
        bnb_F.dequantize_4bit = _orig_dequantize_4bit
        bnb.matmul_4bit = _orig_matmul_4bit
        bnb_fns.matmul_4bit = _orig_fn_matmul_4bit
        if _orig_mod_matmul_4bit is not None:
            bnb_mods.matmul_4bit = _orig_mod_matmul_4bit
        _patched = False
