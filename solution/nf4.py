"""Triton NF4 dequantization, replacing the CUDA kernel bitsandbytes uses to unpack 4-bit weights.

What it replaces (read in bitsandbytes 0.50.2, backends/cuda/ops.py):
  `_dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out)` - the only function that launches
  the C kernel `kDequantizeBlockwise<half, 512, 64, 8, 2>` (cdequantize_blockwise_fp16_nf4). It is looked up
  as a module global by both callers, so one attribute patch intercepts both:
    * forward:  torch.ops.bitsandbytes.gemm_4bit -> `_dequant_linear_fallback` (training has M = B*T = 2048
      rows, above the custom-kernel limit `_gemm_4bit_custom_max_m = 1536`) -> `_dequantize_4bit_impl`,
      then F.linear;
    * backward: MatMul4Bit.backward -> F.dequantize_4bit -> torch.ops.bitsandbytes.dequantize_4bit
      (the registered cuda op) -> `_dequantize_4bit_impl`, then torch.matmul for grad_A.
  That is 196 + 193 = 389 launches per step (docs/FINDINGS.md, profile). The nested (double-quantized)
  absmax is already decoded to fp32 by the callers before this function is reached, so the kernel gets a
  plain fp32 absmax with one entry per 64 weights.

Format (QLoRA, arXiv:2305.14314, and bitsandbytes' own pure-PyTorch reference in backends/default/ops.py):
  two 4-bit codes per byte, the FIRST weight in the high nibble; each code indexes the 16 NF4 values below;
  weight i is code * absmax[i // 64], computed in fp32 and rounded to the output dtype. Odd lengths leave
  the last low nibble unused. The kernel and the table are written here; only the layout is taken from the
  sources above (checked bit for bit against bitsandbytes in solution/test_nf4.py).

Anything else (fp4, other blocksize or dtype, non-CUDA, more than 2**31 weights) goes to the original function.
"""

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
BYTES_PER_PROGRAM = 512      # 1024 weights per program; fixed by hand (no autotune: the cache is empty each run)
NUM_WARPS = 4

_tables = {}
_original = None
_patched = False


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, n, BYTES: tl.constexpr,
                       BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0)
    byte_idx = pid * BYTES + tl.arange(0, BYTES)
    packed = tl.load(packed_ptr + byte_idx, mask=byte_idx * 2 < n, other=0).to(tl.int32)
    half = tl.arange(0, 2)
    # [BYTES, 2] tile: column 0 is the high nibble (weight 2i), column 1 the low nibble (weight 2i + 1)
    nibble = tl.where(half[None, :] == 0, packed[:, None] >> 4, packed[:, None] & 15)
    elem = byte_idx[:, None] * 2 + half[None, :]
    mask = elem < n
    value = tl.load(code_ptr + nibble) * tl.load(absmax_ptr + elem // BLOCKSIZE, mask=mask, other=0.0)
    tl.store(out_ptr + elem, value.to(out_ptr.dtype.element_ty), mask=mask)


def dequantize_nf4(packed, absmax, out):
    """Write the fp16 weights of `packed` (uint8, 2 codes per byte) into `out` (flat or shaped)."""
    n = out.numel()
    table = _tables.get(out.device)
    if table is None:
        table = _tables[out.device] = torch.tensor(NF4_CODE, dtype=torch.float32, device=out.device)
    grid = (triton.cdiv(triton.cdiv(n, 2), BYTES_PER_PROGRAM),)
    nf4_dequant_kernel[grid](packed, absmax, table, out, n, BYTES=BYTES_PER_PROGRAM,
                             BLOCKSIZE=BLOCKSIZE, num_warps=NUM_WARPS)
    return out


def _eligible(A, absmax, blocksize, quant_type, dtype, out):
    return (quant_type == "nf4" and blocksize == BLOCKSIZE and dtype == torch.float16
            and A.is_cuda and A.dtype == torch.uint8 and absmax.dtype == torch.float32
            and out.dtype == torch.float16 and out.is_contiguous() and out.numel() < 2**31
            and A.numel() * 2 >= out.numel() and absmax.numel() * BLOCKSIZE >= out.numel())


def patch():
    """Idempotent: send bitsandbytes' `_dequantize_4bit_impl` through the Triton kernel."""
    global _original, _patched
    if _patched:
        return
    try:
        from bitsandbytes.backends.cuda import ops
    except ImportError:  # no CUDA build of bitsandbytes: nothing to replace
        return
    if not hasattr(ops, "_dequantize_4bit_impl"):
        raise RuntimeError("bitsandbytes layout changed: backends.cuda.ops._dequantize_4bit_impl is gone")
    _original = ops._dequantize_4bit_impl

    def _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out):
        if not _eligible(A, absmax, blocksize, quant_type, dtype, out):
            return _original(A, absmax, blocksize, quant_type, dtype, out=out)
        dequantize_nf4(A.contiguous(), absmax.contiguous(), out)

    ops._dequantize_4bit_impl = _dequantize_4bit_impl
    _patched = True


def unpatch():
    """Undo patch() (used by the unit test to get bitsandbytes' own result)."""
    global _patched
    if _patched:
        from bitsandbytes.backends.cuda import ops

        ops._dequantize_4bit_impl = _original
        _patched = False
