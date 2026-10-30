"""Opt-in NF4 -> fp16 dense decode, with nested scale reconstruction fused in.

This is independently written Triton code for the established bitsandbytes NF4
format, not a new quantization algorithm. Format constants and validation are
shared with ``solution.nf4``. The pinned implementation inspected for rounding
and dispatch is bitsandbytes 0.50.2, functional.py, autograd/_functions.py and
backends/cuda/ops.py. No bitsandbytes or third-party kernel code is reused.

One program decodes consecutive packed bytes; high nibble precedes low nibble.
Scale groups follow flattened weights, not matrix rows. For nested states the
8-bit code times second-level absmax rounds to fp32 BEFORE offset addition,
which also rounds to fp32. NF4 constant times scale rounds once to fp16 after
the fp32 product. Launches disable FMA contraction. No scale/dense cache exists.

Only uint8 storage, fp16 output, blocksize=64 and optional single-level nested
blocksize=256 are supported. ``dequantize_nf4`` is an explicit CUDA primitive;
unsupported calls raise. ``patch_dequant`` is separately opt-in and falls back
to saved originals for unsupported calls, including CPU and graph tracing.
Importing this file changes nothing; solution.apply does not enable it.

The patch replaces only public functional.dequantize_4bit and CUDA's
_dequant_linear_fallback, preserving the existing fused-forward dispatch and
all PyTorch/cuBLAS GEMMs. It does not intercept torch.ops dequantize_4bit calls
or pre-existing imported function aliases. The default kernel has passed an
isolated T4 correctness probe; its large nested decode was slower than bnb.
The private variants below remain unvalidated on T4. CPU tests and SM75
compilation do not establish GPU correctness, performance or eligibility.
"""

import functools
import sys
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .nf4 import (
    NF4_CODEBOOK,
    _check_tensor,
    _validate_quant_state,
    dequantize_nf4_reference as _dense_reference,
    triton,
    tl,
)


_BLOCK_BYTES = 512
_PATCH_MARKER = "_ember_dequant_original"


def _validate(packed, state, out):
    n, k, _ = _validate_quant_state(packed, state)
    tensors = [packed, state.absmax]
    if state.nested:
        tensors.extend((state.state2.code, state.state2.absmax, state.offset))
    if any(isinstance(value, FakeTensor) for value in (*tensors, out)):
        raise NotImplementedError("fake tensors are outside this eager candidate")
    if out is not None:
        _check_tensor(out, "out", dtype=torch.float16, device=packed.device, numel=n * k, ndim=2)
        if tuple(out.shape) != (n, k):
            raise ValueError("out.shape must equal quant_state.shape before the transpose shim")
        # Conservative shared-storage check: unsupported aliases retain bnb's
        # original behavior, rather than introducing read/write kernel races.
        if any(torch._C._overlaps(out, value) for value in tensors):
            raise NotImplementedError("out must not alias packed weights or scale metadata")
    return n, k


def _require_cuda(packed):
    if packed.device.type != "cuda" or torch.version.hip is not None:
        raise NotImplementedError("dequantize_nf4 requires NVIDIA CUDA")
    if triton is None:
        raise NotImplementedError("dequantize_nf4 requires Triton")
    if torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.jit.is_scripting():
        raise NotImplementedError("graph compilation/tracing is outside this eager candidate")
    if torch.cuda.get_device_capability(packed.device) < (7, 5):
        raise NotImplementedError("dequantize_nf4 requires NVIDIA SM75 or newer")


def supports_dequantize_nf4(packed, quant_state, out=None):
    """Metadata-only capability guard; never copies or reads CUDA tensor values."""
    try:
        _validate(packed, quant_state, out)
        _require_cuda(packed)
    except (ValueError, NotImplementedError, AttributeError, TypeError):
        return False
    return True


def _orient(packed, dense):
    # Exact 0.50.2 BC shim, including a one-byte flat/column tensor. This
    # returns a VIEW: out is always filled in state.shape before transposition.
    return dense.t() if packed.shape[0] == 1 else dense


@torch.no_grad()
def dequantize_nf4_reference(packed, quant_state, out=None):
    """CPU/CUDA PyTorch oracle with bitsandbytes' out and transpose semantics."""
    _validate(packed, quant_state, out)
    dense = _dense_reference(packed, quant_state)
    if out is not None:
        out.copy_(dense)
        dense = out
    return _orient(packed, dense)


if triton is not None:

    @triton.jit
    def _dequantize_nf4_kernel(
        Packed, Absmax, ScaleCode, ScaleAbsmax, Offset, Out,
        NUMEL: tl.constexpr, NESTED: tl.constexpr,
        BLOCK_BYTES: tl.constexpr, CODEBOOK: tl.constexpr,
    ):
        byte = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        valid = 2 * byte < NUMEL
        packed = tl.load(Packed + byte, valid, other=0).to(tl.int32)
        group = byte // 32  # 64 weights / 2 weights per stored byte.
        if NESTED:
            scale_index = tl.load(Absmax + group, valid, other=0).to(tl.int32)
            code = tl.load(ScaleCode + scale_index)
            magnitude = tl.load(ScaleAbsmax + group // 256, valid, other=0)
            centered = code * magnitude
            scale = centered + tl.load(Offset)
        else:
            scale = tl.load(Absmax + group, valid, other=0)
        high = packed >> 4
        low = packed & 15
        high_value = tl.full((BLOCK_BYTES,), 0, tl.float32)
        low_value = tl.full((BLOCK_BYTES,), 0, tl.float32)
        for index in tl.static_range(16):
            high_value = tl.where(high == index, CODEBOOK[index], high_value)
            low_value = tl.where(low == index, CODEBOOK[index], low_value)
        high_weight = (high_value * scale).to(tl.float16)
        low_weight = (low_value * scale).to(tl.float16)
        weight = tl.join(high_weight, low_weight).reshape((2 * BLOCK_BYTES,))
        position = tl.program_id(0) * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
        tl.store(Out + position, weight, position < NUMEL)


    @triton.jit
    def _nf4_tree(index, CODEBOOK: tl.constexpr):
        # Four dependent levels / fifteen selects, with four reused bit tests.
        # Constants stay binary32; index 7 is positive zero, not a shortcut
        # around the later scale multiplication (which preserves signed zero).
        bit0 = (index & 1) != 0
        bit1 = (index & 2) != 0
        bit2 = (index & 4) != 0
        bit3 = (index & 8) != 0
        v01 = tl.where(bit0, CODEBOOK[1], CODEBOOK[0])
        v23 = tl.where(bit0, CODEBOOK[3], CODEBOOK[2])
        v45 = tl.where(bit0, CODEBOOK[5], CODEBOOK[4])
        v67 = tl.where(bit0, CODEBOOK[7], CODEBOOK[6])
        v89 = tl.where(bit0, CODEBOOK[9], CODEBOOK[8])
        vab = tl.where(bit0, CODEBOOK[11], CODEBOOK[10])
        vcd = tl.where(bit0, CODEBOOK[13], CODEBOOK[12])
        vef = tl.where(bit0, CODEBOOK[15], CODEBOOK[14])
        v03 = tl.where(bit1, v23, v01)
        v47 = tl.where(bit1, v67, v45)
        v8b = tl.where(bit1, vab, v89)
        vcf = tl.where(bit1, vef, vcd)
        return tl.where(bit3, tl.where(bit2, vcf, v8b), tl.where(bit2, v47, v03))


    @triton.jit
    def _dequantize_nf4_variant_kernel(
        Packed, Absmax, ScaleCode, ScaleAbsmax, Offset, Out,
        NUMEL: tl.constexpr, NESTED: tl.constexpr,
        BLOCK_BYTES: tl.constexpr, CODEBOOK: tl.constexpr, VARIANT: tl.constexpr,
        PAIR_ALIGNED: tl.constexpr,
    ):
        byte = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        if VARIANT == "grouped":
            # Each row is one flattened 64-weight scale group, including rows
            # that straddle matrix boundaries. Reconstruct before broadcasting.
            group = tl.program_id(0) * (BLOCK_BYTES // 32) + tl.arange(0, BLOCK_BYTES // 32)
            group_valid = 64 * group < NUMEL
            if NESTED:
                scale_index = tl.load(Absmax + group, group_valid, other=0).to(tl.int32)
                code = tl.load(ScaleCode + scale_index)
                magnitude = tl.load(ScaleAbsmax + group // 256, group_valid, other=0)
                centered = code * magnitude
                scale = centered + tl.load(Offset)
            else:
                scale = tl.load(Absmax + group, group_valid, other=0)
            scale = tl.broadcast_to(scale[:, None], (BLOCK_BYTES // 32, 32)).reshape((BLOCK_BYTES,))
        else:
            group = byte // 32
            valid = 2 * byte < NUMEL
            if NESTED:
                scale_index = tl.load(Absmax + group, valid, other=0).to(tl.int32)
                code = tl.load(ScaleCode + scale_index)
                magnitude = tl.load(ScaleAbsmax + group // 256, valid, other=0)
                centered = code * magnitude
                scale = centered + tl.load(Offset)
            else:
                scale = tl.load(Absmax + group, valid, other=0)
        packed = tl.load(Packed + byte, 2 * byte < NUMEL, other=0).to(tl.int32)
        high = (_nf4_tree(packed >> 4, CODEBOOK) * scale).to(tl.float16)
        low = (_nf4_tree(packed & 15, CODEBOOK) * scale).to(tl.float16)
        if VARIANT == "tree":
            weight = tl.join(high, low).reshape((2 * BLOCK_BYTES,))
            position = tl.program_id(0) * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
            tl.store(Out + position, weight, position < NUMEL)
        elif PAIR_ALIGNED:
            # CUDA is little endian: adjacent fp16 bits fit one uint32 store.
            # Reinterpret only after both rounded fp16 values exist. This keeps
            # byte ownership through output and needs no layout conversion.
            pair = high.to(tl.uint16, bitcast=True).to(tl.uint32)
            pair = pair | (low.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
            tl.store(Out.to(tl.pointer_type(tl.uint32)) + byte, pair, 2 * byte + 1 < NUMEL)
            if NUMEL % 2:
                tl.store(Out + 2 * byte, high, 2 * byte == NUMEL - 1)
        else:
            # Contiguous out views may be aligned to only two bytes. Keep those
            # legal without issuing a misaligned uint32 store or allocating.
            tl.store(Out + 2 * byte, high, 2 * byte < NUMEL)
            tl.store(Out + 2 * byte + 1, low, 2 * byte + 1 < NUMEL)


def _launch_variant(packed, state, out, *, variant, block_bytes=512, num_warps=4):
    """Private experimental launch into validated storage; no default selection.

    Three cumulative hypotheses: ``tree`` changes only the constant lookup;
    ``paired`` also packs adjacent fp16 bits into aligned uint32 stores;
    ``grouped`` additionally reconstructs scales once per 64-weight group.
    Legal static configurations are deliberately bounded, with no autotuning,
    persistent buffers, value-dependent routing, or CPU implementation.
    """
    if variant not in ("tree", "paired", "grouped"):
        raise ValueError("variant must be tree, paired, or grouped")
    if type(block_bytes) is not int or block_bytes not in (256, 512, 1024):
        raise ValueError("block_bytes must be 256, 512, or 1024")
    if type(num_warps) is not int or num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    nested = state.nested
    code = state.state2.code if nested else state.absmax
    magnitude = state.state2.absmax if nested else state.absmax
    offset = state.offset if nested else state.absmax
    _dequantize_nf4_variant_kernel[(triton.cdiv(out.numel(), 2 * block_bytes),)](
        packed, state.absmax, code, magnitude, offset, out,
        out.numel(), nested, block_bytes, NF4_CODEBOOK, variant, out.data_ptr() % 4 == 0,
        num_warps=num_warps, num_stages=1, enable_fp_fusion=False,
    )


def _dequantize_nf4_variant(packed, quant_state, out=None, *, variant,
                            block_bytes=512, num_warps=4):
    """Private full eager-decode benchmark entry, including guards/allocation.

    Example: ``_dequantize_nf4_variant(packed, state, variant="paired")``.
    Uses the same capability checks, out/transpose semantics and current CUDA
    stream as the baseline. Neither dequantize_nf4 nor patch_dequant selects it.
    Test patch paths in isolation by temporarily replacing ``_launch`` with
    ``functools.partial(_launch_variant, variant="paired")`` and restoring it
    afterwards. No variant is a promoted integration choice.
    """
    _validate(packed, quant_state, out)
    _require_cuda(packed)
    if out is None:
        out = torch.empty(quant_state.shape, dtype=torch.float16, device=packed.device)
    with torch.cuda.device(packed.device):
        _launch_variant(packed, quant_state, out, variant=variant,
                        block_bytes=block_bytes, num_warps=num_warps)
    return _orient(packed, out)


def _launch(packed, state, out):
    """One decode launch into a validated contiguous, untransposed output."""
    nested = state.nested
    # In the ordinary specialization these three pointer arguments are dead.
    code = state.state2.code if nested else state.absmax
    magnitude = state.state2.absmax if nested else state.absmax
    offset = state.offset if nested else state.absmax
    _dequantize_nf4_kernel[(triton.cdiv(out.numel(), 2 * _BLOCK_BYTES),)](
        packed, state.absmax, code, magnitude, offset, out,
        out.numel(), nested, _BLOCK_BYTES, NF4_CODEBOOK,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )


def _dense_cuda(packed, state, out=None):
    if out is None:
        out = torch.empty(state.shape, dtype=torch.float16, device=packed.device)
    with torch.cuda.device(packed.device):
        _launch(packed, state, out)
    return out


def dequantize_nf4(packed, quant_state, out=None):
    """Decode frozen canonical NF4 on CUDA; unsupported calls raise.

    Returns W[N,K] except when packed.shape[0]==1, which returns W.T exactly
    like bitsandbytes 0.50.2. A supplied out must be contiguous fp16 [N,K],
    frozen and disjoint from inputs. The returned transpose shares out storage.
    No fallback, patch, tuning, or persistent allocation occurs here.
    """
    _validate(packed, quant_state, out)
    _require_cuda(packed)
    return _orient(packed, _dense_cuda(packed, quant_state, out))


def _fallback_state(shape, dtype, absmax, blocksize, quant_type,
                    absmax_8bit, absmax_code, absmax_offset):
    nested = absmax_8bit is not None
    # No device values are read, and no QuantState serialization/ownership is
    # involved: these are the backend's already-unpacked metadata arguments.
    return SimpleNamespace(
        shape=shape, dtype=dtype, quant_type=quant_type, blocksize=blocksize,
        absmax=absmax_8bit if nested else absmax, nested=nested,
        state2=SimpleNamespace(absmax=absmax, code=absmax_code, blocksize=256,
                               dtype=torch.float32, nested=False, state2=None) if nested else None,
        offset=absmax_offset,
    )


def _patch_functional(functional):
    original = functional.dequantize_4bit
    if getattr(original, _PATCH_MARKER, None) is not None:
        return

    @functools.wraps(original)
    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
        # quant_state is authoritative in bnb; absmax/blocksize/quant_type
        # arguments are ignored when it is supplied. The legacy no-state path
        # remains original, including its validation and error behavior.
        if quant_state is not None and supports_dequantize_nf4(A, quant_state, out):
            return _orient(A, _dense_cuda(A, quant_state, out))
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    setattr(dequantize_4bit, _PATCH_MARKER, original)
    functional.dequantize_4bit = dequantize_4bit


def _patch_cuda_fallback(backend):
    original = backend._dequant_linear_fallback
    if getattr(original, _PATCH_MARKER, None) is not None:
        return

    @functools.wraps(original)
    def linear_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                        absmax_8bit=None, absmax_code=None, absmax_offset=None):
        state = _fallback_state(shapeB, getattr(A, "dtype", None), absmax, blocksize, quant_type,
                                absmax_8bit, absmax_code, absmax_offset)
        if (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
                and not isinstance(A, FakeTensor) and A.device == B.device
                and supports_dequantize_nf4(B, state)):
            # Backend fallback does NOT use the public BC transpose shim.
            # F.linear keeps existing bias, activation-layout and GEMM behavior.
            dense = _dense_cuda(B, state)
            return torch.nn.functional.linear(A, dense, bias)
        return original(A, B, shapeB, absmax, blocksize, quant_type, bias,
                        absmax_8bit=absmax_8bit, absmax_code=absmax_code,
                        absmax_offset=absmax_offset)

    setattr(linear_fallback, _PATCH_MARKER, original)
    backend._dequant_linear_fallback = linear_fallback


def patch_dequant():
    """Idempotently install the pinned eager candidate; return patched names.

    Call explicitly before training only after target validation. Returns () if
    bitsandbytes is missing, is not exactly 0.50.2, or Triton is unavailable.
    On CPU only the guarded public wrapper is installed. On CUDA the already
    loaded backend fallback is patched too; custom-GEMM dispatch is untouched.
    Each wrapper keeps its original in __wrapped__ and _ember_dequant_original.
    No torch.library registrations, nn.Module methods or autograd are replaced.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return ()
    if bnb.__version__ != "0.50.2" or triton is None:
        return ()
    _patch_functional(bnb.functional)
    patched = ["bitsandbytes.functional.dequantize_4bit"]
    backend = sys.modules.get("bitsandbytes.backends.cuda.ops")
    if backend is not None and callable(getattr(backend, "_dequant_linear_fallback", None)):
        _patch_cuda_fallback(backend)
        patched.append("bitsandbytes.backends.cuda.ops._dequant_linear_fallback")
    return tuple(patched)
