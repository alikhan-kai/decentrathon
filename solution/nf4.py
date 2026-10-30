"""Experimental, opt-in frozen-NF4 input gradient: ``dY[..., N] @ W[N, K]``.

Nothing in this module installs a patch. Call ``nf4_input_grad`` explicitly only
after correctness and end-to-end performance have been checked on the target
GPU. This is a first-order backward primitive, not an autograd-aware linear layer.

Supported storage is the bitsandbytes 0.50.2 row-major uint8 NF4 format, with
64 weights per scale and optionally one level of 8-bit scale quantization in
groups of 256. The original matrix shape comes exclusively from quant_state.
Scale groups follow the original flattened W, even across a matrix row boundary.

Format/rounding contract and the 16 numerical format constants were checked in:
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
The input-gradient orientation was checked in its autograd/_functions.py.
The Triton algorithms below are original; no external kernel is imported/copied.

The runtime allocates only the output, a contiguous dY copy if needed, and one
fp32 scale vector for nested states. Dense weights exist only in the explicitly
named PyTorch reference helpers. No state or dense weight is persistently cached.

Toolchain caveat: offline SM75 compilation with Triton 3.6.0 emits scalar fp32
FMA for tl.dot, not tensor-core MMA. Its AccelerateMatmul pass explicitly
deprecates MMA below SM80. This is a T4 acceleration blocker to resolve before
integration; successful compilation alone is not a performance result.
"""

from math import prod

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    triton = None
    tl = None


# Binary32 NF4 format values, indexed by the stored unsigned nibble.
NF4_CODEBOOK = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def _check_tensor(value, name, *, dtype, device, numel, ndim=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.layout != torch.strided or not value.is_contiguous():
        raise NotImplementedError(f"{name} must have contiguous strided storage")
    if value.dtype != dtype:
        raise NotImplementedError(f"{name} must have dtype {dtype}; got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}; got {value.device}")
    if value.numel() != numel or (ndim is not None and value.ndim != ndim):
        raise ValueError(f"{name} must contain {numel} elements" + (f" in {ndim} dimensions" if ndim is not None else ""))
    if value.requires_grad:
        raise NotImplementedError(f"{name} must be frozen (requires_grad=False)")


def _validate_quant_state(packed_weight, quant_state):
    """Check metadata only; never synchronize CUDA to inspect tensor contents."""
    shape = getattr(quant_state, "shape", None)
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2:
        raise ValueError("quant_state.shape must be (N, K), the original row-major weight shape")
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise ValueError("quant_state.shape must contain two positive integer dimensions")
    n, k = shape
    if n * k > 2**31 - 1:
        raise NotImplementedError("NF4 weight indexing is limited to 2**31-1 elements")
    if getattr(quant_state, "quant_type", None) != "nf4":
        raise NotImplementedError("only quant_type='nf4' is supported")
    if getattr(quant_state, "blocksize", None) != 64:
        raise NotImplementedError("only quant_state.blocksize=64 is supported")
    if getattr(quant_state, "dtype", None) != torch.float16:
        raise NotImplementedError("only quant_state.dtype=torch.float16 is supported")
    if getattr(quant_state, "packing_format_for_cpu", False):
        raise NotImplementedError("packing_format_for_cpu is not the supported NF4 layout")
    if not isinstance(packed_weight, torch.Tensor):
        raise ValueError("packed_weight must be a tensor")
    if not (packed_weight.ndim == 1 or (packed_weight.ndim == 2 and 1 in packed_weight.shape)):
        raise NotImplementedError("packed_weight must have shape (P,), (P, 1), or (1, P)")
    device = packed_weight.device
    if device.type not in ("cpu", "cuda"):
        raise NotImplementedError("NF4 reference supports CPU or CUDA tensors")
    _check_tensor(packed_weight, "packed_weight", dtype=torch.uint8, device=device, numel=(n * k + 1) // 2)

    state2 = getattr(quant_state, "state2", None)
    nested = getattr(quant_state, "nested", None)
    if not isinstance(nested, bool) or nested != (state2 is not None):
        raise ValueError("quant_state.nested must agree with the presence of state2")
    groups = (n * k + 63) // 64
    _check_tensor(
        getattr(quant_state, "absmax", None), "quant_state.absmax",
        dtype=torch.uint8 if nested else torch.float32, device=device, numel=groups, ndim=1,
    )
    if nested:
        if getattr(state2, "nested", False) or getattr(state2, "state2", None) is not None:
            raise NotImplementedError("only a single level of nested scale quantization is supported")
        if getattr(state2, "blocksize", None) != 256:
            raise NotImplementedError("only quant_state.state2.blocksize=256 is supported")
        if getattr(state2, "dtype", None) != torch.float32:
            raise NotImplementedError("quant_state.state2.dtype must be torch.float32")
        _check_tensor(
            getattr(state2, "absmax", None), "quant_state.state2.absmax",
            dtype=torch.float32, device=device, numel=(groups + 255) // 256, ndim=1,
        )
        _check_tensor(
            getattr(state2, "code", None), "quant_state.state2.code",
            dtype=torch.float32, device=device, numel=256, ndim=1,
        )
        _check_tensor(
            getattr(quant_state, "offset", None), "quant_state.offset",
            dtype=torch.float32, device=device, numel=1,
        )
    elif getattr(quant_state, "offset", None) is not None:
        raise ValueError("nonnested quant_state must not carry an offset")
    # As in bnb's dequantize_4bit, the outer state.code is metadata: NF4's
    # fixed codebook is determined by quant_type, not a customizable mapping.
    return n, k, groups


def _validate_gradient(grad_output, n):
    if not isinstance(grad_output, torch.Tensor):
        raise ValueError("grad_output must be a tensor")
    if grad_output.layout != torch.strided:
        raise NotImplementedError("grad_output must have strided storage")
    if grad_output.dtype != torch.float16:
        raise NotImplementedError("only grad_output.dtype=torch.float16 is supported")
    if grad_output.ndim < 1 or grad_output.shape[-1] != n:
        raise ValueError(f"grad_output last dimension must equal N={n} from quant_state.shape")
    if torch.is_grad_enabled() and grad_output.requires_grad:
        raise NotImplementedError("nf4_input_grad is a first-order primitive; higher-order gradients are unsupported")


def _scales_reference(quant_state):
    if not quant_state.nested:
        return quant_state.absmax
    state2 = quant_state.state2
    group = torch.arange(quant_state.absmax.numel(), device=quant_state.absmax.device) // 256
    # Two operations, with fp32 rounding after multiplication, as in bnb's
    # blockwise scale decode followed by the separate offset addition.
    centered = state2.code[quant_state.absmax.long()] * state2.absmax[group]
    return centered + quant_state.offset.reshape(())


@torch.no_grad()
def dequantize_nf4_reference(packed_weight, quant_state):
    """CPU/CUDA PyTorch format oracle returning fp16 W[N, K]; allocates dense W.

    Accepts a bitsandbytes QuantState or an object with the same fields. Packed
    row/column singleton views have identical meaning; no transpose is inferred.
    This helper is for validation and is never called by the Triton runtime.
    """
    n, k, _ = _validate_quant_state(packed_weight, quant_state)
    position = torch.arange(n * k, device=packed_weight.device)
    packed = packed_weight.reshape(-1)[position // 2].long()
    shift = torch.where(position % 2 == 0, 4, 0)
    index = (packed >> shift) & 15
    values = torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device=packed_weight.device)
    scales = _scales_reference(quant_state)
    return (values[index] * scales[position // 64]).half().reshape(n, k)


def nf4_input_grad_reference(grad_output, packed_weight, quant_state):
    """Dense PyTorch oracle with fp16 decoded weights and fp32 accumulation."""
    n, _, _ = _validate_quant_state(packed_weight, quant_state)
    _validate_gradient(grad_output, n)
    if grad_output.device != packed_weight.device:
        raise ValueError("grad_output and packed_weight must be on the same device")
    with torch.no_grad():
        weight = dequantize_nf4_reference(packed_weight, quant_state)
        return torch.matmul(grad_output.float(), weight.float()).to(grad_output.dtype)


if triton is not None:

    @triton.jit
    def _decode_nested_scales(Encoded, ScaleCode, ScaleAbsmax, Offset, Scales, COUNT: tl.constexpr, TILE: tl.constexpr):
        group = tl.program_id(0) * TILE + tl.arange(0, TILE)
        valid = group < COUNT
        index = tl.load(Encoded + group, valid, other=0).to(tl.int32)
        code = tl.load(ScaleCode + index)
        magnitude = tl.load(ScaleAbsmax + group // 256, valid, other=0)
        centered = code * magnitude
        decoded = centered + tl.load(Offset)
        tl.store(Scales + group, decoded, valid)

    @triton.jit
    def _nf4_input_grad_kernel(
        Grad, Packed, Scales, Out,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
        CODEBOOK: tl.constexpr,
    ):
        # One program owns an output rectangle [tokens, input features]. The
        # reduction traverses original W rows (output features), never repacking
        # W in the reduction orientation or changing its quantization groups.
        feature_tiles = tl.cdiv(K, BK)
        row = (tl.program_id(0) // feature_tiles) * BM + tl.arange(0, BM)
        col = (tl.program_id(0) % feature_tiles) * BK + tl.arange(0, BK)
        lane = tl.arange(0, BN)
        accum = tl.full((BM, BK), 0, tl.float32)
        for start in range(tl.cdiv(N, BN)):
            reduced = start * BN + lane
            grad = tl.load(Grad + row[:, None] * N + reduced[None, :],
                           (row[:, None] < M) & (reduced[None, :] < N), other=0)
            position = reduced[:, None] * K + col[None, :]
            valid_weight = (reduced[:, None] < N) & (col[None, :] < K)
            byte = tl.load(Packed + position // 2, valid_weight, other=0).to(tl.int32)
            shift = tl.where((position & 1) == 0, 4, 0)
            index = (byte >> shift) & 15
            # A fixed selection chain avoids a device allocation/copy for the
            # format codebook. These are format constants, not learned values.
            decoded = tl.full((BN, BK), 0, tl.float32)
            for code in tl.static_range(16):
                decoded = tl.where(index == code, CODEBOOK[code], decoded)
            scale = tl.load(Scales + position // 64, valid_weight, other=0)
            # Match bnb: round each dequantized weight to fp16 BEFORE the GEMM.
            weight = (decoded * scale).to(tl.float16)
            accum = tl.dot(grad, weight, accum)
        tl.store(Out + row[:, None] * K + col[None, :], accum.to(tl.float16),
                 (row[:, None] < M) & (col[None, :] < K))


def nf4_input_grad(grad_output, packed_weight, quant_state):
    """Return fp16 ``dY[..., N] @ W[N, K]`` on NVIDIA SM75+ CUDA GPUs.

    Supports arbitrary leading dimensions, including empty ones, and strided dY
    (copied to contiguous storage). W must be frozen canonical uint8 NF4 with
    blocksize 64, quant_state.dtype=float16, and optional fp32 nested scales with
    state2.blocksize=256. Unsupported states raise before launching a kernel.

    A single static 32-token x 64-input-feature tile reduces 32 output features
    per iteration, with four warps, two stages and fp32 accumulation. No tuning,
    automatic patching, fallback to dense W, or speed claim is provided.
    """
    n, k, groups = _validate_quant_state(packed_weight, quant_state)
    _validate_gradient(grad_output, n)
    if grad_output.device.type != "cuda" or torch.version.hip is not None:
        raise NotImplementedError("nf4_input_grad requires NVIDIA CUDA tensors; use the reference helper on CPU")
    if grad_output.device != packed_weight.device:
        raise ValueError("grad_output and packed_weight must be on the same CUDA device")
    if triton is None:
        raise RuntimeError("nf4_input_grad requires Triton; the reference helpers work without it")
    if torch.cuda.get_device_capability(grad_output.device) < (7, 5):
        raise NotImplementedError("nf4_input_grad requires a Turing (SM75) or newer GPU")
    m = prod(grad_output.shape[:-1])
    if max(m * n, m * k) > 2**31 - 1:
        raise NotImplementedError("NF4 gradient indexing is limited to 2**31-1 elements")
    out = torch.empty((*grad_output.shape[:-1], k), dtype=grad_output.dtype, device=grad_output.device)
    if m == 0:
        return out
    with torch.cuda.device(grad_output.device):
        grad = grad_output.contiguous()
        if quant_state.nested:
            scales = torch.empty(groups, dtype=torch.float32, device=grad_output.device)
            _decode_nested_scales[(triton.cdiv(groups, 256),)](
                quant_state.absmax, quant_state.state2.code, quant_state.state2.absmax,
                quant_state.offset, scales, groups, 256,
                num_warps=4, enable_fp_fusion=False,
            )
        else:
            scales = quant_state.absmax
        _nf4_input_grad_kernel[(triton.cdiv(m, 32) * triton.cdiv(k, 64),)](
            grad, packed_weight, scales, out, m, n, k, 32, 64, 32, NF4_CODEBOOK,
            num_warps=4, num_stages=2,
        )
    return out
