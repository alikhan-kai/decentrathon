"""Isolated SM75 fp16 MMA proof. Nothing here patches the training path.

This is original glue/kernel code for NVIDIA's established warp MMA instruction,
not a new multiplication algorithm. Register fragments and instruction semantics:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma

For physical lane l, g = l >> 2 and t = l & 3. A's two packed registers
contain A[g, 2t:2t+2] and A[g+8, 2t:2t+2]; B's packed register contains
B[2t:2t+2, g]. The four fp32 results are C[g,2t], C[g,2t+1],
C[g+8,2t], C[g+8,2t+1]. Lower halfwords hold the first element of each pair.

Contract: rank-two strided fp16/fp32 inputs, rounded separately to fp16;
fp32 accumulation, optional final fp16 rounding. No bias, alpha/beta, batching,
autograd, or automatic CPU fallback. Only a small, explicit experiment envelope
is accepted. Triton imports lazily; ``matmul_reference`` is the CPU oracle.

Triton does not promise a physical lane mapping for inline_asm_elementwise.
Therefore this proof is restricted to Triton 3.6.0 and SM75/SM120, and its
compiler evidence must be inspected with ``compile_probe`` and checked on the
target with the GPU tests before use. Offline compilation is NOT GPU evidence.
This deliberately small kernel has no shared-memory reuse or performance claim.
"""

from functools import lru_cache
import re

import torch


MMA_INSTRUCTION = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
MAX_SHAPE = (256, 4096, 4096)  # M, N, K: bounded proof, not the full vocabulary.


def _validate(a, b, output_dtype):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("a and b must be tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be two-dimensional")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b inner dimensions must agree")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    if any(x.dtype not in (torch.float16, torch.float32) for x in (a, b)):
        raise NotImplementedError("operands must be float16 or float32")
    if output_dtype not in (torch.float16, torch.float32):
        raise NotImplementedError("output_dtype must be float16 or float32")
    if any(x.layout != torch.strided for x in (a, b)):
        raise NotImplementedError("operands must have strided storage")
    m, k = a.shape
    n = b.shape[1]
    if min(m, n, k) <= 0:
        raise ValueError("M, N, K must be positive")
    if any(size > limit for size, limit in zip((m, n, k), MAX_SHAPE)):
        raise NotImplementedError(f"proof shape limit is M,N,K <= {MAX_SHAPE}")
    for x in (a, b):
        if any(s < 0 for s in x.stride()) or sum((d - 1) * s for d, s in zip(x.shape, x.stride())) >= 2**31:
            raise NotImplementedError("proof requires nonnegative strides and signed-32-bit element offsets")
    return m, n, k


def matmul_reference(a, b, *, output_dtype=torch.float32):
    """CPU oracle: half-round each input, multiply in fp64, round the output.

    FP64 makes this an independent error oracle, not an emulation of the MMA
    reduction order. Use a numerical bound for nonexact cases on the GPU.
    """
    _validate(a, b, output_dtype)
    if a.device.type != "cpu":
        raise NotImplementedError("matmul_reference is an explicitly CPU-only oracle")
    return (a.to(torch.float16).double() @ b.to(torch.float16).double()).to(output_dtype)


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    if triton.__version__ != "3.6.0":
        raise NotImplementedError("this mapping proof requires Triton 3.6.0; re-audit other compilers")

    @triton.jit
    def _turing_mma_kernel(
        A, B, C, LaneMap,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
        CHECK_LANES: tl.constexpr,
    ):
        # The audited TTGIR layout is [1] element/thread, [32] threads/warp,
        # [1] warp/CTA, order [0]. No default tl.dot layout is involved.
        lane = tl.arange(0, 32)
        group = lane >> 2
        pair = 2 * (lane & 3)
        row = tl.program_id(0) * 16 + group
        col = tl.program_id(1) * 8 + group
        c0 = tl.full((32,), 0, tl.float32)
        c1 = tl.full((32,), 0, tl.float32)
        c2 = tl.full((32,), 0, tl.float32)
        c3 = tl.full((32,), 0, tl.float32)
        for block_k in range(tl.cdiv(K, 8)):
            kk = block_k * 8 + pair
            # All 32 lanes execute MMA, including edge tiles. Only loads and
            # final stores are masked. Invalid operands are exact fp16 zero.
            a00 = tl.load(A + row * AM + kk * AK, (row < M) & (kk < K), 0).to(tl.float16)
            a01 = tl.load(A + row * AM + (kk + 1) * AK, (row < M) & (kk + 1 < K), 0).to(tl.float16)
            a10 = tl.load(A + (row + 8) * AM + kk * AK, (row + 8 < M) & (kk < K), 0).to(tl.float16)
            a11 = tl.load(A + (row + 8) * AM + (kk + 1) * AK, (row + 8 < M) & (kk + 1 < K), 0).to(tl.float16)
            b0 = tl.load(B + kk * BK + col * BN, (kk < K) & (col < N), 0).to(tl.float16)
            b1 = tl.load(B + (kk + 1) * BK + col * BN, (kk + 1 < K) & (col < N), 0).to(tl.float16)
            # Explicit uint32 packing prevents implicit subword-operand
            # bundling by inline_asm_elementwise from changing fragment order.
            a0 = a00.to(tl.uint16, bitcast=True).to(tl.uint32) | (a01.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
            a1 = a10.to(tl.uint16, bitcast=True).to(tl.uint32) | (a11.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
            bp = b0.to(tl.uint16, bitcast=True).to(tl.uint32) | (b1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
            c0, c1, c2, c3 = tl.inline_asm_elementwise(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};",
                constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",
                args=[a0, a1, bp, c0, c1, c2, c3],
                dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
                # The instruction is a warp collective, not a speculatable
                # lane-local expression. Keep it side-effecting in LLVM.
                is_pure=False,
                pack=1,
            )
        out_col = tl.program_id(1) * 8 + pair
        tl.store(C + row * N + out_col, c0, (row < M) & (out_col < N))
        tl.store(C + row * N + out_col + 1, c1, (row < M) & (out_col + 1 < N))
        tl.store(C + (row + 8) * N + out_col, c2, (row + 8 < M) & (out_col < N))
        tl.store(C + (row + 8) * N + out_col + 1, c3, (row + 8 < M) & (out_col + 1 < N))
        if CHECK_LANES:
            physical_lane = tl.inline_asm_elementwise(
                "mov.u32 $0, %laneid;", constraints="=r,r", args=[lane],
                dtype=tl.int32, is_pure=False, pack=1,
            )
            # One CTA owns the diagnostic output, so there is no write race.
            tl.store(LaneMap + lane, physical_lane,
                     (tl.program_id(0) == 0) & (tl.program_id(1) == 0))

    return _turing_mma_kernel


def turing_matmul(a, b, *, output_dtype=torch.float32, return_lane_map=False):
    """Launch the bounded explicit experiment; never called by solution.apply.

    ``return_lane_map=True`` additionally returns a 32-element logical-index to
    physical-lane map from this same kernel, for target correctness tests.
    """
    m, n, k = _validate(a, b, output_dtype)
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        raise NotImplementedError("this isolated primitive does not implement autograd")
    if a.device.type != "cuda":
        raise NotImplementedError("turing_matmul requires NVIDIA CUDA; use matmul_reference on CPU")
    if torch.cuda.get_device_capability(a.device) not in ((7, 5), (12, 0)):
        raise NotImplementedError("this isolated proof has compiler targets SM75 and SM120 only")
    output = torch.empty((m, n), device=a.device, dtype=output_dtype)
    lanes = torch.empty((32,), device=a.device, dtype=torch.int32) if return_lane_map else output
    with torch.cuda.device(a.device):
        kernel = _kernel()
        # Runtime pointer/stride specialization can differ from an offline
        # compile. Inspect that actual variant before it is ever launched.
        compiled = kernel.warmup(
            a, b, output, lanes, m, n, k, *a.stride(), *b.stride(), return_lane_map,
            grid=((m + 15) // 16, (n + 7) // 8), num_warps=1, num_stages=1,
        )
        _require_audited_layout(compiled)
        kernel[((m + 15) // 16, (n + 7) // 8)](
            a, b, output, lanes, m, n, k, *a.stride(), *b.stride(), return_lane_map,
            num_warps=1, num_stages=1,
        )
    return (output, lanes) if return_lane_map else output


def _inspect_compiled(compiled):
    ptx = compiled.asm["ptx"]
    ttgir = compiled.asm["ttgir"]
    layout = "#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>"
    definitions = re.findall(r"^(#\w+) = (#ttg\..+)$", ttgir, flags=re.MULTILINE)
    one_per_lane = len(definitions) == 1 and definitions[0][1] == layout
    mma_ir = [line for line in ttgir.splitlines() if MMA_INSTRUCTION in line]
    # Follow *every* input and result type at the actual MMA operation. Merely
    # finding a suitable layout elsewhere in the module would be insufficient.
    operand_layout = False
    if one_per_lane and mma_ir:
        alias = definitions[0][0]
        operand_layout = all(
            len(re.findall(r"tensor<32x(?:i32|f32), " + re.escape(alias) + r">", line)) == 11
            and "packed_element = 1 : i32" in line
            for line in mma_ir
        )
    mma_ptx = [line.strip() for line in ptx.splitlines() if "mma.sync" in line]
    return {
        "sm": compiled.metadata.target.arch,
        "num_warps": compiled.metadata.num_warps,
        "shared_bytes": compiled.metadata.shared,
        "mma_count": ptx.count(MMA_INSTRUCTION),
        "scalar_fma_count": len(re.findall(r"\bfma(?:\.[a-z0-9]+)*\.f32\b", ptx)),
        "one_element_per_lane": one_per_lane and operand_layout and "ttg.convert_layout" not in ttgir,
        "unpredicated_mma": bool(mma_ptx) and all(line.startswith(MMA_INSTRUCTION) for line in mma_ptx),
        "one_warp_launch": bool(re.search(r"\.reqntid\s+32(?:\s|$)", ptx)),
        "ptx": ptx, "ttgir": ttgir,
    }


@lru_cache(maxsize=128)
def _require_audited_layout(compiled):
    report = _inspect_compiled(compiled)
    if not (report["sm"] in (75, 120) and report["num_warps"] == 1
            and report["one_element_per_lane"] and report["one_warp_launch"]
            and report["unpredicated_mma"] and report["scalar_fma_count"] == 0):
        raise RuntimeError("Turing MMA physical lane mapping is unaudited; refusing to launch this compiler variant")


def compile_probe(*, sm=75, shape=(16, 8, 8), input_dtype="fp16", output_dtype="fp32", transposed_b=False, check_lanes=False):
    """Compile without a CUDA device and return inspectable compiler evidence.

    ``shape`` is (M,N,K). Raw PTX and TTGIR are retained in the result, alongside
    conservative checks for the exact one-dimensional physical layout used by
    every tensor. This does not establish numerical correctness or performance.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if sm not in (75, 120):
        raise ValueError("compile targets are SM75 or SM120")
    if input_dtype not in ("fp16", "fp32") or output_dtype not in ("fp16", "fp32"):
        raise ValueError("compiler operand/output types must be fp16 or fp32")
    if len(shape) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
        raise ValueError("shape must be positive integers (M,N,K)")
    m, n, k = shape
    if any(size > limit for size, limit in zip(shape, MAX_SHAPE)):
        raise ValueError(f"proof shape limit is M,N,K <= {MAX_SHAPE}")
    fn = _kernel()
    signature = {"A": f"*{input_dtype}", "B": f"*{input_dtype}", "C": f"*{output_dtype}", "LaneMap": "*i32"}
    constants = dict(M=m, N=n, K=k, AM=k, AK=1, BK=1 if transposed_b else n,
                     BN=k if transposed_b else 1, CHECK_LANES=check_lanes)
    compiled = triton.compile(ASTSource(fn, signature, constexprs=constants),
                              target=GPUTarget("cuda", sm, 32),
                              options={"num_warps": 1, "num_stages": 1})
    report = _inspect_compiled(compiled)
    report.update(shape=shape, triton=triton.__version__, input_dtype=input_dtype,
                  output_dtype=output_dtype, transposed_b=transposed_b, check_lanes=check_lanes)
    _require_audited_layout(compiled)
    return report
