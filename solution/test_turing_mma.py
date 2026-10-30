"""Isolated Turing MMA checks; this file never installs a solution patch.

Local: python -m pytest solution/test_turing_mma.py -q
Offline compiler: python -m solution.test_turing_mma --compile-only
Required GPU proof: python -m solution.test_turing_mma --gpu
Bounded timings: python -m solution.test_turing_mma --gpu --benchmark
The CLI fails, rather than skips, when the requested GPU is unavailable.
--benchmark exits 2 if the candidate exceeds 1.25x torch.mm latency; this is
only an exploration bound, not an integration or official scoring threshold.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import pytest
import torch


def _module():
    return importlib.import_module("solution.turing_mma")


def test_reference_rounds_operands_to_half_before_multiplication():
    a = torch.tensor([[1.0003, 2.0006], [0.33333, -0.66667]])
    b = torch.tensor([[1.0003, -2.0006], [0.33333, 0.66667]])
    expected = a.half().double() @ b.half().double()
    result = _module().matmul_reference(a, b)
    torch.testing.assert_close(result, expected.float(), rtol=0, atol=0)
    assert not torch.equal(result, a @ b)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_reference_shape_dtype_and_strides(dtype):
    a = torch.arange(30, dtype=torch.float32).reshape(5, 6)[:, ::2]
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3).t()
    result = _module().matmul_reference(a, b, output_dtype=dtype)
    assert result.shape == (5, 4) and result.dtype == dtype
    torch.testing.assert_close(result, (a.double() @ b.double()).to(dtype), rtol=0, atol=0)


def test_gpu_primitive_never_falls_back_to_cpu():
    with pytest.raises(NotImplementedError, match="CUDA"):
        _module().turing_matmul(torch.ones(16, 8), torch.ones(8, 8))


@pytest.mark.parametrize("a,b,message", [
    (torch.ones(3), torch.ones(3, 2), "two-dimensional"),
    (torch.ones(2, 3), torch.ones(4, 2), "inner dimensions"),
    (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(3, 2), "float16 or float32"),
    (torch.ones(0, 3), torch.ones(3, 2), "positive"),
])
def test_input_validation(a, b, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        _module().turing_matmul(a, b)


def test_primitive_rejects_autograd():
    with pytest.raises(NotImplementedError, match="autograd"):
        _module().turing_matmul(torch.ones(16, 8, requires_grad=True), torch.ones(8, 8))


def test_proof_size_and_output_type_are_bounded():
    with pytest.raises(NotImplementedError, match="shape limit"):
        _module().turing_matmul(torch.ones(257, 1), torch.ones(1, 1))
    with pytest.raises(NotImplementedError, match="output_dtype"):
        _module().turing_matmul(torch.ones(1, 1), torch.ones(1, 1), output_dtype=torch.bfloat16)


def test_reference_does_not_import_triton(monkeypatch):
    # A fresh import must work with Triton unavailable. Already imported Triton
    # modules in this shared pytest process are deliberately hidden for this test.
    for name in tuple(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "triton", None)
    module = importlib.reload(_module())
    torch.testing.assert_close(module.matmul_reference(torch.ones(1, 1), torch.ones(1, 1)), torch.ones(1, 1))


def test_offline_sm75_contains_mma_with_one_element_per_physical_lane():
    pytest.importorskip("triton")
    report = _module().compile_probe(sm=75)
    assert report["mma_count"] == 1
    assert report["scalar_fma_count"] == 0
    assert report["one_element_per_lane"]
    assert report["num_warps"] == 1
    assert report["shared_bytes"] == 0


@pytest.mark.parametrize("change", ["layout", "mma_operand", "predicated", "launch"])
def test_compiler_gate_rejects_unsafe_variants(change):
    pytest.importorskip("triton")
    from types import SimpleNamespace

    module = _module()
    report = module.compile_probe()
    class Compiled:
        pass
    compiled = Compiled()
    compiled.metadata = SimpleNamespace(target=SimpleNamespace(arch=75), num_warps=1, shared=0)
    compiled.asm = {key: report[key] for key in ("ptx", "ttgir")}
    if change == "layout":
        compiled.asm["ttgir"] = compiled.asm["ttgir"].replace("sizePerThread = [1]", "sizePerThread = [2]")
    elif change == "mma_operand":
        lines = compiled.asm["ttgir"].splitlines()
        compiled.asm["ttgir"] = "\n".join(
            line.replace("tensor<32xi32, #blocked>", "tensor<32xi32, #wrong>", 1)
            if module.MMA_INSTRUCTION in line else line for line in lines)
    elif change == "predicated":
        compiled.asm["ptx"] = compiled.asm["ptx"].replace(module.MMA_INSTRUCTION, "@%p1 " + module.MMA_INSTRUCTION)
    else:
        compiled.asm["ptx"] = compiled.asm["ptx"].replace(".reqntid 32", ".reqntid 64")
    with pytest.raises(RuntimeError, match="refusing to launch"):
        module._require_audited_layout(compiled)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a real NVIDIA GPU; CPU checks are not MMA evidence")
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("this proof targets SM75; use the explicit CLI for another supported target")


def _basis_proof():
    module = _module()
    # Each of the 128 A positions and all eight B output columns is isolated.
    # This catches transposition, half order, accumulator order, and lane swaps.
    for row in range(16):
        for k in range(8):
            a = torch.zeros((16, 8), device="cuda", dtype=torch.float16)
            b = torch.arange(1, 65, device="cuda", dtype=torch.float16).reshape(8, 8)
            a[row, k] = 1
            expected = torch.zeros((16, 8), device="cuda", dtype=torch.float32)
            expected[row] = b[k].float()
            for instrument in (False, True):
                result = module.turing_matmul(a, b, return_lane_map=instrument)
                if instrument:
                    actual, lanes = result
                    torch.testing.assert_close(lanes.cpu(), torch.arange(32, dtype=torch.int32), rtol=0, atol=0)
                else:
                    actual = result
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Conversely isolate every B element with a row-distinct signed A.
    a = ((torch.arange(128).reshape(16, 8) % 31) - 15).half().cuda()
    for k in range(8):
        for col in range(8):
            b = torch.zeros((8, 8), device="cuda", dtype=torch.float16)
            b[k, col] = -1
            expected = torch.zeros((16, 8), device="cuda", dtype=torch.float32)
            expected[:, col] = -a[:, k].float()
            torch.testing.assert_close(module.turing_matmul(a, b), expected, rtol=0, atol=0)
    return {"a_basis_cases": 128, "a_instrumentation_variants": 2, "b_basis_cases": 64, "exact": True}


def test_gpu_tiny_basis_and_physical_lanes():
    _require_gpu()
    _basis_proof()


def _numerical_report(actual, a, b):
    """Conservative engineering gate, not an ISA rounding-error guarantee.

    NVIDIA leaves MMA accumulation order and subnormal handling unspecified.
    Test finite, normal-range values and retain exact structural tests as well.
    FP16 output gets one additional half-ULP final-rounding allowance.
    """
    aa, bb = a.half().double(), b.half().double()
    expected = aa @ bb
    magnitude = aa.abs() @ bb.abs()
    nround = 9 * math.ceil(a.shape[1] / 8)
    u = 2**-23
    gamma = nround * u / (1 - nround * u)
    bound = 4 * gamma * magnitude
    if actual.dtype == torch.float16:
        bound = bound + expected.abs() * 2**-11 + 2**-25
    result = actual.detach().cpu().double()
    error = (result - expected).abs()
    if not torch.isfinite(result).all():
        raise AssertionError("MMA produced unexpected nonfinite output")
    zero = magnitude == 0
    if zero.any() and not torch.equal(result[zero], expected[zero]):
        raise AssertionError("MMA produced a nonzero result for a zero dot product")
    normalized = torch.where(bound > 0, error / bound, torch.zeros_like(error))
    worst = float(normalized.max())
    report = {"max_abs_error": float(error.max()), "max_error_over_bound": worst,
              "relative_l2_error": float(torch.linalg.vector_norm(result - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30)),
              "bound": "4*gamma(9*ceil(K/8))*sum(abs(A16*B16)), u=2^-23; optional fp16 rounding"}
    if (error > bound).any():
        raise AssertionError(f"MMA numerical gate failed: {report}")
    return report


def _random_case(shape, transposed_b=False, input_dtype=torch.float32, output_dtype=torch.float32, seed=719, scale=1.0):
    m, n, k = shape
    generator = torch.Generator().manual_seed(seed)
    a = (torch.randn((m, k), generator=generator) * scale).to(input_dtype)
    b = (torch.randn((n, k) if transposed_b else (k, n), generator=generator) * scale).to(input_dtype)
    if transposed_b:
        b = b.t()
    actual = _module().turing_matmul(a.cuda(), b.cuda(), output_dtype=output_dtype)
    return _numerical_report(actual, a, b)


@pytest.mark.parametrize("shape", [(16, 8, 8), (1, 1, 1), (15, 7, 7), (17, 9, 9), (31, 17, 19), (32, 24, 64)])
@pytest.mark.parametrize("transposed_b", [False, True])
def test_gpu_random_tails(shape, transposed_b):
    _require_gpu()
    _random_case(shape, transposed_b)


def _exact_tail_proof():
    # Final-K-only input catches missing partial chunks. Distinct signed integer
    # chunks catch accumulator resets/casts and remain exactly representable.
    reductions = list(range(1, 18)) + [24, 31, 32, 33, 63, 64, 65]
    for k in reductions:
        m, n = 17, 9
        a = torch.zeros((m, k))
        b = torch.zeros((k, n))
        a[:, -1] = torch.arange(1, m + 1)
        b[-1, :] = -torch.arange(1, n + 1)
        expected = a @ b
        torch.testing.assert_close(_module().turing_matmul(a.cuda(), b.cuda()).cpu(), expected, rtol=0, atol=0)
        a = ((torch.arange(m * k).reshape(m, k) % 13) - 6).half()
        b = ((torch.arange(k * n).reshape(k, n) % 17) - 8).half()
        expected = a.float() @ b.float()
        torch.testing.assert_close(_module().turing_matmul(a.cuda(), b.cuda()).cpu(), expected, rtol=0, atol=0)
    # Values lost by fp16 intermediate accumulation: 1 + 2^-12 and
    # 2048 + 1 - 2048. Contributions live in different MMA iterations.
    for contributions, value in (((1.0, 2**-12), 1.0 + 2**-12), ((2048.0, 1.0, -2048.0), 1.0)):
        k = 8 * len(contributions)
        a = torch.zeros((17, k), dtype=torch.float16)
        b = torch.zeros((k, 9), dtype=torch.float16)
        for chunk, contribution in enumerate(contributions):
            a[:, 8 * chunk] = 1
            b[8 * chunk, :] = contribution
        actual = _module().turing_matmul(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(actual, torch.full((17, 9), value), rtol=0, atol=0)
    return {"reduction_sizes": reductions, "cases": len(reductions) * 2 + 2, "exact": True,
            "fp32_accumulator_cases": ["1 + 2^-12", "2048 + 1 - 2048"]}


def test_gpu_exact_k_tails_and_repeated_accumulation():
    _require_gpu()
    _exact_tail_proof()


def _prepared_launch(a, b, output):
    """Test-only preallocated launch for canaries and equal-allocation timing."""
    m, k = a.shape
    n = b.shape[1]
    args = (a, b, output, output, m, n, k, *a.stride(), *b.stride(), False)
    grid = ((m + 15) // 16, (n + 7) // 8)
    kernel = _module()._kernel()
    compiled = kernel.warmup(*args, grid=grid, num_warps=1, num_stages=1)
    _module()._require_audited_layout(compiled)
    return lambda: kernel[grid](*args, num_warps=1, num_stages=1), compiled


def _strided_canary_proof():
    m, n, k = 17, 9, 19
    # Dirty padding and a nonzero offset make invalid loads visible. CUDA .to()
    # can compact a view, so create the backing storage on the device first.
    a_storage = torch.full((m + 2, 2 * k + 5), 123, dtype=torch.float16, device="cuda")
    b_storage = torch.full((n + 2, 2 * k + 5), -117, dtype=torch.float16, device="cuda")
    a = a_storage[1:m + 1, 2:2 * k + 2:2]
    b = b_storage[1:n + 1, 3:2 * k + 3:2].t()
    a.copy_(((torch.arange(m * k).reshape(m, k) % 9) - 4).half())
    b.copy_(((torch.arange(k * n).reshape(k, n) % 7) - 3).half())
    a_before, b_before = a_storage.clone(), b_storage.clone()
    guard = 37
    storage = torch.full((m * n + 2 * guard,), -12345.0, device="cuda")
    output = storage[guard:-guard].reshape(m, n)
    launch, _ = _prepared_launch(a, b, output)
    launch()
    expected = a.cpu().float() @ b.cpu().float()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.all(storage[:guard] == -12345.0) and torch.all(storage[-guard:] == -12345.0)
    assert torch.equal(a_storage, a_before) and torch.equal(b_storage, b_before)
    return {"strides_a": a.stride(), "strides_b": b.stride(), "output_canaries": "intact", "inputs": "unchanged"}


def test_gpu_strided_inputs_storage_offsets_and_output_canaries():
    _require_gpu()
    _strided_canary_proof()


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [0.125, 1.0, 4.0])
def test_gpu_dtypes_scales_and_longer_reduction(input_dtype, output_dtype, scale):
    _require_gpu()
    _random_case((31, 17, 257), True, input_dtype, output_dtype, seed=113, scale=scale)


@pytest.mark.parametrize("shape,dt,out,trans,lane", [
    ((1, 1, 1), "fp16", "fp16", False, True),
    ((17, 9, 9), "fp32", "fp16", True, True),
    ((31, 17, 19), "fp16", "fp32", False, False),
    ((128, 256, 1536), "fp16", "fp16", True, False),
])
def test_offline_sm75_tails_and_loop(shape, dt, out, trans, lane):
    pytest.importorskip("triton")
    report = _module().compile_probe(shape=shape, input_dtype=dt, output_dtype=out, transposed_b=trans, check_lanes=lane)
    assert report["one_element_per_lane"] and report["unpredicated_mma"]
    assert report["mma_count"] >= 1 and report["scalar_fma_count"] == 0


def _compile_matrix(sm):
    reports = []
    for shape, dt, out, trans, lane in [
        ((16, 8, 8), "fp16", "fp32", False, False),
        ((16, 8, 8), "fp16", "fp32", False, True),
        ((1, 1, 1), "fp16", "fp16", False, True),
        ((17, 9, 9), "fp32", "fp16", True, True),
        ((31, 17, 19), "fp16", "fp32", False, False),
        ((128, 256, 1536), "fp16", "fp16", True, False),
    ]:
        result = _module().compile_probe(sm=sm, shape=shape, input_dtype=dt, output_dtype=out, transposed_b=trans, check_lanes=lane)
        reports.append({key: value for key, value in result.items() if key not in ("ptx", "ttgir")})
    return reports


def _benchmark_cases():
    """Two bounded smoke comparisons, not an official harness measurement.

    Same fp16 tensors, strides, outputs and warmed CUDA-event timing for both
    implementations. Allocation and validation are excluded. No autotuning.
    """
    warmup, repeats, iterations = 5, 7, 10
    max_probe_ms, budget_seconds, max_ratio = 100.0, 30.0, 1.25
    started = time.monotonic()
    records = []
    generator = torch.Generator(device="cuda").manual_seed(719)
    for m, n, k in ((128, 256, 1536), (128, 1536, 1536)):
        a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
        b = torch.randn((n, k), device="cuda", dtype=torch.float16, generator=generator).t()
        ours = torch.empty((m, n), device="cuda", dtype=torch.float16)
        baseline = torch.empty_like(ours)
        candidate, compiled = _prepared_launch(a, b, ours)
        reference = lambda: torch.mm(a, b, out=baseline)

        def measure(fn, count):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(count):
                fn()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) / count

        reference()
        candidate()
        error = _numerical_report(ours, a.cpu(), b.cpu())
        baseline_error = _numerical_report(baseline, a.cpu(), b.cpu())
        probe = measure(candidate, 1)
        entry = {"shape_mnk": [m, n, k], "strides_a": a.stride(), "strides_b": b.stride(),
                 "dtype": "fp16 inputs/output; candidate fp32 accumulation; torch.mm ambient settings", "seed": 719,
                 "warmup_calls": warmup, "repeats": repeats, "calls_per_repeat": iterations,
                 "probe_ms": probe, "error": error, "baseline_error": baseline_error,
                 "candidate_n_regs": getattr(compiled, "n_regs", None),
                 "candidate_n_spills": getattr(compiled, "n_spills", None),
                 "max_candidate_over_torch_for_further_work": max_ratio,
                 "torch_ms": [], "candidate_ms": []}
        if probe > max_probe_ms or time.monotonic() - started > budget_seconds:
            entry.update(warmup_calls=0, repeats_completed=0)
            entry["status"] = "STOP: probe exceeds 100 ms or total benchmark budget exceeds 30 s"
            records.append(entry)
            break
        for _ in range(warmup):
            reference()
            candidate()
        torch.cuda.synchronize()
        for repeat in range(repeats):
            if time.monotonic() - started > budget_seconds:
                entry["status"] = "STOP: total benchmark budget exceeds 30 s"
                break
            # Alternate the order to reduce systematic clock/cache bias.
            order = [("torch_ms", reference), ("candidate_ms", candidate)]
            if repeat % 2:
                order.reverse()
            for label, fn in order:
                entry[label].append(measure(fn, iterations))
        if len(entry["torch_ms"]) == repeats:
            ratio = statistics.median(entry["candidate_ms"]) / statistics.median(entry["torch_ms"])
            entry["candidate_over_torch"] = ratio
            entry["status"] = "within exploration bound" if ratio <= max_ratio else "STOP: exceeds exploration bound"
        entry["repeats_completed"] = len(entry["torch_ms"])
        records.append(entry)
        if entry["status"].startswith("STOP"):
            break
    passed = len(records) == 2 and all(x["status"] == "within exploration bound" for x in records)
    return {"conditions": "preallocated outputs, hot inputs, CUDA events, no allocation/validation or training",
            "torch_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "torch_allow_fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
            "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "max_probe_ms": max_probe_ms, "timing_budget_seconds": budget_seconds,
            "exploration_bound_passed": passed, "cases": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    parser.add_argument("--sm", type=int, choices=(75, 120), default=75)
    parser.add_argument("--benchmark", action="store_true", help="after correctness, run two bounded cuBLAS comparisons")
    args = parser.parse_args(argv)
    if args.benchmark and not args.gpu:
        parser.error("--benchmark requires --gpu")
    result = {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "target_sm": args.sm, "official_harness_evidence": False,
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("turing_mma.py", "test_turing_mma.py")}}
    try:
        if args.gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU proof requested, but NVIDIA CUDA is unavailable; no GPU test was run")
            capability = torch.cuda.get_device_capability()
            if capability != (args.sm // 10, args.sm % 10):
                raise RuntimeError(f"requested SM{args.sm}, found {capability}; refusing a different target")
            result["device"] = torch.cuda.get_device_name()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            result["device_memory_bytes"] = props.total_memory
            result["multiprocessors"] = props.multi_processor_count
        result["compiler"] = _compile_matrix(args.sm)
        if args.gpu:
            torch.manual_seed(719)
            result["basis"] = _basis_proof()
            result["exact_tails"] = _exact_tail_proof()
            result["strided_canaries"] = _strided_canary_proof()
            numerical = []
            for shape in ((16, 8, 8), (1, 1, 1), (15, 7, 7), (17, 9, 9), (31, 17, 19), (32, 24, 64)):
                for trans in (False, True):
                    numerical.append(dict(shape=shape, transposed_b=trans, **_random_case(shape, trans)))
            for input_dtype in (torch.float16, torch.float32):
                for output_dtype in (torch.float16, torch.float32):
                    for scale in (0.125, 1.0, 4.0):
                        numerical.append(dict(shape=[31, 17, 257], input_dtype=str(input_dtype), output_dtype=str(output_dtype),
                                              scale=scale, **_random_case((31, 17, 257), True, input_dtype, output_dtype, 113, scale)))
            result["numerical"] = numerical
            if args.benchmark:
                result["benchmark"] = _benchmark_cases()
        result["status"] = "targeted GPU checks passed" if args.gpu else "offline compiler checks passed; GPU unverified"
        if args.benchmark and not result["benchmark"]["exploration_bound_passed"]:
            result["status"] += "; performance exploration bound failed"
            print(json.dumps(result, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(result, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
