#!/usr/bin/env python3
"""Allocating residual RMSNorm forward+backward benchmark; run only on NVIDIA CUDA.

Inputs/upstreams are preallocated; each timed iteration builds a fresh autograd
graph, allocates both outputs and both input gradients, then releases them.
Seven blocks alternate team/baseline/candidate and its reverse. Event timings
include host launch gaps; synchronized wall timings also include Python overhead.
No CUDA graphs, retained graphs, output buffers, installs, or transformer imports.
JSON goes to stdout (optionally --output); progress goes to stderr. Exit 1 means
correctness or T-independent specialization failed, even if timings were collected.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import triton

B, H, TS, EPS = 4, 1536, (127, 256, 512), 1e-6
NAMES = ("team", "baseline", "candidate")
KERNELS = ("residual_norm_fwd_kernel", "residual_norm_bwd_kernel")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dense(r, a, w, eps):
    # Exact Qwen eager forward; gradients come from autograd, not a manual formula.
    s = r + a
    x = s.to(torch.float32)
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return s, w * (x * inv).to(s.dtype)


def inputs(t):
    shape = (B, t, H)
    r = torch.randn(shape, device="cuda", dtype=torch.float32).requires_grad_()
    a = (torch.randn(shape, device="cuda") / 4).half().requires_grad_()
    w = torch.rand(H, device="cuda", dtype=torch.float32) + .5
    upstream = tuple(torch.randn_like(r) / 32 for _ in range(2))
    return r, a, w, upstream


def step(fn, r, a, w, upstream, eps=EPS):
    outputs = fn(r, a, w, eps)
    grads = torch.autograd.grad(outputs, (r, a), upstream)
    # Returning nothing releases outputs and gradient allocations each iteration.
    del outputs, grads


def values(fn, name, r, a, w, upstream, eps):
    rr, aa = (x.detach().clone().requires_grad_() for x in (r, a))
    out = fn(rr, aa, w, eps)
    if name in ("candidate", "baseline"):
        assert type(out[0].grad_fn).__name__ == "_ResidualNormBackward", "eager fallback"
    assert all(x.dtype == torch.float32 for x in out), "wrong output dtype"
    grad = torch.autograd.grad(out, (rr, aa), upstream)
    assert grad[0].dtype == torch.float32 and grad[1].dtype == torch.float16
    return tuple(x.detach() for x in (*out, *grad))


def check(functions, r, a, w, upstream, scale, case, eps=EPS, cancel=False):
    expected = values(dense, "dense", r, a, w, upstream, eps)
    records = []
    for name, fn in functions.items():
        record = dict(implementation=name, shape=list(r.shape), case=case, scale=scale,
                      passed=True, max_abs_error={}, errors=[])
        try:
            actual = values(fn, name, r, a, w, upstream, eps)
            for i, (label, got, want) in enumerate(zip(("sum", "norm", "dr", "da"), actual, expected)):
                tol = 2e-6 if i < 2 else (2e-3 if got.dtype == torch.float16 else 1e-4)
                x, y = got.float(), want.float()
                if i >= 2:
                    x, y = x / scale, y / scale
                record["max_abs_error"][label] = (x - y).abs().max().item()
                try:
                    assert torch.isfinite(got).all().item() and torch.isfinite(want).all().item(), label
                    torch.testing.assert_close(x, y, atol=tol, rtol=tol, msg=label)
                    if cancel and i >= 2:
                        assert torch.equal(got, torch.zeros_like(got)), f"{label}: cancellation not exact"
                except AssertionError as exc:
                    record["passed"] = False
                    record["errors"].append(str(exc))
        except AssertionError as exc:
            record["passed"] = False
            record["errors"].append(str(exc))
        records.append(record)
    return records


def cache_snapshot(candidate):
    counts, keys = {}, {}
    for name in KERNELS:
        kernel = getattr(candidate, name)
        # Read without creating a device binder for an unlaunched kernel.
        state = kernel.device_caches.get(torch.cuda.current_device())
        cache = {} if state is None else state[0]
        counts[name] = len(cache)
        keys[name] = set(cache)
        assert all(k.asm.get("cubin") for k in cache.values()), "not compiled CUDA kernels"
    return counts, keys


def measure(fn, data, iterations, warmup):
    for _ in range(warmup):
        step(fn, *data)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Materialize events before timing to exclude their lazy initialization.
    start.record()
    end.record()
    end.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(iterations):
        step(fn, *data)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000
    cuda_ms = start.elapsed_time(end)
    return dict(cuda_total_ms=cuda_ms, wall_total_ms=wall_ms,
                cuda_ms_per_step=cuda_ms / iterations, wall_ms_per_step=wall_ms / iterations)


def summarize(blocks):
    result = {"medians": {}, "paired_ratios": {}}
    for metric in ("cuda_ms_per_step", "wall_ms_per_step"):
        result["medians"][metric] = {
            name: statistics.median(b["timings"][name][metric] for b in blocks) for name in NAMES}
        result["paired_ratios"][metric] = {}
        for numerator, denominator in (("team", "candidate"), ("baseline", "candidate"), ("team", "baseline")):
            ratios = [b["timings"][numerator][metric] / b["timings"][denominator][metric] for b in blocks]
            result["paired_ratios"][metric][f"{numerator}/{denominator}"] = {
                "raw": ratios, "median": statistics.median(ratios)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=930)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    if triton.knobs.runtime.interpret or not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("requires NVIDIA CUDA and real Triton JIT; do not run on CPU/interpreter")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    paths = {"team": args.root / "baseline/rmsnorm.py",
             "baseline": args.root / "baseline/residual_norm.py",
             "candidate": args.root / "candidate/residual_norm.py"}
    modules = {name: load(f"bench_{name}", path) for name, path in paths.items()}

    def team(r, a, w, eps):
        s = r + a
        return s, modules["team"].rmsnorm(s, w, eps)

    functions = {"team": team, "baseline": modules["baseline"].fused_residual_norm,
                 "candidate": modules["candidate"].fused_residual_norm}
    report = dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  seed=args.seed, B=B, H=H, T=list(TS), blocks=7, iterations=args.iterations,
                  warmup_per_measurement=args.warmup, residual_dtype="float32", attention_dtype="float16",
                  weight_dtype="float32", weight_requires_grad=False, both_outputs_and_inputs=True,
                  tolerances=dict(outputs=2e-6, dr_unscaled=1e-4, da_unscaled=2e-3),
                  ratio_convention="numerator / denominator; >1 means denominator is faster",
                  timing_scope="allocating forward + autograd.grad; preallocated inputs/upstreams; no graph replay",
                  source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                  specialization=[], checks=[], benchmarks=[])

    # Isolate T variation before correctness changes epsilon, shape, or strides.
    initial, _ = cache_snapshot(modules["candidate"])
    report["initial_candidate_cache_counts"] = initial
    previous = None
    independent = True
    for t in TS:
        data = inputs(t)
        values(functions["candidate"], "candidate", *data, EPS)
        torch.cuda.synchronize()
        counts, keys = cache_snapshot(modules["candidate"])
        same = previous is None or keys == previous
        independent &= same and all(counts.values())
        report["specialization"].append(dict(T=t, counts=counts, same_keys_as_first_T=same))
        if previous is None:
            previous = keys
        del data
    report["candidate_cache_independent_of_T"] = bool(independent)

    for t in TS:
        print(f"T={t}: correctness and 7 alternating timing blocks", file=sys.stderr, flush=True)
        data = inputs(t)
        r, a, w, upstream = data
        strided = (torch.randn(B, t, H, 2, device="cuda").mul_(1 / 32)[..., 0],
                   (torch.randn((), device="cuda") / 32).expand_as(r))
        for scale in (1, 65536):
            scaled_dense = tuple(g * scale for g in upstream)
            # Multiplication of a view can materialize it: recreate strided/broadcast upstreams explicitly.
            gs = torch.empty(B, t, H, 2, device="cuda")
            gs[..., 0].copy_(strided[0] * scale)
            gy = (strided[1][0, 0, 0] * scale).expand_as(r)
            for case, gradients in (("contiguous", scaled_dense), ("strided_broadcast", (gs[..., 0], gy))):
                report["checks"].extend(check(functions, r, a, w, gradients, scale, case))
        blocks = []
        for index in range(7):
            order = NAMES if index % 2 == 0 else NAMES[::-1]
            timings = {name: measure(functions[name], data, args.iterations, args.warmup) for name in order}
            block = dict(block=index + 1, order=list(order), timings=timings)
            blocks.append(block)
            print(json.dumps(dict(T=t, **block)), file=sys.stderr, flush=True)
        report["benchmarks"].append(dict(T=t, raw_blocks=blocks, **summarize(blocks)))
        del data, r, a, w, upstream, strided, scaled_dense, gradients, gs, gy

    for hidden in (17, H):
        r = torch.zeros(2, hidden, device="cuda", requires_grad=True)
        a = torch.zeros_like(r, dtype=torch.float16, requires_grad=True)
        w = torch.ones(hidden, device="cuda")
        upstream = (torch.full_like(r, -65536), torch.full_like(r, 65536))
        report["checks"].extend(check(functions, r, a, w, upstream, 65536, "exact_cancellation", eps=1., cancel=True))
    report["passed"] = report["candidate_cache_independent_of_T"] and all(x["passed"] for x in report["checks"])
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
