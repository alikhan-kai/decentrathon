#!/usr/bin/env python3
"""Bounded CUDA comparison. --help imports no torch and performs no CUDA work.

Run only while owning the GPU; wrap with the hard timeout in CUDA_RUNNER.md.
JSONL output is flushed after every case/sample. Fractions mean SUPERVISED rows.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
H, V, CHUNK = 1536, 151936, 512
FRACTIONS = (0.0, 0.01, 0.25, 1.0)
STATE = {"validation_passed": 0, "validation_failed": 0, "benchmark_samples": 0}


def emit(kind, **fields):
    print(json.dumps({"event": kind, **fields}, allow_nan=False), flush=True)


class BudgetExpired(Exception):
    pass


class Budget:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.limit = seconds
        self.done = threading.Event()
        threading.Thread(target=self.watchdog, daemon=True).start()

    def elapsed(self):
        return time.monotonic() - self.start

    def check(self, reserve=3.0):
        if self.elapsed() + reserve >= self.limit:
            raise BudgetExpired("internal wall-clock budget exhausted")

    def watchdog(self):
        if not self.done.wait(max(0, self.limit - self.elapsed())):
            # No CUDA calls here: this must still terminate a blocked synchronize.
            payload = {"event": "summary", "status": "partial_timeout",
                       "reason": "internal hard watchdog", "elapsed_s": self.elapsed(),
                       **STATE}
            os.write(1, (json.dumps(payload) + "\n").encode())
            os._exit(124)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(torch, n, fraction, seed, device):
    # Host RNG chooses scattered indices; mask creation is outside timed regions.
    cpu = torch.Generator().manual_seed(seed)
    count = round(n * fraction)
    order = torch.randperm(n, generator=cpu)
    selected = order[:count].clone()
    # Include the chunk-tail row without making the rest of the mask contiguous.
    if count and not bool((selected == n - 1).any()):
        selected[-1] = n - 1
    target = torch.full((n,), -100, dtype=torch.long)
    if count:
        target[selected] = torch.randint(V, (count,), generator=cpu)
        target[selected[-1]] = V - 1
        if count > 2:
            target[selected[1]] = 0
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    hidden = torch.randn((n, H), generator=gen, device=device, dtype=torch.float32)
    return hidden, target.to(device), selected.tolist(), count


def run_validation_path(torch, module, base, weight, target, selected, denom, amp, budget):
    """One graph at a time, one real fp32 parameter, real unscale/step/update."""
    parameter = torch.nn.Parameter(base.clone())
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=65536.0, enabled=amp)
    n = base.shape[0]
    rows, entropy, hits = [], [], []
    total_loss = torch.zeros((), device=base.device)
    if module is None:
        # Dense over the COMPLETE vocabulary, bounded to CHUNK rows. Global
        # normalization is identical in every tile; never retain two CE graphs.
        for start in range(0, n, CHUNK):
            budget.check()
            stop = min(start + CHUNK, n)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = (parameter[start:stop].half() @ weight.t()).float()
                # A deliberate correct prediction makes accuracy checks nonvacuous.
                witness = selected[0] if selected else -1
                if start <= witness < stop:
                    target[witness] = logits[witness - start].detach().argmax()
                row = torch.nn.functional.cross_entropy(
                    logits, target[start:stop], ignore_index=-100, reduction="none")
                loss = row.sum() / denom
            total_loss += loss.detach()
            rows.append(row.detach().cpu())
            scaler.scale(loss).backward()
            with torch.no_grad():
                # Keep metric temporaries bounded to 64 rows, after CE backward.
                for offset in range(0, stop - start, 64):
                    x = logits.detach()[offset:offset + 64]
                    logp = x.log_softmax(-1)
                    entropy.append(-(logp.exp() * logp).sum(-1).cpu())
                    t = target[start + offset:min(start + offset + 64, stop)]
                    hits.append(((x.argmax(-1) == t) & t.ne(-100)).cpu())
                    del x, logp, t
            del logits, row, loss
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, row, ent, hit = module.fused_head_loss(
                parameter, weight, target, denom, chunk=CHUNK)
        if any(x.requires_grad for x in (row, ent, hit)):
            raise AssertionError("metrics unexpectedly require gradients")
        total_loss = loss.detach()
        scaler.scale(loss).backward()
        rows, entropy, hits = [row.detach().cpu()], [ent.detach().cpu()], [hit.bool().cpu()]
        del loss, row, ent, hit
    if parameter.grad is None or parameter.grad.dtype != torch.float32:
        raise AssertionError("expected real fp32 optimizer parameter gradient")
    scaled_grad = parameter.grad.detach().cpu().clone()
    if not bool(torch.isfinite(scaled_grad).all()):
        raise AssertionError("nonfinite scaled gradient")
    scaler.unscale_(optimizer)
    grad = parameter.grad.detach().cpu().clone()
    before = parameter.detach().clone()
    expected = before.add(parameter.grad, alpha=-0.1)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(parameter, expected, atol=1e-7, rtol=1e-6)
    if scaler.get_scale() != scale_before or (amp and scale_before != 65536.0):
        raise AssertionError("GradScaler changed scale / skipped finite-gradient step")
    result = {"loss": total_loss.detach().cpu(), "row": torch.cat(rows),
              "entropy": torch.cat(entropy), "hit": torch.cat(hits),
              "grad": grad, "scaled_grad": scaled_grad, "scale": scale_before}
    del parameter, optimizer, scaler, before, expected, total_loss
    return result


def compare(torch, actual, reference, target):
    ignored = target.cpu().eq(-100)
    for key in ("row", "entropy"):
        if not bool(torch.isfinite(actual[key]).all()):
            raise AssertionError(f"nonfinite {key}")
        torch.testing.assert_close(actual[key], reference[key], atol=2e-4, rtol=2e-4)
    if bool(ignored.all()):
        if not bool(torch.isnan(actual["loss"])):
            raise AssertionError("all-ignored denom=0 must preserve NaN loss")
    else:
        torch.testing.assert_close(actual["loss"], reference["loss"], atol=2e-4, rtol=2e-4)
        if int(reference["hit"].sum()) == 0:
            raise AssertionError("accuracy witness did not produce a reference hit")
    if not torch.equal(actual["hit"], reference["hit"]):
        raise AssertionError("argmax-hit metrics differ")
    if int(torch.count_nonzero(actual["row"][ignored])) or bool(actual["hit"][ignored].any()):
        raise AssertionError("ignored row loss/hit must be exactly zero")
    errors = {}
    for key in ("grad", "scaled_grad"):
        a, r = actual[key], reference[key]
        if not bool(torch.isfinite(a).all()):
            raise AssertionError(f"nonfinite {key}")
        if int(torch.count_nonzero(a[ignored])):
            raise AssertionError(f"ignored {key} must be exactly zero")
        magnitude = float(r.abs().max())
        atol = magnitude * 0.002  # No large absolute floor masking tiny gradients.
        torch.testing.assert_close(a, r, atol=atol, rtol=0.02)
        delta = a - r
        errors[key] = {"max_abs_error": float(delta.abs().max()), "atol": atol,
                       "rtol": 0.02, "reference_max_abs": magnitude,
                       "relative_l2": float(delta.norm() / r.norm()) if magnitude else 0.0}
    return {"gradient_errors": errors, "hits": int(actual["hit"].sum()),
            "mean_entropy": float(actual["entropy"].mean()),
            "loss": float(actual["loss"]) if bool(torch.isfinite(actual["loss"])) else "nan"}


def validate(torch, modules, weight, device, budget):
    # Interleave sizes and scaling so early partial output covers both shapes.
    for fraction in FRACTIONS:
        for n in (513, 1024):
            for amp in (False, True):
                budget.check()
                fields = {"n": n, "supervised_fraction": fraction, "scale": 65536 if amp else 1}
                emit("validation_begin", **fields)
                base, target, selected, count = inputs(torch, n, fraction, 4100 + n, device)
                denom = torch.tensor(count, device=device, dtype=torch.float32)
                reference = run_validation_path(torch, None, base, weight, target, selected, denom, amp, budget)
                for name, module in modules.items():
                    budget.check()
                    try:
                        actual = run_validation_path(torch, module, base, weight, target, selected, denom, amp, budget)
                        details = compare(torch, actual, reference, target)
                        STATE["validation_passed"] += 1
                        emit("validation", implementation=name, status="passed", supervised_count=count,
                             **fields, **details)
                        del actual
                    except AssertionError as exc:
                        STATE["validation_failed"] += 1
                        emit("validation", implementation=name, status="failed", **fields, error=str(exc)[:3000])
                del reference, base, target, denom
                gc.collect()
    # Positive external divisor is separately checked for the all-ignored mask.
    base, target, selected, _ = inputs(torch, 513, 0.0, 6913, device)
    for name, module in modules.items():
        budget.check()
        actual = run_validation_path(torch, module, base, weight, target, selected,
                                     torch.tensor(7.0, device=device), True, budget)
        if float(actual["loss"]) != 0 or int(torch.count_nonzero(actual["grad"])):
            raise AssertionError(f"{name}: all-ignored positive-divisor result is not zero")
        emit("ignored_positive_denom", implementation=name, status="passed", n=513, scale=65536)


def benchmark_sample(torch, module, base, weight, target, denom, mode):
    hidden = base.detach().clone().requires_grad_(True)
    # Warmed event creation/recording is outside the wall interval.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    output = None
    if mode == "backward":
        output = module.fused_head_loss(hidden, weight, target, denom, chunk=CHUNK)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    start.record()
    if mode == "full":
        output = module.fused_head_loss(hidden, weight, target, denom, chunk=CHUNK)
    output[0].backward()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    result = {"wall_ms": wall_ms, "cuda_event_ms": start.elapsed_time(end),
              "allocated_before_bytes": allocated_before,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_extra_allocated_bytes": max(0, torch.cuda.max_memory_allocated() - allocated_before),
              "reserved_bytes": torch.cuda.memory_reserved()}
    del hidden, output, start, end
    return result


def benchmark(torch, modules, weight, device, budget, sizes=(1024,)):
    for n in sizes:
        budget.check()
        if n == 2048:
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            # Chunked head needs <1 GiB temporary; allow a generous 2 GiB margin.
            required = 2 * 1024 ** 3 + n * H * 4 * 4
            if free < required:
                emit("benchmark_skipped", n=n, reason="optional shape does not fit conservative free-memory bound",
                     free_bytes=free, required_free_bytes=required)
                continue
        for fraction in FRACTIONS:
            base, target, _, count = inputs(torch, n, fraction, 8100 + n, device)
            # Positive external divisor for empty labels keeps timing loss finite.
            denom = torch.tensor(max(count, 1), device=device, dtype=torch.float32)
            for mode in ("full", "backward"):
                common = {"n": n, "mode": mode, "supervised_fraction": fraction, "supervised_count": count}
                emit("benchmark_begin", **common)
                for name, module in modules.items():
                    budget.check()
                    benchmark_sample(torch, module, base, weight, target, denom, mode)
                    emit("warmup_complete", implementation=name, **common)
                samples = {name: [] for name in modules}
                for iteration in range(3):
                    order = list(modules)
                    if iteration % 2:
                        order.reverse()
                    for name in order:
                        budget.check()
                        sample = benchmark_sample(torch, modules[name], base, weight, target, denom, mode)
                        samples[name].append(sample)
                        STATE["benchmark_samples"] += 1
                        emit("benchmark_sample", implementation=name, iteration=iteration, order=order,
                             **common, **sample)
                medians = {name: {key: statistics.median(s[key] for s in values)
                                  for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_extra_allocated_bytes")}
                           for name, values in samples.items()}
                emit("benchmark_summary", **common, medians=medians,
                     wall_speedup_baseline_over_candidate=medians["baseline"]["wall_ms"] / medians["candidate"]["wall_ms"],
                     cuda_speedup_baseline_over_candidate=medians["baseline"]["cuda_event_ms"] / max(medians["candidate"]["cuda_event_ms"], 1e-9))
            del base, target, denom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "head_loss.correct-r6.py")
    parser.add_argument("--candidate", type=Path, default=ROOT / "head_loss.py")
    parser.add_argument("--phase", choices=("all", "validate", "benchmark"), default="all")
    parser.add_argument("--include-n2048", action="store_true", help="append optional N2048 benchmark if memory/time permit")
    parser.add_argument("--budget-seconds", type=float, default=110.0, help="entire process work budget, capped at 110 seconds")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index within CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if not 5 <= args.budget_seconds <= 110:
        parser.error("--budget-seconds must be in [5, 110]")
    budget = Budget(args.budget_seconds)
    status, reason, code = "complete", None, 0
    try:
        os.environ["TRITON_INTERPRET"] = "0"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        import torch
        budget.check()
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this runner has no CPU fallback")
        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        modules = {"baseline": load_module(args.baseline.resolve(), "rotary_runner_baseline"),
                   "candidate": load_module(args.candidate.resolve(), "rotary_runner_candidate")}
        for name, module in modules.items():
            if module.CHUNK != CHUNK:
                raise RuntimeError(f"{name}: unexpected CHUNK={module.CHUNK}")
        emit("configuration", h=H, vocab=V, chunk=CHUNK, benchmark_n=1024, validation_n=[513, 1024],
             supervised_fractions=FRACTIONS, warmups=1, measured_iterations=3,
             phase=args.phase, optional_n2048=args.include_n2048, weight_std=0.01, weight_dtype="float32",
             budget_seconds=args.budget_seconds, torch_version=torch.__version__,
             cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(device),
             baseline=str(args.baseline.resolve()), candidate=str(args.candidate.resolve()),
             baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
             candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
             expected_validation_comparisons=32 if args.phase != "benchmark" else 0,
             expected_required_benchmark_samples=48 if args.phase != "validate" else 0)
        budget.check()
        generator = torch.Generator(device=device).manual_seed(11927)
        weight = torch.randn((V, H), generator=generator, device=device, dtype=torch.float32).mul_(0.01)
        # Benchmarks first: dense-oracle allocation/caching cannot bias head peaks.
        if args.phase in ("all", "benchmark"):
            benchmark(torch, modules, weight, device, budget)
        if args.phase in ("all", "validate"):
            validate(torch, modules, weight, device, budget)
        if args.include_n2048 and args.phase != "validate":
            # Only optional work follows required validation and N1024 timings.
            benchmark(torch, modules, weight, device, budget, sizes=(2048,))
        if STATE["validation_failed"]:
            status, code = "failed", 1
    except BudgetExpired as exc:
        status, reason, code = "partial_timeout", str(exc), 124
    except Exception as exc:
        status, reason, code = "failed", f"{type(exc).__name__}: {exc}", 1
    emit("summary", status=status, reason=reason, elapsed_s=budget.elapsed(), phase=args.phase,
         **STATE, expected_validation_comparisons=32 if args.phase != "benchmark" else 0,
         expected_required_benchmark_samples=48 if args.phase != "validate" else 0)
    budget.done.set()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
