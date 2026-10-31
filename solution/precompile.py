"""Compile this solution's Triton kernels on a worker thread while Soup loads the model.

The harness starts every run with an empty Triton cache, so the kernel specializations the training
step needs used to be compiled inside step 1: 5.54 s against a 0.42 s median step in the final run
(base: 1.28 s against 0.87 s; docs/measurements/submission_632851f/*/metrics.json, `step_s`).
patch() starts one thread that compiles them with Triton's own `JITFunction.warmup` (compile only:
no launch, no GPU work, no tensors), so the compile overlaps model loading, NF4 quantization, LoRA
setup and dataset preparation instead of the first step. Triton's AsyncCompileMode runs the compiles
on a small thread pool and puts each result into the kernel's in-memory cache, where the real launch
in step 1 finds it.

Training is unchanged: same kernels, same launch arguments, same code path; the main thread still
compiles on a cache miss as before. A specialization below that does not match the real launch
(dtype, constexpr, 16-byte alignment, integer == 1 or % 16) only costs background CPU time; the real
launch then compiles it as before. At exit `PRECOMPILE_STATS {json}` reports the precompiled count and,
per kernel, how many variants the run compiled beyond it ("extra"); all zeros means step 1 compiled
nothing. Not used: autotune, CUDA graphs, caches from earlier runs, extra processes.
"""

import atexit
import json
import os
import sys
import threading
import time

ENABLED = True
WORKERS = 4

# Qwen2.5-1.5B geometry. Triton keys an integer argument only by its type and by whether it is 1 or
# a multiple of 16; constexpr arguments are keyed by value.
HIDDEN, VOCAB, HEADS, KV_HEADS, HEAD_DIM = 1536, 151936, 12, 2, 128

_STATS = {"specs": 0, "compiled": 0, "failed": [], "seconds": None}
_specs_per_kernel = {}
_thread = None


def _specs():
    """(kernel, args, kwargs) for every training launch; torch dtypes stand for tensors."""
    import torch
    import triton

    from . import head_loss, nf4, rmsnorm, rotary, swiglu

    f16, f32, i32, i64, u8 = torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8
    norm = dict(BLOCK=triton.next_power_of_2(HIDDEN), num_warps=rmsnorm.NUM_WARPS)
    head = dict(BLOCK=head_loss.BLOCK, num_warps=8)
    rope = rotary._triton_kernel()

    specs = [
        # 1. Immediate forward kernels needed on Layer 0 first
        (rmsnorm.rmsnorm_fwd_kernel, (f32, f32, f32, f32, HIDDEN, 1e-6), norm),
        (nf4.nf4_dequant_kernel, (u8, f32, f32, f16, HIDDEN * HIDDEN),
         dict(BYTES=nf4.BYTES_PER_PROGRAM, BLOCKSIZE=nf4.BLOCKSIZE, num_warps=nf4.NUM_WARPS)),
    ]
    if rope is not None:
        specs.append((rope, (f32,) * 6 + (1000, 1000, 4, 1000, HEADS, KV_HEADS, HEAD_DIM)
                      + (1000,) * 14 + (False, 256),
                      dict(num_warps=4, enable_fp_fusion=False)))
    n = 2048 * 8960
    specs.append((swiglu.example_swiglu_fwd_kernel, (f16, f16, f32, n), dict(BLOCK=swiglu.BLOCK)))
    specs.append((swiglu.example_swiglu_fwd_kernel, (f32, f32, f32, n), dict(BLOCK=swiglu.BLOCK)))

    # 2. Backward & loss kernels needed later in the step
    specs.append((rmsnorm.rmsnorm_bwd_kernel, (f32, f32, f32, f32, f32, HIDDEN), norm))
    if rope is not None:
        specs.append((rope, (f32,) * 6 + (1000, 1000, 4, 1000, HEADS, KV_HEADS, HEAD_DIM)
                      + (1000,) * 14 + (True, 256),
                      dict(num_warps=4, enable_fp_fusion=False)))
    specs.append((swiglu.example_swiglu_bwd_kernel, (f32, f16, f16, f32, f32, n), dict(BLOCK=swiglu.BLOCK)))
    specs.append((swiglu.example_swiglu_bwd_kernel, (f32, f32, f32, f32, f32, n), dict(BLOCK=swiglu.BLOCK)))
    specs.append((head_loss.head_loss_kernel, (f16, i64, VOCAB, VOCAB, f32, f32, i32, None),
                  dict(WITH_GRAD=False, **head)))
    specs.append((head_loss.head_loss_kernel, (f16, i64, VOCAB, VOCAB, None, None, None, f32),
                  dict(WITH_GRAD=True, **head)))
    return specs


def _kernel_name(kernel):
    return getattr(getattr(kernel, "fn", None), "__name__", repr(kernel))


def _variants(kernel):
    return sum(len(cache[0]) for cache in list(kernel.device_caches.values()))


def compile_all():
    """Compile every spec; never raises (the real launch compiles on a miss anyway)."""
    t0 = time.perf_counter()
    results = []
    try:
        from concurrent.futures import ThreadPoolExecutor

        from triton.runtime._async_compile import AsyncCompileMode

        specs = _specs()
        _STATS["specs"] = len(specs)
        workers = max(1, min(WORKERS, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool, AsyncCompileMode(pool, ignore_errors=True):
            for kernel, args, kwargs in specs:
                name = _kernel_name(kernel)
                try:
                    results.append((name, kernel, kernel.warmup(*args, grid=(1,), **kwargs)))
                except Exception as exc:  # noqa: BLE001 - reported below, launch compiles on a miss
                    _STATS["failed"].append(f"{name}: {exc!r}")
        for name, kernel, compiled in results:
            # A FutureKernel whose compile failed keeps kernel None (ignore_errors=True).
            if compiled is None or getattr(compiled, "kernel", True) is None:
                _STATS["failed"].append(f"{name}: compile failed")
            else:
                _STATS["compiled"] += 1
                _specs_per_kernel[kernel] = _specs_per_kernel.get(kernel, 0) + 1
    except Exception as exc:  # noqa: BLE001
        _STATS["failed"].append(repr(exc))
    _STATS["seconds"] = round(time.perf_counter() - t0, 2)
    if _STATS["failed"]:
        print("PRECOMPILE FAILED (kernels compile at first launch instead): " + "; ".join(_STATS["failed"]),
              file=sys.stderr, flush=True)


def stats():
    """Counters of this process; extra = variants compiled beyond the precompiled ones, per kernel."""
    return dict(_STATS, extra={_kernel_name(k): _variants(k) - n for k, n in _specs_per_kernel.items()})


def _report_stats():
    if _thread is not None:
        _thread.join()
        print("PRECOMPILE_STATS " + json.dumps(stats(), separators=(",", ":")), flush=True)


def patch():
    """Idempotent: start the compile thread once, on a CUDA machine only."""
    global _thread
    if _thread is not None:
        return
    import torch

    if not torch.cuda.is_available():
        return
    from . import rotary

    # Build the lazily created rotary kernel here, so both threads share one JITFunction (and cache).
    rotary._triton_kernel()
    # Not a daemon: interpreter exit waits for it instead of killing it inside a compile.
    _thread = threading.Thread(target=compile_all, name="solution-precompile")
    _thread.start()
    atexit.register(_report_stats)
