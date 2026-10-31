"""Compile this solution's Triton kernels on a worker thread while Soup loads the model.

The harness starts every run with an empty Triton cache, so the kernel specializations the training
step needs used to be compiled inside step 1. patch() starts one worker thread right after apply()'s
Python imports finish so compilation overlaps C++/Rust/CUDA model loading, NF4 quantization, LoRA
setup and dataset preparation without GIL contention during Python imports.

Only the exact 9 specializations launched during training are compiled, in order of first use in
step 1 (forward then backward), with per-kernel readiness events so early forward launches never
wait on backward kernel compilation. Not used: autotune, CUDA graphs, caches from earlier runs,
extra processes.
"""

import atexit
import json
import sys
import threading
import time

ENABLED = True

# Qwen2.5-1.5B geometry. Triton keys an integer argument only by its type and by whether it is 1 or
# a multiple of 16; constexpr arguments are keyed by value.
HIDDEN, VOCAB, HEADS, KV_HEADS, HEAD_DIM = 1536, 151936, 12, 2, 128

_STATS = {"specs": 0, "compiled": 0, "failed": [], "seconds": None}
_specs_per_kernel = {}
_events = {}
_thread = None


def _specs():
    """(kernel, args, kwargs) for every training launch, in exact order of first use in step 1."""
    import torch
    import triton

    from . import dropout_pack, head_loss, nf4, rmsnorm, rotary, swiglu

    f16, f32, i32, i64, u8 = torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8
    norm = dict(BLOCK=triton.next_power_of_2(HIDDEN), num_warps=rmsnorm.NUM_WARPS)
    head = dict(BLOCK=head_loss.BLOCK, num_warps=8)
    rope = rotary._triton_kernel()
    rope_args = (f16,) * 6 + (1000, 1000, 4, 1000, HEADS, KV_HEADS, HEAD_DIM) + (1000,) * 14
    rope_kwargs = dict(num_warps=4, enable_fp_fusion=False)
    n = 2048 * 8960
    fwd, bwd = swiglu.example_swiglu_fwd_kernel, swiglu.example_swiglu_bwd_kernel
    specs = [
        (rmsnorm.rmsnorm_fwd_kernel, (f32, f32, f32, f32, HIDDEN, 1e-6), norm),
        (nf4.nf4_dequant_kernel, (u8, u8, f32, f32, f32, f32, f16, HIDDEN * HIDDEN),
         dict(BYTES=nf4.BYTES_PER_PROGRAM, BLOCKSIZE=nf4.BLOCKSIZE, BLOCKSIZE2=nf4.BLOCKSIZE2,
              NESTED=True, num_warps=nf4.NUM_WARPS)),
        (dropout_pack.pack_dropout_mask, (torch.bool, u8, n, 128), {}),
        (rope, rope_args + (False, 256), rope_kwargs),
        (fwd, (f16, f16, f16, n), dict(BLOCK=swiglu.BLOCK)),
        (head_loss.head_loss_kernel, (f16, i64, VOCAB, VOCAB, f32, f32, i32, None),
         dict(WITH_GRAD=False, **head)),
        (head_loss.head_loss_kernel, (f16, i64, VOCAB, VOCAB, None, None, None, f32),
         dict(WITH_GRAD=True, **head)),
        (bwd, (f16, f16, f16, f16, f16, n), dict(BLOCK=swiglu.BLOCK)),
        (rope, rope_args + (True, 256), rope_kwargs),
        (rmsnorm.rmsnorm_bwd_kernel, (f32, f32, f32, f32, f32, HIDDEN), norm),
        (dropout_pack.packed_dropout_backward, (f32, u8, f32, n, 1.0 / 0.95, 512),
         dict(enable_fp_fusion=False)),
    ]
    return [spec for spec in specs if spec[0] is not None]


def _kernel_name(kernel):
    return getattr(getattr(kernel, "fn", None), "__name__", repr(kernel))


def _variants(kernel):
    return sum(len(cache[0]) for cache in list(kernel.device_caches.values()))


def compile_all(specs=None):
    """Compile every spec in order; never raises (the real launch compiles on a miss anyway)."""
    t0 = time.perf_counter()
    remaining = {}
    try:
        specs = _specs() if specs is None else specs
        _STATS["specs"] = len(specs)
        for kernel, _, _ in specs:
            remaining[id(kernel)] = remaining.get(id(kernel), 0) + 1
        for kernel, args, kwargs in specs:
            try:
                kernel.warmup(*args, grid=(1,), **kwargs)
            except Exception as exc:  # noqa: BLE001 - reported below, launch compiles on a miss
                _STATS["failed"].append(f"{_kernel_name(kernel)}: {exc!r}")
            else:
                _STATS["compiled"] += 1
                _specs_per_kernel[kernel] = _specs_per_kernel.get(kernel, 0) + 1
            finally:
                kid = id(kernel)
                remaining[kid] -= 1
                if remaining[kid] <= 0 and kid in _events:
                    _events[kid].set()
    except Exception as exc:  # noqa: BLE001
        _STATS["failed"].append(repr(exc))
    finally:
        for ev in _events.values():
            ev.set()
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
    """Start one worker thread; idempotent. Does nothing if CUDA is unavailable."""
    global _thread
    if _thread is not None:
        return
    import torch

    if not torch.cuda.is_available():
        return
    from . import rotary

    # Build the lazily created rotary kernel here, so both threads share one JITFunction (and cache).
    rotary._triton_kernel()
    specs = _specs()
    for kernel in {id(k): k for k, _, _ in specs}.values():
        ev = threading.Event()
        _events[id(kernel)] = ev

        def _make_hook(event=ev):
            def _wait_kernel(*args, **kwargs):
                if not event.is_set() and threading.current_thread() is not _thread:
                    event.wait()
            return _wait_kernel

        kernel.add_pre_run_hook(_make_hook())
    # Not a daemon: interpreter exit waits for it instead of killing it inside a compile.
    _thread = threading.Thread(target=compile_all, args=(specs,), name="solution-precompile")
    _thread.start()
    atexit.register(_report_stats)
