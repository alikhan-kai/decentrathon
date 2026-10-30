"""Bounded, synthetic CUDA diagnostics; these are not official challenge scores.

Run from the repository root: python -m solution.benchmark --help
No patches are installed, no model/data are downloaded, and NF4 is opt-in.
"""

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F


def _arguments():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run alone on the GPU. Finish this process before starting the official harness.",
    )
    parser.add_argument("--component", choices=("head", "rmsnorm", "nf4"), default="head",
                        help="isolated operation; NF4 is experimental and opt-in")
    parser.add_argument("--m", type=int, default=128, help="total token rows, not sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="head only; sequence length is M / batch")
    parser.add_argument("--hidden-size", type=int, default=1536, help="head/RMSNorm width")
    parser.add_argument("--vocab-size", type=int, default=151936, help="head vocabulary size")
    parser.add_argument("--chunk-size", type=int, default=128, help="head projection chunk in token rows")
    parser.add_argument("--input-dtype", choices=("fp32", "fp16"), default="fp32",
                        help="head/RMSNorm activation dtype; frozen weights stay fp32, NF4 uses fp16")
    parser.add_argument("--nf4-shape", choices=("1536x1536", "256x1536", "8960x1536", "1536x8960"),
                        default="1536x1536", help="NF4 weight [N,K]; dY[M,N] @ W[N,K] -> dX[M,K]")
    parser.add_argument("--warmup", type=int, default=2, help="untimed invocations per implementation (1..20)")
    parser.add_argument("--iterations", type=int, default=5, help="timed invocations per implementation (1..100)")
    parser.add_argument("--device", type=int, default=0, help="logical CUDA device index")
    parser.add_argument("--seed", type=int, default=17, help="synthetic input seed")
    parser.add_argument("--json", type=Path, metavar="PATH", help="also save the successful JSON report here")
    args = parser.parse_args()
    for name in ("m", "batch_size", "hidden_size", "vocab_size", "chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.device < 0 or not 1 <= args.warmup <= 20 or not 1 <= args.iterations <= 100:
        parser.error("device must be nonnegative; warmup must be 1..20 and iterations 1..100")
    if args.component == "head" and (args.m % args.batch_size or args.m // args.batch_size < 4):
        parser.error("head requires M divisible by batch-size and at least 4 tokens per sequence")
    if args.component == "rmsnorm" and args.hidden_size > 8192:
        parser.error("RMSNorm width must be <= 8192 to exercise the original CUDA kernel")
    return args


def _close(actual, expected, *, rtol, atol):
    # assert_close rejects NaNs; reject matching infinities as well.
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError("non-finite value in candidate or oracle")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {"max_abs_error": float((actual.float() - expected.float()).abs().max()),
            "rtol": rtol, "atol": atol}


def _dense_head(hidden, weight, labels, denominator):
    """Independent PyTorch graph, including the fp16 logits -> fp32 CE boundary."""
    logits = F.linear(hidden, weight).float()
    targets = F.pad(labels[:, 1:], (0, 1), value=-100)
    token_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(labels)
    loss = token_loss.sum() / denominator
    with torch.no_grad():
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        predictions = logits.argmax(-1)
    return {"loss": loss, "token_loss": token_loss, "token_entropy": entropy, "predictions": predictions}


def _head_metrics(outputs, labels, attention_mask):
    # Accuracy and Rewind use the shifted supervised population. Entropy and
    # num_tokens also include prompts and the last attention-valid token.
    target = labels[:, 1:]
    valid = target != -100
    counts = valid.sum(-1)
    row_sum = outputs["token_loss"][:, :-1].sum(-1)
    correct = ((outputs["predictions"][:, :-1] == target) & valid).sum()
    return {
        "entropy": (outputs["token_entropy"] * attention_mask).sum() / attention_mask.sum(),
        "accuracy": correct.float() / counts.sum().clamp_min(1),
        "attention_tokens": attention_mask.sum(),
        "supervised_tokens": counts.sum(),
        "row_supervised_tokens": counts,
        "row_loss_sum": row_sum,
        "row_mean_loss": row_sum / counts.clamp_min(1),
    }


def _head(args, device):
    from .head import frozen_head_loss

    dtype = torch.float32 if args.input_dtype == "fp32" else torch.float16
    batch, length = args.batch_size, args.m // args.batch_size
    hidden = torch.randn(batch, length, args.hidden_size, device=device, dtype=dtype).requires_grad_()
    weight = torch.randn(args.vocab_size, args.hidden_size, device=device) * 0.02
    labels = torch.randint(args.vocab_size, (batch, length), device=device)
    positions = torch.arange(length, device=device)
    lengths = length - torch.arange(batch, device=device) % max(1, length // 4)
    attention_mask = positions[None, :] < lengths[:, None]
    labels.masked_fill_(~attention_mask, -100)
    labels[:, :max(1, length // 4)] = -100
    if batch > 1:
        labels[0] = -100  # Prompt-only row still contributes entropy.
    supervised = (labels[:, 1:] != -100).sum()
    # A nontrivial supplied denominator catches accidental mean-of-means or
    # ignoring num_items_in_batch. The default denominator is checked separately.
    denominator = supervised + 7

    def candidate(x, w, y, denom):
        result = frozen_head_loss(x, w, y, num_items_in_batch=denom, chunk_size=args.chunk_size)
        return {name: getattr(result, name) for name in ("loss", "token_loss", "token_entropy", "predictions")}

    def run(implementation, scale=65536.0):
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = implementation(hidden, weight, labels, denominator)
        dx, = torch.autograd.grad(outputs["loss"], hidden, outputs["loss"].new_tensor(scale))
        return {key: value.detach() for key, value in outputs.items()}, dx

    checks = {}
    metrics = None
    for scale in (1.0, 65536.0):
        actual, dx = run(candidate, scale)
        expected, reference_dx = run(_dense_head, scale)
        for name in actual:
            exact = name == "predictions"
            checks[f"scale_{scale:g}/{name}"] = _close(
                actual[name], expected[name], rtol=0 if exact else 1e-3, atol=0 if exact else 2e-3)
        checks[f"scale_{scale:g}/unscaled_input_gradient"] = _close(
            dx.float() / scale, reference_dx.float() / scale, rtol=1e-2, atol=2e-6)
        actual_metrics = _head_metrics(actual, labels, attention_mask)
        reference_metrics = _head_metrics(expected, labels, attention_mask)
        for name in actual_metrics:
            exact = name in ("accuracy", "attention_tokens", "supervised_tokens", "row_supervised_tokens")
            checks[f"scale_{scale:g}/metrics/{name}"] = _close(
                actual_metrics[name], reference_metrics[name], rtol=0 if exact else 1e-3, atol=0 if exact else 2e-3)
        metrics = {name: value.tolist() for name, value in actual_metrics.items()}
        del actual, expected, dx, reference_dx, actual_metrics, reference_metrics
    with torch.autocast("cuda", dtype=torch.float16):
        default = frozen_head_loss(hidden, weight, labels, chunk_size=args.chunk_size)
        expected = _dense_head(hidden, weight, labels, supervised)
    checks["default_denominator_loss"] = _close(default.loss, expected["loss"], rtol=1e-3, atol=2e-3)
    del default, expected
    return {
        "shape": {"hidden": list(hidden.shape), "weight": list(weight.shape), "chunk_size": args.chunk_size},
        "input_dtype": str(dtype), "weight_dtype": "torch.float32", "projection_dtype": "torch.float16",
        "timed_work": "forward loss + all-position statistics + input backward; upstream scale 65536",
        "denominator": int(denominator), "metrics": metrics, "parity": checks,
    }, {"original_head": lambda: run(candidate), "pytorch_dense_head": lambda: run(_dense_head)}


def _rmsnorm(args, device):
    from .rmsnorm import rms_norm

    dtype = torch.float32 if args.input_dtype == "fp32" else torch.float16
    x = torch.randn(args.m, args.hidden_size, device=device, dtype=dtype).requires_grad_()
    weight = torch.randn(args.hidden_size, device=device, dtype=torch.float32)
    upstream = torch.randn(args.m, args.hidden_size, device=device, dtype=torch.float32)
    eps = 1e-6

    def reference(value, w, epsilon):
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + epsilon)
        return w * normalized.to(value.dtype)

    def run(implementation):
        with torch.autocast("cuda", dtype=torch.float16):
            output = implementation(x, weight, eps)
        dx, = torch.autograd.grad(output, x, upstream)
        return output.detach(), dx

    actual, dx = run(rms_norm)
    expected, reference_dx = run(reference)
    tolerance = {"rtol": 2e-3, "atol": 2e-3} if dtype == torch.float16 else {"rtol": 2e-5, "atol": 2e-6}
    checks = {"output": _close(actual, expected, **tolerance),
              "input_gradient": _close(dx, reference_dx, **tolerance)}
    return {
        "shape": list(x.shape), "input_dtype": str(dtype), "weight_dtype": "torch.float32", "eps": eps,
        "timed_work": "forward + input backward (frozen fp32 weight)", "parity": checks,
    }, {"original_rmsnorm": lambda: run(rms_norm), "pytorch_rmsnorm": lambda: run(reference)}


def _nf4(args, device):
    import bitsandbytes.functional as bnb
    from .nf4 import nf4_input_grad

    n, k = map(int, args.nf4_shape.split("x"))
    weight = torch.randn(n, k, device=device, dtype=torch.float16) / 8
    packed, state = bnb.quantize_4bit(weight, blocksize=64, quant_type="nf4", compress_statistics=True)
    del weight
    dy = torch.randn(args.m, n, device=device, dtype=torch.float16) / 8

    def candidate():
        with torch.autocast("cuda", dtype=torch.float16):
            return nf4_input_grad(dy, packed, state)

    def reference():
        # Recreate the dense temporary every call, as the bnb backward does.
        with torch.autocast("cuda", dtype=torch.float16):
            return torch.matmul(dy, bnb.dequantize_4bit(packed, state))

    actual, baseline = candidate(), reference()
    with torch.autocast("cuda", enabled=False):
        oracle = (dy.float() @ bnb.dequantize_4bit(packed, state).float()).half()
    checks = {
        "bnb_dequantize_fp16_matmul": _close(actual, baseline, rtol=2e-3, atol=2e-3),
        "bnb_dequantize_pytorch_fp32_oracle": _close(actual, oracle, rtol=2e-3, atol=2e-3),
    }
    return {
        "experimental": True, "shape": {"dy": list(dy.shape), "weight": [n, k], "dx": [args.m, k]},
        "input_dtype": "torch.float16", "nested_quantization": True, "blocksize": 64,
        "timed_work": "input gradient only, including scale decoding; baseline dequantizes each call",
        "parity": checks,
    }, {"original_nf4_input_grad": candidate, "bnb_dequantize_then_matmul": reference}


def _measure(run, args, device):
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    for _ in range(args.warmup):
        result = run()
        del result
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(args.iterations):
        start.record()
        result = run()
        end.record()
        end.synchronize()  # Bound outstanding work and surface asynchronous errors.
        samples.append(start.elapsed_time(end))
        del result
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "cuda_event_ms": {"median": statistics.median(samples), "mean": statistics.mean(samples),
                          "min": min(samples), "max": max(samples), "samples": samples},
        "allocated_before_timing_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_increment_over_inputs_bytes": peak - baseline,
    }


def _environment(device):
    properties = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ("torch", "triton", "soup-cli", "transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        inventory = query.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        inventory = None
    return {
        "python": sys.version.split()[0], "packages": packages, "torch_cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "gpu_total_memory_bytes": properties.total_memory, "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "nvidia_smi_inventory_index_uuid_name_driver": inventory,
        "autocast_dtype": "torch.float16", "allow_tf32": False,
        "allow_fp16_reduced_precision_reduction": False,
    }


def main():
    args = _arguments()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        print(f"error: NVIDIA CUDA is required; torch={torch.__version__}, torch CUDA={torch.version.cuda}. "
              "No parity checks or measurements ran. --help works on CPU.", file=sys.stderr)
        return 2
    if args.device >= torch.cuda.device_count():
        print(f"error: CUDA device {args.device} is not visible", file=sys.stderr)
        return 2
    try:
        import triton  # Fail before any eager fallback could be timed as an original kernel.

        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        environment = _environment(device)
        print(f"Diagnostic only: {environment['gpu_name']}, {args.component}, M={args.m}; "
              "explicit fp16 autocast. Keep the official harness stopped.", file=sys.stderr, flush=True)
        details, implementations = {"head": _head, "rmsnorm": _rmsnorm, "nf4": _nf4}[args.component](args, device)
        print("Parity passed; running bounded warmup and CUDA-event measurements.", file=sys.stderr, flush=True)
        timings = {}
        for name, run in implementations.items():
            print(f"Measuring {name}: {args.warmup} warmup + {args.iterations} samples", file=sys.stderr, flush=True)
            timings[name] = _measure(run, args, device)
        report = {
            "diagnostic_only": True, "official_score": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment, "component": args.component,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "measurement_notes": [
                "Synthetic isolated operations, not an end-to-end Soup training run.",
                "CUDA-event intervals after parity/warmup exclude cold compilation; they are not official wall time.",
                "Allocator peaks include shared inputs; incremental peaks subtract allocated inputs after warmup.",
                "Allocated bytes exclude reserved memory, CUDA context and other processes; not nvidia-smi scored memory.",
                "Precision settings and synthetic data are diagnostic; official harness/config remain unchanged.",
            ],
            "details": details, "measurements": timings,
        }
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(encoded)
        print(encoded, end="")
    except (AssertionError, RuntimeError, ImportError, ValueError, OSError) as exc:
        print(f"error: {args.component} diagnostic failed ({type(exc).__name__}): {exc}\n"
              "No successful report was written. Fix the failure before using timings.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
