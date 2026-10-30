"""Per-operation / per-module GPU profile of the SOLUTION run, on the same 16 active steps as the
harness profile (copy of the logic of harness/_train.py, which is not edited).

    python tools/op_profile.py --config runs_try/solution_1/task.yaml --out out/op_profile.json
    python tools/op_profile.py --selftest        # tiny CPU model, checks the attribution code

What it writes (JSON + a text report next to it):
  ops_by_name / ops_by_shape  top ATen ops by self GPU time (record_shapes=True)
  modules                     GPU ms per step attributed to the source module, forward and backward
                              separately (exclusive: an op counts for the innermost module range)
  dtypes, calls_per_step      dtype of module outputs (RMSNorm, decoder layer = residual stream, ...)
                              and how many times each module runs per step
  coverage                    attributed GPU time / total GPU time; if it is low, trust `ops_*`
                              and treat `modules` as a lower bound. Every stage is wrapped so one
                              failure does not hide the others (see `errors`).
Module ranges are `record_function("MOD::<label>")` wrappers around class forwards (installed after
solution.apply(), so they wrap the patched forwards). Backward ops are mapped to their forward
module through the autograd sequence number (FunctionEvent.sequence_nr / fwd_thread).
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
from _train import PROFILE_ACTIVE, PROFILE_WAIT, PROFILE_WARMUP  # noqa: E402

PREFIX = "MOD::"
TOP_OPS = 40
STATE = {"calls": defaultdict(int), "calls_per_step": [], "dtypes": defaultdict(set),
         "weights": {}, "errors": []}


def first_tensor(out):
    import torch

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            found = first_tensor(item)
            if found is not None:
                return found
    return getattr(out, "last_hidden_state", None)


def instrument(owner, attr, label, record_dtype=True):
    """Replace owner.attr (a function or an unbound method) by a wrapper that opens a profiler
    range named MOD::label, counts calls and notes the dtype of the first tensor it returns."""
    from torch.profiler import record_function

    original = getattr(owner, attr)

    def wrapper(*args, **kwargs):
        STATE["calls"][label] += 1
        with record_function(PREFIX + label):
            out = original(*args, **kwargs)
        if record_dtype:
            tensor = first_tensor(out)
            if tensor is not None:
                STATE["dtypes"][label].add(str(tensor.dtype))
            if args and hasattr(args[0], "weight") and label not in STATE["weights"]:
                weight = getattr(args[0], "weight", None)
                if weight is not None and hasattr(weight, "dtype"):
                    STATE["weights"][label] = {"dtype": str(weight.dtype),
                                               "requires_grad": bool(weight.requires_grad)}
        return out

    wrapper.__wrapped__ = original
    setattr(owner, attr, wrapper)


def install_model_hooks():
    """Wrap the classes the model is built from. Missing classes are skipped and reported."""
    import torch.nn as nn

    todo = []
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        todo += [(mq.Qwen2RMSNorm, "forward", "RMSNorm"), (mq.Qwen2MLP, "forward", "MLP"),
                 (mq.Qwen2Attention, "forward", "Attention"),
                 (mq.Qwen2DecoderLayer, "forward", "DecoderLayer"),
                 (mq, "apply_rotary_pos_emb", "rotary")]
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"qwen2 classes: {exc!r}")
    try:
        from peft.tuners.lora import bnb as peft_bnb

        todo.append((peft_bnb.Linear4bit, "forward", "LoRA_Linear4bit"))
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"peft lora bnb Linear4bit: {exc!r}")
    try:
        import bitsandbytes as bnb

        todo.append((bnb.nn.Linear4bit, "forward", "bnb_Linear4bit"))
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"bnb Linear4bit: {exc!r}")
    todo += [(nn.Embedding, "forward", "Embedding"), (nn.Dropout, "forward", "Dropout"),
             (nn.Linear, "forward", "nn_Linear")]
    try:
        import solution.head_loss as head_loss

        todo.append((head_loss, "fused_head_loss", "head_loss_fused"))
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"head_loss: {exc!r}")
    for owner, attr, label in todo:
        try:
            instrument(owner, attr, label)
        except Exception as exc:  # noqa: BLE001
            STATE["errors"].append(f"instrument {label}: {exc!r}")


# ------------------------------------------------------------------ post-processing

def device_us(event):
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def shapes_text(shapes):
    return str(shapes) if shapes else ""


def tables_from_key_averages(prof, steps):
    from torch.autograd import DeviceType

    out = {}
    rows = []
    for ev in prof.key_averages(group_by_input_shape=True):
        if ev.device_type == DeviceType.CUDA:
            continue
        us = device_us(ev)
        if us <= 0 or ev.key.startswith(PREFIX):
            continue
        rows.append({"op": ev.key, "shapes": shapes_text(getattr(ev, "input_shapes", "")),
                     "ms_per_step": us / 1e3 / steps, "count_per_step": ev.count / steps})
    if not rows:  # GPU time was not linked to ATen ops: fall back to CUDA kernel names
        for ev in prof.key_averages():
            if ev.device_type == DeviceType.CUDA and not ev.key.startswith("ProfilerStep") \
                    and not getattr(ev, "is_user_annotation", False):
                rows.append({"op": "[cuda kernel] " + ev.key[:120], "shapes": "",
                             "ms_per_step": device_us(ev) / 1e3 / steps,
                             "count_per_step": ev.count / steps})
        STATE["errors"].append("no GPU time on ATen ops: ops_* list CUDA kernels instead")
    rows.sort(key=lambda r: -r["ms_per_step"])
    out["ops_by_shape"] = rows[:TOP_OPS]
    by_name = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_name[r["op"]][0] += r["ms_per_step"]
        by_name[r["op"]][1] += r["count_per_step"]
    out["ops_by_name"] = [{"op": k, "ms_per_step": v[0], "count_per_step": v[1]}
                          for k, v in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:TOP_OPS]]
    total = 0.0
    for ev in prof.key_averages():
        if ev.device_type == DeviceType.CUDA and not getattr(ev, "is_user_annotation", False) \
                and not ev.key.startswith("ProfilerStep"):
            total += float(getattr(ev, "self_device_time_total", None) or
                           getattr(ev, "self_cuda_time_total", 0) or 0)
    out["total_gpu_ms_per_step"] = total / 1e3 / steps
    return out


def module_of(event):
    """Nearest MOD:: range above the event, or an optimizer / other label."""
    node = event
    while node is not None:
        name = node.name
        if name.startswith(PREFIX):
            return name[len(PREFIX):]
        if name.startswith("Optimizer") or "optimizer" in name.lower():
            return "optimizer"
        node = getattr(node, "cpu_parent", None)
    return None


def attribute(prof, steps, cpu_fallback=False):
    """GPU time per (phase, module) in ms per step via the event tree; also total attributed."""
    events = list(prof.events())
    forward_index = {}
    for ev in events:
        sq = getattr(ev, "sequence_nr", -1)
        if sq is not None and sq >= 0 and not getattr(ev, "fwd_thread", 0):
            forward_index[(ev.thread, sq)] = ev

    def own_us(ev):
        kernels = getattr(ev, "kernels", None)
        if kernels:
            return float(sum(k.duration for k in kernels))
        if cpu_fallback:
            return float(getattr(ev, "self_cpu_time_total", 0.0))
        return 0.0

    totals = defaultdict(float)
    for ev in events:
        us = own_us(ev)
        if us <= 0:
            continue
        label, phase = module_of(ev), "fwd"
        if label is None:
            node = ev
            while node is not None and label is None:
                sq, ft = getattr(node, "sequence_nr", -1), getattr(node, "fwd_thread", 0)
                if sq is not None and sq >= 0 and ft:
                    fwd = forward_index.get((ft, sq))
                    if fwd is not None:
                        label, phase = module_of(fwd), "bwd"
                node = getattr(node, "cpu_parent", None)
        if label is None:
            label = "unattributed"
        totals[(phase, label)] += us
    rows = [{"phase": p, "module": m, "ms_per_step": us / 1e3 / steps}
            for (p, m), us in sorted(totals.items(), key=lambda kv: -kv[1])]
    by_module = defaultdict(float)
    for r in rows:
        by_module[r["module"]] += r["ms_per_step"]
    return {"modules": rows,
            "modules_total": [{"module": m, "ms_per_step": v}
                              for m, v in sorted(by_module.items(), key=lambda kv: -kv[1])],
            "attributed_ms_per_step": sum(r["ms_per_step"] for r in rows)}


IDLE_BUSY_LIMIT = 0.85
ATTN_OPS = ("aten::_efficient_attention_forward", "aten::_efficient_attention_backward", "aten::bmm", "aten::_softmax",
            "aten::_scaled_dot_product_efficient_attention", "aten::_scaled_dot_product_flash_attention")
LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")


def label_finder(events):
    """(phase, module label) of a CPU event: nearest MOD:: range, else the forward module of its autograd node."""
    forward_index = {}
    for ev in events:
        sq = getattr(ev, "sequence_nr", -1)
        if sq is not None and sq >= 0 and not getattr(ev, "fwd_thread", 0):
            forward_index[(ev.thread, sq)] = ev

    def find(ev):
        label, phase = module_of(ev), "fwd"
        node = ev
        while label is None and node is not None:
            sq, ft = getattr(node, "sequence_nr", -1), getattr(node, "fwd_thread", 0)
            if sq is not None and sq >= 0 and ft:
                fwd = forward_index.get((ft, sq))
                if fwd is not None:
                    label, phase = module_of(fwd), "bwd"
            node = getattr(node, "cpu_parent", None)
        return phase, label or "unattributed"

    return find


def idle_stats(prof, steps):
    """How busy the GPU is inside the active ProfilerStep windows, how many CUDA kernels a step launches and how much
    CPU time the launch calls take (cudaLaunchKernel and friends, profiler overhead included). Same events as the
    harness profile (tools/op_profile.py copies the idea of harness/_train.py's profiler, the harness is not touched)."""
    from torch.autograd import DeviceType

    events = list(prof.events())
    windows = sorted((ev.time_range.start, ev.time_range.end) for ev in events
                     if ev.device_type == DeviceType.CPU and ev.name.startswith("ProfilerStep"))
    kernels = sorted((ev.time_range.start, ev.time_range.end) for ev in events
                     if ev.device_type == DeviceType.CUDA and not getattr(ev, "is_user_annotation", False)
                     and not ev.name.startswith("ProfilerStep"))
    n_kernels = sum(1 for ev in events if ev.device_type == DeviceType.CUDA
                    and not getattr(ev, "is_user_annotation", False) and not ev.name.startswith("ProfilerStep")
                    and not ev.name.startswith(("Memcpy", "Memset")))
    launches = [ev for ev in events if ev.device_type == DeviceType.CPU and ev.name.startswith(LAUNCH_NAMES)]
    result = {"windows_found": len(windows), "kernels_all": len(kernels), "kernel_launches_per_step": n_kernels / steps,
              "launch_calls_per_step": len(launches) / steps}
    busy_total = span_total = 0.0
    per_step = []
    for start, end in windows:
        busy, cursor = 0.0, start
        for ks, ke in kernels:                      # union of the kernel intervals clipped to the window
            ks, ke = max(ks, cursor), min(ke, end)
            if ke > ks:
                busy += ke - ks
                cursor = ke
        per_step.append({"step_ms": (end - start) / 1e3, "gpu_busy_ms": busy / 1e3, "busy": busy / (end - start)})
        busy_total, span_total = busy_total + busy, span_total + (end - start)
    result["per_step"] = per_step
    result["gpu_busy_fraction"] = busy_total / span_total if span_total else None
    result["launch_cpu_ms_per_step"] = sum(ev.time_range.elapsed_us() for ev in launches) / 1e3 / steps
    find = label_finder(events)
    sources = defaultdict(float)
    for ev in events:
        found = getattr(ev, "kernels", None)
        if found:
            sources[find(ev)] += len(found)
    result["launch_sources_per_step"] = [{"phase": p, "module": m, "kernels_per_step": n / steps}
                                         for (p, m), n in sorted(sources.items(), key=lambda kv: -kv[1])]
    result["busy_limit"] = IDLE_BUSY_LIMIT
    counted = defaultdict(int)
    for ev in events:
        if ev.device_type == DeviceType.CPU and ev.name in ATTN_OPS:
            counted[ev.name] += 1
    result["attn_stats"] = {name: counted[name] / steps for name in ATTN_OPS}
    return result


def stack_samples(prof, names):
    samples = {}
    for ev in prof.events():
        if ev.name in names and ev.name not in samples and getattr(ev, "stack", None):
            samples[ev.name] = [s.strip() for s in ev.stack[:4]]
    return samples


def analyse(prof, steps, cpu_fallback=False):
    result = {"active_steps": steps}
    for name, fn in (("tables", lambda: tables_from_key_averages(prof, steps)),
                     ("attribution", lambda: attribute(prof, steps, cpu_fallback))):
        try:
            result.update(fn())
        except Exception:  # noqa: BLE001
            STATE["errors"].append(f"{name}: {traceback.format_exc()}")
    try:
        result["idle"] = idle_stats(prof, steps)
    except Exception:  # noqa: BLE001
        STATE["errors"].append(f"idle: {traceback.format_exc()}")
    try:
        result["stack_samples"] = stack_samples(prof, {r["op"] for r in result.get("ops_by_name", [])[:20]})
    except Exception:  # noqa: BLE001
        STATE["errors"].append(f"stacks: {traceback.format_exc()}")
    total, attributed = result.get("total_gpu_ms_per_step"), result.get("attributed_ms_per_step")
    result["coverage"] = (attributed / total) if total and attributed is not None else None
    result["dtypes"] = {k: sorted(v) for k, v in STATE["dtypes"].items()}
    result["weights"] = STATE["weights"]
    result["calls_per_step"] = STATE["calls_per_step"][-1] if STATE["calls_per_step"] else None
    result["errors"] = STATE["errors"]
    return result


def report(result):
    lines = [f"active steps: {result['active_steps']}; total GPU ms/step: {result.get('total_gpu_ms_per_step')};"
             f" attributed ms/step: {result.get('attributed_ms_per_step')}; coverage: {result.get('coverage')}",
             "", "== top ATen ops by self GPU time (all shapes of an op summed) =="]
    for r in result.get("ops_by_name", [])[:20]:
        lines.append(f"{r['ms_per_step']:9.2f} ms/step  {r['count_per_step']:8.1f} calls/step  {r['op']}")
    lines += ["", "== top ops with input shapes =="]
    for r in result.get("ops_by_shape", [])[:20]:
        lines.append(f"{r['ms_per_step']:9.2f} ms/step  {r['count_per_step']:8.1f}/step  {r['op']}  {r['shapes'][:110]}")
    lines += ["", "== GPU ms/step by source module (fwd/bwd) =="]
    for r in result.get("modules", [])[:40]:
        lines.append(f"{r['ms_per_step']:9.2f}  {r['phase']}  {r['module']}")
    lines += ["", "== by module, fwd+bwd =="]
    for r in result.get("modules_total", []):
        lines.append(f"{r['ms_per_step']:9.2f}  {r['module']}")
    idle = result.get("idle")
    if idle:
        lines += ["", "== GPU idle (profiler windows of the active steps; profiler overhead included) ==",
                  f"GPU busy fraction {'n/a' if idle['gpu_busy_fraction'] is None else format(idle['gpu_busy_fraction'], '.3f')} over {idle['windows_found']} steps; "
                  f"CUDA kernels/step {idle['kernel_launches_per_step']:.0f}; launch calls/step {idle['launch_calls_per_step']:.0f}; "
                  f"CPU ms in launch calls/step {idle['launch_cpu_ms_per_step']:.1f}"]
        if idle["gpu_busy_fraction"] is not None and idle["gpu_busy_fraction"] < idle["busy_limit"]:
            lines.append(f"busy < {idle['busy_limit']:.0%}: kernel launches per step by source")
            lines += [f"{r['kernels_per_step']:9.1f}  {r['phase']}  {r['module']}" for r in idle["launch_sources_per_step"][:15]]
        lines.append("ATTN_STATS " + json.dumps({"calls_per_step": idle["attn_stats"], "kernels_per_step": idle["kernel_launches_per_step"],
                                                 "gpu_busy_fraction": idle["gpu_busy_fraction"]}))
    lines += ["", f"dtypes of module outputs: {result.get('dtypes')}", f"weights: {result.get('weights')}",
              f"calls per step: {result.get('calls_per_step')}", f"errors: {result.get('errors')}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ the run

def make_profiler(cpu_only=False):
    from torch.profiler import ProfilerActivity, profile, schedule

    activities = [ProfilerActivity.CPU] if cpu_only else [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    return profile(activities=activities,
                   schedule=schedule(wait=PROFILE_WAIT, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE,
                                     repeat=1),
                   record_shapes=True, with_stack=True)


def run_soup(args):
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    sys.modules["unsloth"] = None
    sys.modules["unsloth_zoo"] = None
    import torch

    torch.empty(1, device="cuda")
    import transformers

    sys.path.insert(0, str(REPO))
    import solution

    solution.apply()
    install_model_hooks()
    prof = make_profiler()
    prof.start()
    total_steps = PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE

    class Probe(transformers.TrainerCallback):
        def __init__(self):
            self.n = 0

        def on_step_begin(self, train_args, state, control, **kwargs):
            STATE["calls"].clear()

        def on_step_end(self, train_args, state, control, **kwargs):
            torch.cuda.synchronize()
            STATE["calls_per_step"].append(dict(STATE["calls"]))
            prof.step()
            self.n += 1
            if self.n >= total_steps:
                control.should_training_stop = True

    original_init = transformers.Trainer.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        self.add_callback(Probe())

    transformers.Trainer.__init__ = init
    code = 0
    try:
        from soup_cli.cli import run

        run()
    except SystemExit as exc:
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
    prof.stop()
    return prof, code


def selftest():
    """Tiny CPU model: embedding -> 2 x (norm, linear, residual) with backward. Checks that the
    attribution code runs and maps backward ops to their forward module."""
    import torch
    import torch.nn as nn

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(32))

        def forward(self, x):
            return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n, self.lin = Norm(), nn.Linear(32, 32)

        def forward(self, x):
            return x + self.lin(self.n(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.layers = nn.Embedding(100, 32), nn.ModuleList([Layer(), Layer()])

        def forward(self, ids):
            x = self.emb(ids)
            for layer in self.layers:
                x = layer(x)
            return x.sum()

    instrument(Norm, "forward", "RMSNorm")
    instrument(Layer, "forward", "DecoderLayer")
    instrument(nn.Embedding, "forward", "Embedding")
    instrument(nn.Linear, "forward", "nn_Linear")
    model, ids = Model(), torch.randint(0, 100, (4, 16))
    prof = make_profiler(cpu_only=True)
    prof.start()
    for _ in range(PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE):
        STATE["calls"].clear()
        model(ids).backward()
        STATE["calls_per_step"].append(dict(STATE["calls"]))
        prof.step()
    prof.stop()
    result = analyse(prof, PROFILE_ACTIVE, cpu_fallback=True)
    text = report(result)
    print(text)
    mods = {(r["phase"], r["module"]) for r in result.get("modules", [])}
    assert ("fwd", "RMSNorm") in mods, mods
    assert ("bwd", "RMSNorm") in mods or ("bwd", "DecoderLayer") in mods, mods
    assert result["calls_per_step"]["RMSNorm"] == 2, result["calls_per_step"]
    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config")
    parser.add_argument("--out", default="out/op_profile.json")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.config:
        sys.exit("--config is required")
    started = time.perf_counter()
    prof, code = run_soup(args)
    result = analyse(prof, PROFILE_ACTIVE)
    result["exit_code"], result["wall_s"] = code, round(time.perf_counter() - started, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    text = report(result)
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    sys.exit(code)


if __name__ == "__main__":
    main()
