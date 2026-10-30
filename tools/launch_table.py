"""CUDA kernel launches per step by source, from the saved pack_and_idle profiles (no GPU).

    python3 tools/launch_table.py [docs/measurements/pack_and_idle_f8948e8]

Sources are the module labels of tools/op_profile.py (a kernel belongs to the innermost labelled module that launched it;
backward to the forward module of its autograd node). Per source: launches/step (idle.launch_sources_per_step), GPU ms/step
(modules), launch share, and the CPU time estimate = launches x the average cudaLaunchKernel time of the whole profile
(idle.launch_cpu_ms_per_step / idle.launch_calls_per_step; profiler overhead included, one average for all sources).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = [("SDPA/attention (Qwen2Attention excl. projections)", {"Attention"}),
          ("peft LoRA (A, B, dropout, scale, add)", {"LoRA_Linear4bit", "nn_Linear", "Dropout"}),
          ("bnb (dequant + gemm)", {"bnb_Linear4bit"}),
          ("rmsnorm", {"RMSNorm"}), ("swiglu (MLP)", {"MLP"}), ("rotary", {"rotary"}),
          ("residual adds (DecoderLayer)", {"DecoderLayer"}), ("head_loss", {"head_loss_fused"}),
          ("optimizer", {"optimizer"}), ("other (embedding, unattributed)", {"Embedding", "unattributed"})]


def table(path, title):
    data = json.loads(Path(path).read_text())
    idle = data["idle"]
    launches, ms = defaultdict(float), defaultdict(float)
    for row in idle["launch_sources_per_step"]:
        launches[(row["module"], row["phase"])] += row["kernels_per_step"]
    for row in data["modules"]:
        ms[(row["module"], row["phase"])] += row["ms_per_step"]
    total_launches = idle["kernel_launches_per_step"]
    cpu_per_launch_us = idle["launch_cpu_ms_per_step"] * 1e3 / idle["launch_calls_per_step"]
    print(f"\n== {title}: {total_launches:.0f} kernels/step, GPU busy {idle['gpu_busy_fraction']:.3f}, GPU {data['total_gpu_ms_per_step']:.1f} ms/step, "
          f"CPU in launch calls {idle['launch_cpu_ms_per_step']:.1f} ms/step ({cpu_per_launch_us:.2f} us per launch)")
    print(f"{'source':52s} {'fwd':>7s} {'bwd':>7s} {'launches':>9s} {'share':>6s} {'GPU ms':>7s} {'CPU ms*':>8s} {'GPU us/launch':>13s}")
    seen = set()
    for name, modules in GROUPS:
        f = sum(launches[(m, "fwd")] for m in modules)
        b = sum(launches[(m, "bwd")] for m in modules)
        g = sum(ms[(m, p)] for m in modules for p in ("fwd", "bwd"))
        seen |= modules
        n = f + b
        print(f"{name:52s} {f:7.0f} {b:7.0f} {n:9.0f} {n / total_launches * 100:5.1f}% {g:7.1f} {n * cpu_per_launch_us / 1e3:8.1f} {g * 1e3 / n if n else 0:13.1f}")
    rest = {m for m, _ in list(launches) + list(ms)} - seen
    if rest:
        print("sources not in any group:", sorted(rest))
    return data


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/measurements/pack_and_idle_f8948e8")
    start = table(base / "idle_start.json", "start aa46608 (padded, SDPA with mask)")
    this = table(base / "idle_this.json", "start-ansar f8948e8 (packed, per-row SDPA)")
    print("\n* CPU ms = launches x average cudaLaunchKernel time; one average for all sources, so only the order of magnitude per source.")
    print("\nSDPA kernels by ATen op (idle_this.json ops_by_name, calls per step; only ops in the saved top-40 list):")
    for label, d in (("start", start), ("this", this)):
        ops = {o["op"]: o for o in d["ops_by_name"]}
        keys = [k for k in ops if any(w in k for w in ("attention", "bmm", "softmax", "isneginf", "where", "aten::all", "aten::cat"))]
        print(" ", label, {k: (round(ops[k]["ms_per_step"], 1), round(ops[k]["count_per_step"])) for k in keys})


if __name__ == "__main__":
    main()
