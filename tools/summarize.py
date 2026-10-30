"""Tables from a downloaded Kaggle output.   python3 tools/summarize.py [kaggle/out_<MODE>_<time>]

Reads result.json, checks.json, solution_profile/profile.json and base_profile.json from the
directory (default: the newest kaggle/out_*). Prints only what is in those files; what is missing
is printed as "не знаю". Standard library only.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_SHARE = 0.05  # check 4: own kernels >= 5% of the timed loop

# First matching rule wins (case-insensitive substring of the CUDA kernel name).
GROUPS = [
    ("NF4 dequant", ("dequant", "bitsandbytes", "4bit")),
    ("softmax/CE", ("softmax", "nll", "cross_entropy", "head_loss_kernel")),
    ("attention", ("attention", "flash", "fmha", "sdpa", "efficient_")),
    ("matmul", ("gemm", "gemv", "cutlass", "cublas", "xmma", "s16816", "s1688", "s884", "turing_",
                "volta_", "ampere_", "splitk", "wmma", "nvjet")),
    ("elementwise/cast", ("elementwise", "vectorized", "unrolled", "copy", "cast", "catarray",
                          "swiglu", "index", "fill", "triton_poi")),
]


def find(root, name):
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError):
        return None


def fmt(value, spec=""):
    return "не знаю" if value is None else format(value, spec)


def median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def group_of(name):
    low = name.lower()
    for group, words in GROUPS:
        if any(w in low for w in words):
            return group
    return "прочее"


def is_example(name):
    return name.startswith("example_")


def section_a(result):
    print("\n## a) База vs решение  (источник: result.json)")
    if not result:
        print("result.json не найден"); return
    runs, summ = result.get("runs", {}), result.get("summary", {})
    print(f"{'':10}{'wall_s':>9}{'peak MiB':>10}{'train_loop_s':>14}{'median step s':>15}{'eval loss':>11}")
    for side in ("base", "solution"):
        s = summ.get(side)
        if not s or not s.get("ok"):
            print(f"{side:10}{'нет/упал':>9}"); continue
        med = median([r.get("metrics", {}).get("median_step_s") for r in runs.get(side, [])])
        print(f"{side:10}{s['wall_s']:>9}{s['gpu_peak_mib']:>10}{fmt(s.get('train_loop_s')):>14}"
              f"{fmt(med):>15}{fmt(s.get('eval_loss')):>11}")
    r = summ.get("ratios")
    if r:
        print(f"speedup {r['speedup']}x, memory_ratio {r['memory_ratio']}x, "
              f"quality_delta_pct {fmt(r.get('quality_delta_pct'))} %")
    else:
        print("speedup / memory_ratio / quality_delta_pct: не знаю (нужны base и solution)")


def section_b(checks):
    print("\n## b) Проверки 1-10  (источник: checks.json)")
    if not checks:
        print("checks.json не найден"); return
    for c in checks.get("checks", []):
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}" + ("" if c["ok"] else f"\n      причина: {c['detail']}"))
    for n in checks.get("notes", []):
        print("NOTE ", n)


def section_c(result):
    print("\n## c) Накладные расходы и наши кернелы  (источник: result.json, metrics)")
    if not result:
        print("result.json не найден"); return
    for side, runs in result.get("runs", {}).items():
        for r in runs:
            m = r.get("metrics", {})
            loop = m.get("train_loop_s")
            over = round(r["wall_s"] - loop, 2) if loop else None
            print(f"{r['run']}: wall {r['wall_s']} s - train_loop {fmt(loop)} s = вне цикла {fmt(over)} s; "
                  f"apply_s {fmt(m.get('apply_s'))}")
    sol = result.get("runs", {}).get("solution", [])
    for r in sol:
        m = r.get("metrics", {})
        loop, ms, n = m.get("timed_loop_s"), m.get("own_kernel_ms", {}), m.get("triton_launches", {})
        print(f"\n{r['run']}: timed_loop_s {fmt(loop)}")
        tot_own = 0.0
        for name in sorted(ms, key=lambda k: -ms[k]):
            share = 100 * ms[name] / 1000 / loop if loop else None
            tag = "пример (не считается)" if is_example(name) else "наш"
            if not is_example(name):
                tot_own += ms[name]
            print(f"  {name:34} запусков {n.get(name, 'не знаю'):>6}  {ms[name]:>9.1f} мс  {fmt(share, '.2f')}%  [{tag}]")
        if loop:
            print(f"  наши кернелы (без example_) вместе: {tot_own:.1f} мс = {100 * tot_own / 1000 / loop:.2f}% (нужно >= 5%)")


def profile_table(title, prof, active_default=None):
    print(f"\n### {title}")
    if not prof:
        print("профиль не найден: не знаю"); return None
    kernels, total = prof["kernels"], prof["total_gpu_us"]
    steps = prof.get("active_steps") or active_default
    print(f"всего GPU {total / 1e3:.1f} мс за {steps} шагов = {total / 1e3 / steps:.1f} мс/шаг (с накладными профайлера)")
    print(f"{'#':>2} {'кернел':62}{'мс':>9}{'шт':>7}{'%':>6}")
    for i, k in enumerate(kernels[:25], 1):
        print(f"{i:>2} {k['name'][:61]:62}{k['time_us'] / 1e3:>9.1f}{k['count']:>7}{100 * k['time_us'] / total:>6.1f}")
    groups = {}
    for k in kernels:
        g = group_of(k["name"])
        groups[g] = groups.get(g, 0) + k["time_us"]
    print("по группам:")
    for g, us in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:18}{us / 1e3:>9.1f} мс  {100 * us / total:>5.1f}%  ({us / 1e3 / steps:.2f} мс/шаг)")
    return groups


def section_d(base_prof, sol_prof):
    print("\n## d) Топ-25 CUDA-кернелов  (источник: base_profile.json, solution_profile/profile.json)")
    print("правила групп (первое совпавшее, подстрока в имени без учёта регистра):")
    for g, words in GROUPS:
        print(f"  {g}: {', '.join(words)}")
    print("  прочее: всё остальное")
    profile_table("база", base_prof)
    profile_table("решение", sol_prof)


def section_e(result):
    print("\n## e) Сколько нужно нашим кернелам для 5%")
    sol = (result or {}).get("runs", {}).get("solution", [])
    rows = []
    for r in sol:
        m = r.get("metrics", {})
        loop, steps = m.get("timed_loop_s"), m.get("timed_steps")
        if not loop or not steps:
            continue
        own = sum(v for k, v in m.get("own_kernel_ms", {}).items() if not is_example(k))
        step_ms = 1000 * loop / steps
        need = MIN_SHARE * step_ms
        rows.append((r["run"], step_ms, own / steps, need))
        print(f"{r['run']}: шаг (с шага 3) {step_ms:.1f} мс; наши {own / steps:.2f} мс/шаг; "
              f"для 5% нужно {need:.2f} мс/шаг; не хватает {max(0.0, need - own / steps):.2f} мс/шаг")
    if not rows:
        print("не знаю (нет решения в result.json)")
    else:
        print("оговорка: если кернелы ускоряют шаг, 5% считается от более короткого шага, нужно чуть меньше")


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        outs = sorted((ROOT / "kaggle").glob("out_*"))
        root = outs[-1] if outs else None
    if root is None or not root.exists():
        sys.exit("usage: tools/summarize.py <kaggle output dir>")
    print(f"# Отчёт по {root}")
    status = find(root, "status.txt")
    if status:
        print("\nстатусы стадий (status.txt):\n" + status.read_text().rstrip())
    result = load(find(root, "result.json"))
    section_a(result)
    section_b(load(find(root, "checks.json")))
    section_c(result)
    section_d(load(find(root, "base_profile.json")),
              load(find(root / "out" if (root / "out").exists() else root, "profile.json")))
    section_e(result)


if __name__ == "__main__":
    main()
