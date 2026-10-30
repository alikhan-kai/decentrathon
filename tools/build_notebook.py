"""Build kaggle/soup_kernel_debug.ipynb from templates/kaggle_notebook.ipynb.

    python3 tools/build_notebook.py --username NAME [--branch main] [--commit SHA] [--repo URL]
        [--mode quick|unit|full|full_profile|check_and_ops|pack_check|quick_ab|pack_and_idle|robust|ab_attn] [--module-test solution/test_x.py]
        [--slug soup-kernel-debug]
Writes kaggle/build/<username>/soup_kernel_debug.ipynb and kernel-metadata.json (the folder is in
.gitignore, so nobody's build conflicts with anybody else's). No tokens are touched.

The install and clone cells are copied from the template; everything after them runs stages that
write a status line each to /kaggle/working/out/status.txt, so a failed stage hides nothing.
COMMIT empty = tip of BRANCH. The template itself is never changed.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(True)}


def build(repo_url, branch, commit, mode, username, module_test):
    tpl = json.loads((ROOT / "templates" / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in tpl["cells"]]
    install = next(line for line in cells[2].splitlines() if line.startswith("!python -m pip install"))
    clone = cells[3].replace(
        '["git", *auth, "clone", "-q", REPO_URL, "/kaggle/working/repo"]',
        '["git", *auth, "clone", "-q", "-b", BRANCH, REPO_URL, "/kaggle/working/repo"]')
    assert clone != cells[3], "template clone cell changed: update tools/build_notebook.py"

    out = [md("# Soup Kernel Challenge: debug runner (generated from templates/kaggle_notebook.ipynb "
              "by tools/build_notebook.py)\n\n"
              "`MODE`: `unit` (pytest of MODULE_TEST, a file or a directory; skip reasons printed), `quick` (pytest + one solution run), "
              "`full` (base + solution x2 + checks), `full_profile` (full + profilers of solution and "
              "base), `check_and_ops` (quick + per-op/per-module profile of the solution).\n"
              "Statuses: `/kaggle/working/out/status.txt`; logs and json: `/kaggle/working/out/`.\n")]
    out.append(code(f'REPO_URL = "{repo_url}"\nBRANCH = "{branch}"\n'
                    f'COMMIT = "{commit}"  # empty = tip of BRANCH\nMODE = "{mode}"\n'
                    f'MODULE_TEST = "{module_test}"  # unit mode: a test file or a directory (default solution/)\n'
                    f'USERNAME = "{username}"  # Kaggle account that runs this notebook\n'))
    out.append(code(
        'import subprocess\n'
        'print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)\n'
        'names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],\n'
        '                       capture_output=True, text=True).stdout\n'
        'assert "Tesla T4" in names, "Not a Tesla T4: " + names\n'))
    out.append(code(
        'import os\ntry:\n    from kaggle_secrets import UserSecretsClient\n'
        '    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")\n'
        'except Exception:\n    pass\n' + install + '\n'))
    out.append(code(clone))
    out.append(code(
        'import glob, json, shutil, subprocess, sys\n'
        'from pathlib import Path\n'
        'REPO = Path("/kaggle/working/repo")\n'
        'OUT = Path("/kaggle/working/out")\n'
        'OUT.mkdir(parents=True, exist_ok=True)\n'
        '(OUT / "status.txt").write_text("")\n'
        'ENV = dict(os.environ)\n\n'
        'def stage(name, cmd, env=None):\n'
        '    """Run one stage, keep its log, record the status; never raises."""\n'
        '    with open(OUT / f"{name}.log", "w") as log:\n'
        '        code = subprocess.run(cmd, cwd=REPO, env=env or ENV, stdout=log,\n'
        '                              stderr=subprocess.STDOUT).returncode\n'
        '    line = f"{name}: {\'ok\' if code == 0 else \'FAIL (exit %d)\' % code}"\n'
        '    with open(OUT / "status.txt", "a") as f:\n'
        '        f.write(line + "\\n")\n'
        '    print(line)\n'
        '    if code:\n'
        '        print("".join(open(OUT / f"{name}.log").readlines()[-60:]))\n'
        '    return code\n'))
    out.append(code('if MODE != "unit":\n    stage("prepare_data", ["python", "task/prepare_data.py"])\n'))
    out.append(code(
        'def pytests():\n'
        '    stage("pytest_solution", ["python", "-m", "pytest", "solution", "-q"])\n'
        '    stage("pytest_tests", ["python", "-m", "pytest", "tests", "-q"],\n'
        '          env=dict(ENV, SOUP_APPLY_SOLUTION="1"))\n\n'
        'def base_profile():\n'
        '    """Profile of the BASE (run_task profiles only the solution). Same env as run_task."""\n'
        '    import yaml\n'
        '    src = REPO / "runs/base_1/task.yaml"\n'
        '    if not src.exists():\n'
        '        found = sorted(glob.glob(str(REPO / "runs/**/task.yaml"), recursive=True))\n'
        '        print("runs/base_1/task.yaml missing; task.yaml files found:", found)\n'
        '        (OUT / "status.txt").open("a").write("base_profile: FAIL (no runs/base_1/task.yaml)\\n")\n'
        '        return\n'
        '    run_dir = REPO / "runs/base_profile"\n'
        '    run_dir.mkdir(parents=True, exist_ok=True)\n'
        '    cfg = yaml.safe_load(src.read_text())\n'
        '    cfg["output"] = str(run_dir / "adapter")  # do not overwrite the measured base adapter\n'
        '    (run_dir / "task.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))\n'
        '    sys.path.insert(0, str(REPO / "harness"))\n'
        '    from common import OFFLINE_ENV\n'
        '    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),\n'
        '               TRITON_CACHE_DIR=str(run_dir / "triton_cache"),\n'
        '               TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"))\n'
        '    stage("base_profile", ["python", "harness/_train.py", "--config", str(run_dir / "task.yaml"),\n'
        '                           "--metrics", str(OUT / "base_profile_metrics.json"),\n'
        '                           "--profile", str(OUT / "base_profile.json")], env=env)\n\n'
        'def op_profile():\n'
        '    """Per-op / per-module profile of the solution (tools/op_profile.py)."""\n'
        '    import yaml\n'
        '    src = REPO / "runs_try/solution_1/task.yaml"\n'
        '    if not src.exists():\n'
        '        found = sorted(glob.glob(str(REPO / "runs*/**/task.yaml"), recursive=True))\n'
        '        print("runs_try/solution_1/task.yaml missing; found:", found)\n'
        '        (OUT / "status.txt").open("a").write("op_profile: FAIL (no task.yaml)\\n")\n'
        '        return\n'
        '    run_dir = REPO / "runs_try/op_profile"\n'
        '    run_dir.mkdir(parents=True, exist_ok=True)\n'
        '    cfg = yaml.safe_load(src.read_text())\n'
        '    cfg["output"] = str(run_dir / "adapter")\n'
        '    (run_dir / "task.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))\n'
        '    sys.path.insert(0, str(REPO / "harness"))\n'
        '    from common import OFFLINE_ENV\n'
        '    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),\n'
        '               TRITON_CACHE_DIR=str(run_dir / "triton_cache"),\n'
        '               TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"))\n'
        '    stage("op_profile", ["python", "tools/op_profile.py", "--config", str(run_dir / "task.yaml"),\n'
        '                         "--out", str(OUT / "op_profile.json")], env=env)\n\n'
        'def swap_solution(rev):\n'
        '    """Replace solution/ by the one of git revision `rev` (whole folder, no leftovers)."""\n'
        '    shutil.rmtree(REPO / "solution")\n'
        '    return stage(f"git_solution_{rev[:7]}", ["git", "checkout", rev, "--", "solution"])\n\n'
        'def pack_and_idle():\n'
        '    """One session: accuracy of the packed path (tools/pack_accuracy.py), then the GPU idle profile of start and\n'
        '    of this commit (tools/op_profile.py, 24 steps each)."""\n'
        '    import yaml\n'
        '    stage("warm_model", ["python", "-c", "from huggingface_hub import snapshot_download as d; "\n'
        '                                          "print(d(\'Qwen/Qwen2.5-1.5B-Instruct\'))"])\n'
        '    stage("pack_accuracy", ["python", "tools/pack_accuracy.py", "--out", str(OUT / "pack_accuracy.json")])\n'
        '    mine = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()\n'
        '    stage("fetch_start", ["git", "fetch", "-q", "origin", "start"])\n'
        '    start = subprocess.run(["git", "rev-parse", "FETCH_HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()\n'
        '    (OUT / "idle_commits.txt").write_text(f"start {start}\\nthis {mine}\\n")\n'
        '    sys.path.insert(0, str(REPO / "harness"))\n'
        '    from common import OFFLINE_ENV\n'
        '    for tag, rev in (("start", start), ("this", mine)):\n'
        '        swap_solution(rev)\n'
        '        run_dir = REPO / "runs_idle" / tag\n'
        '        run_dir.mkdir(parents=True, exist_ok=True)\n'
        '        cfg = yaml.safe_load((REPO / "task/task.yaml").read_text())\n'
        '        cfg["data"]["train"], cfg["output"] = str(REPO / "data/train.jsonl"), str(run_dir / "adapter")\n'
        '        cfg["training"]["seed"] = 1\n'
        '        (run_dir / "task.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))\n'
        '        env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),\n'
        '                   TRITON_CACHE_DIR=str(run_dir / "triton_cache"),\n'
        '                   TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"))\n'
        '        stage(f"idle_{tag}", ["python", "tools/op_profile.py", "--config", str(run_dir / "task.yaml"),\n'
        '                              "--out", str(OUT / f"idle_{tag}.json")], env=env)\n'
        '    swap_solution(mine)\n\n'
        'def set_variant(v):\n'
        '    """Attention variant of solution/packed_attn.py, switched by rewriting its VARIANT line (no environment)."""\n'
        '    import re\n'
        '    path = REPO / "solution/packed_attn.py"\n'
        '    text = path.read_text()\n'
        '    new = re.sub(r\'(?m)^VARIANT = "[A-Z0-9]+"\', f\'VARIANT = "{v}"\', text)\n'
        '    assert f\'VARIANT = "{v}"\' in new, "VARIANT line not found"\n'
        '    path.write_text(new)\n\n'
        'def ab_attn():\n'
        '    """One T4 session: variants A, A2, B, A, A2, B, each a quick solution run; the first run of every variant is also\n'
        '    profiled once (GPU busy, kernels per step, ATTN_STATS line)."""\n'
        '    import yaml\n'
        '    sys.path.insert(0, str(REPO / "harness"))\n'
        '    from common import OFFLINE_ENV\n'
        '    seen = set()\n'
        '    for i, v in enumerate(["A", "A2", "B", "A", "A2", "B"]):\n'
        '        set_variant(v)\n'
        '        stage(f"run_{i}_{v}", ["python", "harness/run_task.py", "--only", "solution", "--out", f"runs_attn_{i}_{v}"])\n'
        '        if v in seen:\n'
        '            continue\n'
        '        seen.add(v)\n'
        '        run_dir = REPO / "runs_attn_profile" / v\n'
        '        run_dir.mkdir(parents=True, exist_ok=True)\n'
        '        cfg = yaml.safe_load((REPO / "task/task.yaml").read_text())\n'
        '        cfg["data"]["train"], cfg["output"] = str(REPO / "data/train.jsonl"), str(run_dir / "adapter")\n'
        '        cfg["training"]["seed"] = 1\n'
        '        (run_dir / "task.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))\n'
        '        env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),\n'
        '                   TRITON_CACHE_DIR=str(run_dir / "triton_cache"),\n'
        '                   TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"))\n'
        '        stage(f"profile_{v}", ["python", "tools/op_profile.py", "--config", str(run_dir / "task.yaml"),\n'
        '                               "--out", str(OUT / f"attn_profile_{v}.json")], env=env)\n'
        '        print("variant", v, [l for l in open(OUT / f"profile_{v}.log") if l.startswith("ATTN_STATS")][-1:])\n'
        '    set_variant("A")\n\n'
        'def ab():\n'
        '    """Same T4 session: start (A) / this commit (B) / start (A) so session drift shows."""\n'
        '    mine = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()\n'
        '    stage("fetch_start", ["git", "fetch", "-q", "origin", "start"])\n'
        '    start = subprocess.run(["git", "rev-parse", "FETCH_HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()\n'
        '    (OUT / "ab_commits.txt").write_text(f"start {start}\\nthis {mine}\\n")\n'
        '    for tag, rev in (("start1", start), ("this", mine), ("start2", start)):\n'
        '        swap_solution(rev)\n'
        '        stage(f"run_{tag}", ["python", "harness/run_task.py", "--only", "solution", "--out", f"runs_ab_{tag}"])\n'
        '    swap_solution(mine)\n\n'
        'def quick_metrics():\n'
        '    mp = REPO / "runs_try/solution_1/metrics.json"\n'
        '    if mp.exists():\n'
        '        m = json.load(open(mp))\n'
        '        for k in ("triton_launches", "own_kernel_ms", "timed_loop_s", "torch_max_reserved_mib"):\n'
        '            print(k, "=", m.get(k))\n\n'
        'if MODE == "unit":\n'
        '    assert MODULE_TEST, "unit mode needs MODULE_TEST"\n'
        '    stage("pytest_unit", ["python", "-m", "pytest", *MODULE_TEST.split(), "-q", "-rs"])\n'
        'elif MODE == "ab_attn":\n'
        '    pytests()\n'
        '    ab_attn()\n'
        'elif MODE == "robust":\n'
        '    stage("make_robust_data", ["python", "tools/make_robust_data.py", "--out", "data_robust"])\n'
        '    stage("run_task_robust", ["python", "harness/run_task.py", "--repeats", "2", "--seed", "7",\n'
        '                              "--train-file", "data_robust/train.jsonl", "--eval-file", "data_robust/eval.jsonl",\n'
        '                              "--out", "runs_robust"])\n'
        '    stage("check_rules_robust", ["python", "harness/check_rules.py", "--result", "runs_robust/result.json"])\n'
        'elif MODE == "pack_and_idle":\n'
        '    pack_and_idle()\n'
        'elif MODE == "quick_ab":\n'
        '    pytests()\n'
        '    ab()\n'
        'elif MODE == "pack_check":\n'
        '    stage("pack_check", ["python", "tools/pack_check.py", "--out", str(OUT / "pack_check.json")])\n'
        'elif MODE in ("quick", "check_and_ops"):\n'
        '    pytests()\n'
        '    stage("run_task_solution", ["python", "harness/run_task.py", "--only", "solution",\n'
        '                                "--out", "runs_try"])\n'
        '    quick_metrics()\n'
        '    if MODE == "check_and_ops":\n'
        '        op_profile()\n'
        'elif MODE in ("full", "full_profile"):\n'
        '    if MODE == "full_profile":\n'
        '        pytests()\n'
        '    flags = ["--profile"] if MODE == "full_profile" else []\n'
        '    stage("run_task_full", ["python", "harness/run_task.py", "--repeats", "2", *flags])\n'
        '    stage("check_rules", ["python", "harness/check_rules.py"])\n'
        '    if MODE == "full_profile":\n'
        '        base_profile()\n'
        'else:\n'
        '    raise SystemExit("MODE must be unit, quick, full, full_profile, check_and_ops, pack_check, quick_ab, pack_and_idle, robust or ab_attn")\n'))
    out.append(code(
        'for pattern in ("runs*/**/train.log", "runs*/eval_*.log", "runs*/result.json",\n'
        '                "runs*/checks.json", "runs*/**/metrics.json", "runs*/solution_profile/profile.json",\n'
        '                "runs/base_profile/train.log", "runs_try/op_profile/train.log"):\n'
        '    for p in glob.glob(str(REPO / pattern), recursive=True):\n'
        '        rel = Path(p).relative_to(REPO)\n'
        '        dst = OUT / "files" / rel\n'
        '        dst.parent.mkdir(parents=True, exist_ok=True)\n'
        '        shutil.copy(p, dst)\n'
        'shutil.rmtree(REPO, ignore_errors=True)  # keep the kernel output small\n'
        'print((OUT / "status.txt").read_text())\n'))
    return {"cells": out, "metadata": tpl["metadata"], "nbformat": tpl.get("nbformat", 4),
            "nbformat_minor": tpl.get("nbformat_minor", 5)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--username", required=True, help="Kaggle account that will run the notebook")
    p.add_argument("--repo", "--repo-url", dest="repo", default="https://github.com/alikhan-kai/decentrathon.git")
    p.add_argument("--branch", default="main")
    p.add_argument("--commit", default="")
    p.add_argument("--mode", default="quick",
                   choices=("quick", "unit", "full", "full_profile", "check_and_ops", "pack_check", "quick_ab", "pack_and_idle", "robust", "ab_attn"))
    p.add_argument("--module-test", default="", help="unit mode: path of one test file")
    p.add_argument("--slug", default="soup-kernel-debug")
    a = p.parse_args()
    out_dir = ROOT / "kaggle" / "build" / a.username
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = build(a.repo, a.branch, a.commit, a.mode, a.username, a.module_test)
    (out_dir / "soup_kernel_debug.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    meta = {"id": f"{a.username}/{a.slug}", "title": a.slug, "code_file": "soup_kernel_debug.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": []}
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
