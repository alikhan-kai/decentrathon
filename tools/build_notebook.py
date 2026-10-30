"""Build kaggle/soup_kernel_debug.ipynb from templates/kaggle_notebook.ipynb.

    python3 tools/build_notebook.py [--out PATH] [--repo-url URL] [--branch main]
                                    [--commit SHA] [--mode quick|full|full_profile]

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


def build(repo_url, branch, commit, mode):
    tpl = json.loads((ROOT / "templates" / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in tpl["cells"]]
    install = next(line for line in cells[2].splitlines() if line.startswith("!python -m pip install"))
    clone = cells[3].replace(
        '["git", *auth, "clone", "-q", REPO_URL, "/kaggle/working/repo"]',
        '["git", *auth, "clone", "-q", "-b", BRANCH, REPO_URL, "/kaggle/working/repo"]')
    assert clone != cells[3], "template clone cell changed: update tools/build_notebook.py"

    out = [md("# Soup Kernel Challenge: debug runner (generated from templates/kaggle_notebook.ipynb "
              "by tools/build_notebook.py)\n\n"
              "`MODE`: `quick` (pytest + one solution run), `full` (base + solution x2 + checks), "
              "`full_profile` (full + profilers of solution and base).\n"
              "Statuses: `/kaggle/working/out/status.txt`; logs and json: `/kaggle/working/out/`.\n")]
    out.append(code(f'REPO_URL = "{repo_url}"\nBRANCH = "{branch}"\n'
                    f'COMMIT = "{commit}"  # empty = tip of BRANCH\nMODE = "{mode}"\n'))
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
    out.append(code('stage("prepare_data", ["python", "task/prepare_data.py"])\n'))
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
        'if MODE == "quick":\n'
        '    pytests()\n'
        '    stage("run_task_solution", ["python", "harness/run_task.py", "--only", "solution",\n'
        '                                "--out", "runs_try"])\n'
        '    mp = REPO / "runs_try/solution_1/metrics.json"\n'
        '    if mp.exists():\n'
        '        m = json.load(open(mp))\n'
        '        for k in ("triton_launches", "own_kernel_ms", "timed_loop_s", "torch_max_reserved_mib"):\n'
        '            print(k, "=", m.get(k))\n'
        'elif MODE in ("full", "full_profile"):\n'
        '    if MODE == "full_profile":\n'
        '        pytests()\n'
        '    flags = ["--profile"] if MODE == "full_profile" else []\n'
        '    stage("run_task_full", ["python", "harness/run_task.py", "--repeats", "2", *flags])\n'
        '    stage("check_rules", ["python", "harness/check_rules.py"])\n'
        '    if MODE == "full_profile":\n'
        '        base_profile()\n'
        'else:\n'
        '    raise SystemExit("MODE must be quick, full or full_profile")\n'))
    out.append(code(
        'for pattern in ("runs*/**/train.log", "runs*/eval_*.log", "runs*/result.json",\n'
        '                "runs*/checks.json", "runs*/**/metrics.json", "runs*/solution_profile/profile.json",\n'
        '                "runs/base_profile/train.log"):\n'
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
    p.add_argument("--out", default=str(ROOT / "kaggle" / "soup_kernel_debug.ipynb"))
    p.add_argument("--repo-url", default="https://github.com/alikhan-kai/decentrathon.git")
    p.add_argument("--branch", default="main")
    p.add_argument("--commit", default="")
    p.add_argument("--mode", default="quick", choices=("quick", "full", "full_profile"))
    a = p.parse_args()
    nb = build(a.repo_url, a.branch, a.commit, a.mode)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
