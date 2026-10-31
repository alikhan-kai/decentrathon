"""Bounded fresh-process CPU probe; no CUDA jobs and no package installation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--python", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise SystemExit("use a new evidence directory")
a.output.mkdir(parents=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
probe = Path(__file__).with_name("import_probe.py")
rows = []
deadline = time.monotonic() + 180
for variant in ("metadata", "lazy-bnb"):
    for arm in ("control", variant, variant, "control") * 2:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit("CPU measurement deadline exceeded")
        command = [a.python, "-B", str(probe), arm]
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, env=env,
                                timeout=min(20, remaining))
        stem = f"{len(rows):02d}-{variant}-{arm}"
        (a.output / (stem + ".stdout.log")).write_text(result.stdout)
        (a.output / (stem + ".stderr.log")).write_text(result.stderr)
        if result.returncode:
            raise SystemExit(f"probe failed: {stem}; see raw logs")
        row = json.loads(result.stdout.splitlines()[-1])
        row.update(experiment=variant, process_s=time.perf_counter() - start, command=command)
        rows.append(row)
        (a.output / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
summaries = {}
for variant in ("metadata", "lazy-bnb"):
    summaries[variant] = {arm: {key: statistics.median(
        r[key] for r in rows if r["experiment"] == variant and r["arm"] == arm)
        for key in ("import_s", "process_s")} for arm in ("control", variant)}
receipt = {"complete": True, "scope": "CPU import-only, four fresh processes per arm",
           "summaries": summaries, "rows": rows}
(a.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
