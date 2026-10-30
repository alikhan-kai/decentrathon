"""Build a private, self-contained GPU probe before repository access is ready.

This development notebook is not the commit-pinned final submission notebook.
Only the explicitly listed project code is bundled; credentials and local state
are excluded. Run: python -m solution.make_probe /tmp/ember-probe.ipynb
"""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [root / "LICENSE", root / "NOTICE"]
    for directory in ("solution", "harness", "tests", "task"):
        paths.extend(sorted((root / directory).glob("*.py")))
    paths.append(root / "task/task.yaml")
    paths = [path for path in paths if path.name != "make_probe.py"]
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 10, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    data = memory.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    payload = base64.b64encode(data).decode()
    cells = []
    def add(kind, source):
        if kind == "code":
            ast.parse(source)
        item = {"id": f"ember-probe-{len(cells)}", "cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    add("markdown", "# Ember — private T4 development probe\n\nContains the exact source snapshot named by the SHA256 below. Select GPU T4 x2 and Internet on. This development artifact is not the final commit-pinned submission. No credentials are embedded. The original harness, task and regression tests are included unchanged.")
    add("code", '''import os, subprocess, sys
SETUP_READY = SOURCE_READY = VALIDATION_READY = BENCHMARK_READY = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True)
print(gpus)
assert "T4" in gpus, "Select GPU T4 x2"
# Kaggle's preinstalled torchao 0.10 is incompatible with PEFT 0.20.
# Remove this unused optional package before both measured implementations.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soup-cli[train]==0.77.0", "transformers==5.16.1", "peft==0.20.0", "trl==0.29.1", "bitsandbytes==0.50.2", "accelerate==1.14.0", "pytest"], check=True)
SETUP_READY = True
''')
    add("code", f'''import base64, hashlib, io, pathlib, tempfile, zipfile
SOURCE_READY = VALIDATION_READY = BENCHMARK_READY = False
assert SETUP_READY, "The hardware/dependency setup must succeed first"
payload = {payload!r}
bundle = base64.b64decode(payload)
assert hashlib.sha256(bundle).hexdigest() == {digest!r}
root = pathlib.Path(tempfile.mkdtemp(prefix="ember-probe-", dir="/kaggle/working"))
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    for name in archive.namelist():
        assert not pathlib.PurePosixPath(name).is_absolute() and ".." not in pathlib.PurePosixPath(name).parts
    archive.extractall(root)
del payload, bundle
print("Source snapshot SHA256:", {digest!r})
def run(*args, env=None):
    subprocess.run([sys.executable, *args], cwd=root, env=env, check=True)
SOURCE_READY = True
''')
    add("code", '''import importlib.metadata, json
VALIDATION_READY = BENCHMARK_READY = False
assert SOURCE_READY, "Source setup must succeed first"
versions = {name: importlib.metadata.version(name) for name in ["torch", "triton", "soup-cli", "transformers", "peft", "trl", "bitsandbytes", "accelerate"]}
print(json.dumps(versions, indent=2))
run("-m", "pytest", "solution", "-q", "--tb=short")
run("-m", "pytest", "tests", "-q", env=dict(os.environ, SOUP_APPLY_SOLUTION="1"))
VALIDATION_READY = True
''')
    add("markdown", "## Initial official measurement\n\nThe next cells run one baseline and one solution with the unchanged harness. These are development measurements. Final submission still needs two repeats from the platform repository's final commit. No component benchmarks run concurrently.")
    add("code", '''BENCHMARK_READY = False
assert VALIDATION_READY, "CUDA and regression tests must pass before benchmarking"
run("task/prepare_data.py")
run("harness/run_task.py")
BENCHMARK_READY = True
''')
    add("code", '''assert BENCHMARK_READY, "The benchmark must finish before checking results"
run("harness/check_rules.py")
run("harness/score.py", "runs/result.json")
''')
    add("code", '''import shutil
from IPython.display import display, FileLink
assert SOURCE_READY and (root / "runs").is_dir(), "No run logs exist to export"
print("Diagnostic artifacts; exporting them does not imply the benchmark or rule checks passed.")
path = shutil.make_archive("/kaggle/working/ember-probe-results", "zip", root_dir=root, base_dir="runs")
display(FileLink(path))
''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "cells": cells}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(json.dumps({"path": str(args.output), "files": len(paths), "bundle_bytes": len(data), "sha256": digest}))


if __name__ == "__main__":
    main()
