"""Run the isolated head-loss regressions against a candidate source file.

Copy this directory, including head_loss.correct-r6.py, rewind_reference.py,
NOTICE and LICENSE, to Kaggle; then run: python run_smoke.py --device cuda
No solution registry import or installation is needed. CPU mode uses Triton's
interpreter; CUDA mode runs the compiled kernel and actual CUDA GradScaler.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--head-loss", type=Path, default=root / "head_loss.py")
    args = parser.parse_args()
    candidate = args.head_loss.resolve()
    if not candidate.is_file():
        parser.error(f"candidate source does not exist: {candidate}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TRITON_CACHE_DIR"] = str(root / "triton-cache")
    if args.device == "cpu":
        env["TRITON_INTERPRET"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("TRITON_INTERPRET", None)
    command = [sys.executable, "-B", str(root / "test_head_loss_regression.py"),
               "--head-loss", str(candidate), "--device", args.device]
    print(f"Testing {candidate} on {args.device}", flush=True)
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
