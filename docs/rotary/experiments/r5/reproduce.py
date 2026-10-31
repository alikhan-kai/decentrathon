"""Stage the rejected RoPE experiment outside the checkout; run only on a T4."""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parent
target = args.out.resolve()
if target.exists() or source.parents[3] in target.parents:
    parser.error("--out must be a new directory outside this experiment")
(target / "solution").mkdir(parents=True)
(target / "solution/__init__.py").write_text("")
for name in ("baseline.py", "baseline512.py", "benchmark.py"):
    shutil.copy2(source / name, target / name)
shutil.copy2(source / "packed_candidate.py", target / "solution/rotary.py")
shutil.copy2(source / "solution/test_rotary.py", target / "solution/test_rotary.py")
print(target)
print("CUDA_VISIBLE_DEVICES=0 python -m pytest solution/test_rotary.py -q")
print("CUDA_VISIBLE_DEVICES=0 python benchmark.py > benchmark.json")
