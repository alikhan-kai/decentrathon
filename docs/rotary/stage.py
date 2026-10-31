"""Make an isolated measurement copy with an explicit registry selection.

Run from a clean checkout, then execute the printed commands on one T4.
This does not change the team's source checkout or install dependencies.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("team", "rotary", "combined"), required=True)
    parser.add_argument("--head-correction", action="store_true",
                        help="Explicitly stage the separate loss-gradient correction")
    parser.add_argument("--out", type=Path, required=True, help="New, nonexisting directory")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    destination = args.out.expanduser().resolve()
    if destination.exists() or source == destination or source in destination.parents:
        parser.error("--out must be a new directory outside the source checkout")
    entry = source / "solution/__init__.py"
    original = 'MODULES = ["swiglu", "head_loss", "rmsnorm", "nf4", "lora"]'
    text = entry.read_text()
    if text.count(original) != 1:
        parser.error("Team registry changed; reconcile the explicit selection first")
    if args.head_correction:
        expected = "9692acc5d32318872f7b50b67e4e4c8d61d86a33dea2458f06443eddfd1c3934"
        if hashlib.sha256((source / "solution/head_loss.py").read_bytes()).hexdigest() != expected:
            parser.error("Team head changed; review the correction against that version first")
    extra = {"team": "", "rotary": ', "rotary"',
             "combined": ', "rotary", "residual_norm"'}[args.variant]
    destination.mkdir(parents=True)
    for directory in ("solution", "harness", "tests", "task"):
        for path in sorted((source / directory).glob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml"):
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(source / name, destination / name)
    if args.head_correction:
        shutil.copy2(Path(__file__).resolve().parent / "head_correction/head_loss.py",
                     destination / "solution/head_loss.py")
    (destination / "solution/__init__.py").write_text(
        text.replace(original, original[:-1] + extra + "]"))
    for name, enabled in (("rotary", args.variant != "team"),
                          ("residual_norm", args.variant == "combined")):
        path = destination / "solution" / f"{name}.py"
        lines = path.read_text().splitlines(keepends=True)
        assert sum(line.startswith("ENABLED = ") for line in lines) == 1
        path.write_text("".join(f"ENABLED = {enabled}\n" if line.startswith("ENABLED = ")
                                else line for line in lines))
    manifest = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged {args.variant}: {destination}")
    print(f"cd {shlex.quote(str(destination))}")
    print("export CUDA_VISIBLE_DEVICES=0")
    print("python -m pytest solution -q")
    print("python task/prepare_data.py")
    print("python harness/run_task.py --repeats 2")
    print("python harness/check_rules.py")


if __name__ == "__main__":
    main()
