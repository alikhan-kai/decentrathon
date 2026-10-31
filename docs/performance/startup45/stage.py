"""Copy a pinned Git runtime into a new control/candidate pair. Never runs CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HELPERS = {"metadata": "startup_metadata.py", "lazy-bnb": "lazy_bnb_compile.py"}
ROOT = Path(__file__).resolve().parents[3]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def build(repo, revision, destination, kind, train, evaluation):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("provide a full pinned commit SHA, not a moving branch")
    if git(repo, "rev-parse", revision + "^{commit}").decode().strip() != revision:
        raise ValueError("commit identity mismatch")
    destination = Path(destination).resolve()
    if destination.exists() or any((p / ".git").exists()
                                  for p in destination.parents):
        raise ValueError("output must be a new directory outside every checkout")
    helper = ROOT / "solution" / HELPERS[kind]
    helper_bytes = helper.read_bytes()
    data = {"data/train.jsonl": Path(train).read_bytes(),
            "data/eval_public.jsonl": Path(evaluation).read_bytes()}
    files = {}
    for record in git(repo, "ls-tree", "-r", revision).decode().splitlines():
        identity, name = record.split("\t", 1)
        mode, object_type, blob = identity.split()
        path = Path(name)
        if path.parts[0] not in {"solution", "harness", "task", "tests", "LICENSE", "NOTICE"}:
            continue
        if path.is_absolute() or ".." in path.parts or mode not in {"100644", "100755"}:
            raise ValueError("unsupported source entry: " + name)
        files[name] = git(repo, "cat-file", "blob", blob)
    init_name = "solution/__init__.py"
    if init_name not in files or "solution/" + HELPERS[kind] in files:
        raise ValueError("unsupported or already staged source")
    candidate = dict(files)
    candidate["solution/" + HELPERS[kind]] = helper_bytes
    old_cache_disabled = False
    if kind == "metadata":
        name = "solution/overhead.py"
        overhead = candidate[name]
        if b"PACKAGES_CACHE" in overhead:
            if overhead.count(b"PACKAGES_CACHE = True\n") == 1:
                candidate[name] = overhead.replace(b"PACKAGES_CACHE = True\n",
                                                   b"PACKAGES_CACHE = False\n", 1)
                old_cache_disabled = True
            elif overhead.count(b"PACKAGES_CACHE = False\n") != 1:
                raise ValueError("unknown persistent metadata cache; review before staging")
        elif b"_patch_packages_distributions" in overhead:
            raise ValueError("unrecognized older metadata cache")
        suffix = """

# Explicit isolated startup experiment. Existing apply adapters stay in order.
from . import startup_metadata as _startup_metadata
_startup_metadata.ENABLED = True
apply = _startup_metadata.wrap_apply(apply)
"""
    else:
        suffix = """

# Explicit isolated startup experiment. Existing apply adapters stay in order.
from . import lazy_bnb_compile as _startup_lazy_bnb
_startup_previous_apply = apply
def apply():
    with _startup_lazy_bnb.install(enabled=True):
        return _startup_previous_apply()
"""
    candidate[init_name] += suffix.encode()
    changed = sorted(n for n in candidate if candidate[n] != files.get(n))
    protected = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()
                 if n.split("/")[0] in {"harness", "task", "tests"}}
    for label, content in (("control", files), ("candidate", candidate)):
        for name, body in {**content, **data}.items():
            p = destination / label / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
    manifest = {"base_sha": revision, "variant": kind, "changed_files": changed,
                "old_cache_disabled": old_cache_disabled,
                "protected": protected, "control": hashes(destination / "control"),
                "candidate": hashes(destination / "candidate"),
                "gpu_run": False, "helpers_disabled_in_checkout": True}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=HELPERS, required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, required=True)
    a = p.parse_args()
    result = build(a.repo, a.commit, a.output, a.variant, a.train_file, a.eval_file)
    print(json.dumps({k: result[k] for k in ("base_sha", "variant", "changed_files")}))
