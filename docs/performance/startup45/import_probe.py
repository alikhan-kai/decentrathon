"""CPU-only import-chain measurement, not a training benchmark."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("arm", choices=("control", "metadata", "lazy-bnb"))
a = p.parse_args()
import torch
import transformers
if torch.version.cuda is not None:
    raise SystemExit("this probe is restricted to CPU-only Torch")
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))  # same order as the harness: after Transformers
from solution import startup_metadata, lazy_bnb_compile
original_metadata, original_compile = metadata.packages_distributions, torch.compile
startup_metadata.ENABLED = a.arm == "metadata"
scope = (startup_metadata.reuse_package_mapping() if a.arm == "metadata" else
         lazy_bnb_compile.install(enabled=True) if a.arm == "lazy-bnb" else nullcontext())
start = time.perf_counter()
with scope:
    metadata_admitted = metadata.packages_distributions is not original_metadata
    from transformers.models.qwen2 import modeling_qwen2
    import trl
    from transformers import Trainer
    from trl import SFTTrainer
    from soup_cli.trainer import sft
    import soup_cli.cli
elapsed = time.perf_counter() - start
assert metadata.packages_distributions is original_metadata
assert torch.compile is original_compile
assert not torch.cuda.is_initialized()
from trl import import_utils as trl_imports
assert trl_imports.PACKAGE_DISTRIBUTION_MAPPING == original_metadata()
assert (a.arm == "metadata") == metadata_admitted
stats = lazy_bnb_compile.stats()
if a.arm == "lazy-bnb":
    assert len(stats["functions"]) == 4, stats
    assert all(x["calls"] == x["constructions"] == 0 for x in stats["functions"]), stats
print(json.dumps({"arm": a.arm, "import_s": elapsed,
    "scope": "CPU dependency imports through Soup CLI; no apply/model/training",
    "metadata_admitted": metadata_admitted, "lazy_stats": stats,
    "inductor_loaded": "torch._inductor.compile_fx" in sys.modules,
    "restored": True, "mapping_equal": True, "cuda_initialized": False,
    "source_sha256": {n: hashlib.sha256((root / "solution" / n).read_bytes()).hexdigest()
                      for n in ("startup_metadata.py", "lazy_bnb_compile.py")},
    "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                 "transformers": transformers.__version__, "trl": trl.__version__}}))
