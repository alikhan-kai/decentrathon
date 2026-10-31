"""Fresh-process import diagnostic, not a workload or GPU performance test."""
import json
import sys
import time

variant = sys.argv[1]
marks = {}
def timed(name, fn):
    start = time.perf_counter()
    result = fn()
    marks[name] = time.perf_counter() - start
    return result

before = set(sys.modules)
overhead = timed("overhead_import_s", lambda: __import__("overhead"))
assert not ({"torch", "transformers", "trl", "soup_cli", "datasets"} & (set(sys.modules)-before))
assert not overhead.ENABLED and not overhead.DEVICE_MAP and not overhead.FAST_MAP
torch = timed("torch_import_s", lambda: __import__("torch"))
transformers = timed("transformers_import_s", lambda: __import__("transformers"))
kwargs = dict(device_map=variant in ("device", "both"), fast_map=variant in ("map", "both"))
sft = timed("soup_sft_import_s", lambda: __import__("soup_cli.trainer.sft", fromlist=["SFTTrainerWrapper"]))
before_install = (sft._map_text_sft_rows, sft.resolve_device_map, sft.SFTTrainerWrapper._setup_transformers)
timed("install_s", lambda: overhead.patch(**kwargs))
saved = (sft._map_text_sft_rows, sft.resolve_device_map, sft.SFTTrainerWrapper._setup_transformers)
assert (saved[0] is not before_install[0]) == kwargs["fast_map"]
assert (saved[1] is not before_install[1]) == kwargs["device_map"]
assert (saved[2] is not before_install[2]) == kwargs["device_map"]
overhead.patch(**kwargs)
assert saved == (sft._map_text_sft_rows, sft.resolve_device_map, sft.SFTTrainerWrapper._setup_transformers)
qwen = timed("qwen_materialize_s", lambda: __import__("transformers.models.qwen2.modeling_qwen2", fromlist=["Qwen2MLP"]))
trl = timed("trl_import_s", lambda: __import__("trl"))
trainer = timed("sft_trainer_materialize_s", lambda: trl.SFTTrainer)
from trl.trainer.sft_trainer import SFTTrainer
assert trainer is SFTTrainer
assert sft.SFTTrainerWrapper.__module__ == "soup_cli.trainer.sft"
assert all(getattr(mod, "__file__", None) is None or isinstance(mod.__file__, str)
           for name, mod in list(sys.modules.items()) if name.startswith("trl."))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(variant=variant, times=marks, cuda_initialized=torch.cuda.is_initialized(),
                      identities_preserved=True, trl_trainers=sorted(n for n in sys.modules if n.startswith("trl.trainer.")))))
