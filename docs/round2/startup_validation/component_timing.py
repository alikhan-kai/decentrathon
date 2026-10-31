"""External diagnostic timers only. No cache APIs, RNG changes, or extra work.

Times are inclusive/nested and must not be summed. Wrappers call the captured
real implementation exactly once, preserve its result/error, and restore it.
Do not ship this module as part of the solution or compare it to an uninstrumented
control. It does not disable fingerprints or warm/delete/create compiler caches.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
from time import perf_counter


class Timings:
    def __init__(self):
        self.records = {}
        self.undo = []

    @contextmanager
    def span(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            entry = self.records.setdefault(name, {"calls": 0, "seconds": 0.0})
            entry["calls"] += 1
            entry["seconds"] += elapsed

    def wrap(self, owner, name, label):
        descriptor = inspect.getattr_static(owner, name)
        original = getattr(owner, name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            with self.span(label):
                return original(*args, **kwargs)
        if isinstance(descriptor, classmethod):
            def bound(cls, *args, **kwargs):
                with self.span(label):
                    return descriptor.__func__(cls, *args, **kwargs)
            replacement = classmethod(wraps(descriptor.__func__)(bound))
        elif isinstance(descriptor, staticmethod):
            replacement = staticmethod(wrapped)
        else:
            replacement = wrapped
        setattr(owner, name, replacement)
        self.undo.append((owner, name, descriptor))

    def restore(self):
        for owner, name, original in reversed(self.undo):
            setattr(owner, name, original)
        self.undo.clear()

    def map_components(self, candidate):
        from datasets import Dataset
        self.wrap(Dataset, "from_list", "initial_arrow_from_list")
        self.wrap(Dataset, "from_dict", "arrow_from_dict_inclusive")
        self.wrap(Dataset, "map", "dataset_map_inclusive")
        self.wrap(candidate, "_supported_formatter", "formatter_guard")
        self.wrap(candidate, "_supported_schema", "normalized_schema_guard")
        self.wrap(candidate, "_map_normalized", "normalized_map_core_inclusive")

    def setup_components(self, candidate):
        from soup_cli.trainer import sft
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.integrations import accelerate
        self.map_components(candidate)
        self.wrap(sft, "resolve_device_map", "soup_device_resolver")
        self.wrap(sft, "_map_text_sft_rows", "text_map_total")
        self.wrap(sft.SFTTrainerWrapper, "_setup_transformers", "model_setup_total")
        self.wrap(AutoTokenizer, "from_pretrained", "tokenizer_load")
        self.wrap(AutoModelForCausalLM, "from_pretrained", "causal_model_load")
        self.wrap(accelerate, "get_balanced_memory", "balanced_memory")
        self.wrap(accelerate, "infer_auto_device_map", "infer_auto_device_map")
