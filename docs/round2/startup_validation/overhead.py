"""Two independent, disabled startup experiments for Soup 0.77.0.

Idea attribution: Alikhan-kai, 44d1a47; separation/refinement: iencislennon,
1fea4ce4774d4660467438cb6340f41eddae4967. This implementation deliberately
retains Arrow input normalization and does not use the raw-dictionary donor.
No dependency replacements, import-time patches, cache management or timers.
"""
from contextvars import ContextVar
from functools import wraps
from types import CodeType, FunctionType, MappingProxyType

ENABLED = False
DEVICE_MAP = False
FAST_MAP = False
SUPPORTED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_fixed_setup = ContextVar("startup_fixed_qwen_setup", default=False)
_installed = set()


def _supported_setup(cfg, tcfg):
    # Only the known 4-bit text task; never infer model identity from a device.
    from soup_cli.config.schema import SoupConfig, TrainingConfig
    return (
        type(cfg) is SoupConfig and type(tcfg) is TrainingConfig
        and cfg.training is tcfg
        and cfg.base == SUPPORTED_MODEL and cfg.task == "sft"
        and cfg.backend == "transformers"
        and cfg.data.format == "alpaca" and cfg.data.max_length == 512
        and tcfg.quantization == "4bit" and tcfg.quantization_aware is False
        and tcfg.rope_scaling_type is None
    )


def _eligible_facts(*, cuda_build, available, visible_count, current_device,
                    distributed, env, free_bytes, total_bytes):
    """Pure policy, separately testable without pretending a CPU is a GPU."""
    return (
        cuda_build and available and visible_count == 1 and current_device == 0
        and not distributed
        and not any(key in env for key in ("LOCAL_RANK", "RANK", "LOCAL_WORLD_SIZE"))
        and env.get("WORLD_SIZE", "1") == "1"
        # Conservative capacity envelope, still requires parent placement gate.
        and total_bytes >= 14 * 1024**3 and free_bytes >= 8 * 1024**3
    )


def _cuda0_ready(device):
    if not _fixed_setup.get():
        return False
    import os
    import torch
    if not (type(device) is str or type(device) is torch.device):
        return False
    if str(device) not in ("cuda", "cuda:0"):
        return False
    if torch.version.cuda is None:
        return False
    if any(k in os.environ for k in ("LOCAL_RANK", "RANK", "LOCAL_WORLD_SIZE")):
        return False
    if os.environ.get("WORLD_SIZE", "1") != "1":
        return False
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            return False
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed or torch.cuda.current_device() != 0:
            return False
        free, total = torch.cuda.mem_get_info(0)
        return _eligible_facts(
            cuda_build=bool(torch.version.cuda), available=True, visible_count=1,
            current_device=0, distributed=distributed, env=os.environ,
            free_bytes=free, total_bytes=total,
        )
    except (RuntimeError, AssertionError):
        # Failure to establish the extra optimization preconditions must not
        # replace the original loader's diagnostics.
        return False


def _resolve(original, device, eligible=_cuda0_ready):
    result = original(device)  # Preserve original exceptions and distributed maps.
    if type(result) is str and result == "auto" and eligible(device):
        return {"": 0}
    return result


def _make_setup(original):
    @wraps(original)
    def setup(self, cfg, tcfg):
        token = _fixed_setup.set(
            _supported_setup(cfg, tcfg)
            and not getattr(self, "deepspeed_config", None)
            and not getattr(self, "fsdp_config", None)
            and not getattr(self, "_trust_remote_code", False)
        )
        try:
            return original(self, cfg, tcfg)
        finally:
            _fixed_setup.reset(token)
    return setup


def _nested_code(factory, name):
    code = getattr(factory, "__code__", None)
    if not isinstance(code, CodeType):
        return None
    return next((c for c in code.co_consts
                 if isinstance(c, CodeType) and c.co_name == name), None)


def _closure(function):
    try:
        return dict(zip(function.__code__.co_freevars,
                        (cell.cell_contents for cell in function.__closure__ or ())))
    except ValueError:  # An empty closure cell is not a supported formatter.
        return {}


# Pins are from loaded Soup 0.77.0 code, not external source. Bytecode is
# interpreter-specific: an unknown interpreter/build fails closed to Soup.map.
# No tokenizer calls, formatter calls, or dependency imports at module import.
def _code_fingerprint(function):
    import hashlib
    import json
    def encode(value):
        if type(value) is CodeType:
            return [value.co_code.hex(), value.co_exceptiontable.hex(),
                    value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount,
                    value.co_flags, value.co_names, value.co_varnames,
                    value.co_freevars, value.co_cellvars,
                    [encode(c) for c in value.co_consts]]
        if type(value) in (tuple, list):
            return [encode(v) for v in value]
        if type(value) is dict:
            return sorted((k, encode(v)) for k, v in value.items())
        if type(value) in (str, int, float, bool) or value is None:
            return [type(value).__name__, value]
        raise TypeError("unsupported code constant/default")
    value = [encode(function.__code__), encode(function.__defaults__),
             encode(function.__kwdefaults__)]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


_FORMATTER_PIN_RUNTIME = "CPython 3.13.5 / Soup 0.77.0"
_FORMATTER_PINS = MappingProxyType({'loss_mask._apply_template_with_mask': 'ce7c81c25d0add4d6190724c9ffe38f347fed568584790996a80109aaa71743e',
 'loss_mask._check_messages': '2d0a820596586641a6cc2dbc7e7302758f68389f0d22d071b2d280a6fea253cd',
 'loss_mask._coerce_int_list': 'ab9ecd6a2be1f62b54ad582eeb19855e970c707bcb7900b91625791295ceccd0',
 'loss_mask._extend_mask_to_eot': 'ac43c1f384844371d2d7a2cb6b56f482b3c6192da566c8dbbcb897de5dbd9f38',
 'loss_mask._has_user_turn': '6f1dccb975c123e99483cfa3a3fa1ed6042eeb6152f56792adb0e4f6e41f02bd',
 'loss_mask._is_mapping_like': '2384e4896897c68f2b4a6be5d9831532b13e44cf8e6aaf691351272c14243e75',
 'loss_mask._resolve_eos_token_id': 'aa667ef5f79d2f3c303ff015d3e36d2d70643c39d9b80f678e30e4ae35e11e16',
 'loss_mask._tokenize_only': 'f7538bdc3458739299c276c95fdf1e84cc17f05355982feb8d8bf3f77ed1ab13',
 'loss_mask._truncate': '11d03385b6b3f448c76084b583bea694414e8808743fd5777dba57036251681f',
 'loss_mask._validate_max_length': '33e3e713086cf71e759cd8dff5af3aecb61a89b54b7c6ef58b57647b9ac76676',
 'loss_mask.build_assistant_only_labels': '98fe7b0fa59bf961f1404e842b9ee9d862fadfbc52a179a53cc8b2696fed7d4d',
 'loss_mask.coerce_token_ids': '85cbeac7a0c804a13fa5d6ef2372994acde6d1d1b610484fe6d514ac6ec82c78',
 'loss_mask.ensure_causal_loss_target': '69dce0414a6f2801757ead3190b56d52c4de6926e880ba4f719499e54468bceb',
 'loss_mask.keep_only_the_last_assistant_turn': '07eee9de857d5eed4dc6b145a965a5a37241071ad517d6a7bcd146bed33b0c02',
 'sft_format._build_assistant_only_format_row': 'fe857f7c0a0ca3eb5b49de575a48fdba55a21ad4351780363fa471d1ff4d1f2f',
 'sft_format._wrap_dropping_arrow_nulls': '8d5476d836f81abbf16cf1a48ecf2b51090a5ca532ca0673631d487e9ea25b80'})
_formatter_reference = None


def _formatter_reference_valid():
    """Authenticate then freeze the factory/helper graph and exact globals.

    Recheck code, defaults and bindings on every eligibility decision, including
    in-place __code__ mutation. Copied globals never qualify, even with the same
    code/closure. Freeze references, not the third-party mutable dictionaries.
    """
    import builtins
    import dis
    import operator
    from collections.abc import Mapping
    from soup_cli.data import sft_format, loss_mask
    global _formatter_reference
    modules = {"sft_format": sft_format, "loss_mask": loss_mask}
    functions = {}
    try:
        for qualified, expected in _FORMATTER_PINS.items():
            module_name, name = qualified.split(".")
            module = modules[module_name]
            fn = getattr(module, name, None)
            if (type(fn) is not FunctionType or fn.__globals__ is not vars(module)
                    or fn.__closure__ is not None or _code_fingerprint(fn) != expected):
                return False
            functions[qualified] = fn
        if sft_format.build_assistant_only_labels is not functions["loss_mask.build_assistant_only_labels"]:
            return False
        if (type(sft_format._SPARSE_MESSAGE_KEYS) is not tuple
                or sft_format._SPARSE_MESSAGE_KEYS != ("tool_calls", "tool_call_id")
                or type(loss_mask.IGNORE_INDEX) is not int or loss_mask.IGNORE_INDEX != -100
                or loss_mask.Mapping is not Mapping or loss_mask.index is not operator.index
                or type(loss_mask._MISSING) is not object):
            return False
        bindings = []
        def globals_used(code):
            names = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"}
            for c in code.co_consts:
                if type(c) is CodeType:
                    names.update(globals_used(c))
            return names
        for qualified, fn in functions.items():
            if fn.__builtins__ is not vars(builtins):
                return False
            for name in sorted(globals_used(fn.__code__)):
                value = fn.__globals__.get(name, vars(builtins).get(name))
                # Module overrides of builtins would change formatter semantics.
                if name in vars(builtins) and value is not vars(builtins)[name]:
                    return False
                bindings.append((qualified, name, value))
        current = (MappingProxyType(functions),
                   tuple((name, vars(module)) for name, module in modules.items()),
                   tuple(bindings))
        if _formatter_reference is None:
            _formatter_reference = current
        frozen_functions, frozen_globals, frozen_bindings = _formatter_reference
        return (all(fn is frozen_functions[name] for name, fn in functions.items())
                and all(dictionary is dict(frozen_globals)[name] for name, dictionary in current[1])
                and len(bindings) == len(frozen_bindings)
                and all(a[:2] == b[:2] and a[2] is b[2]
                        for a, b in zip(bindings, frozen_bindings)))
    except (AttributeError, TypeError, ValueError, KeyError):
        return False


def _supported_formatter(format_row, max_length):
    # Inspect exact existing closure code; do not call the factory or formatter
    # to guess its behavior. Custom strategies/other modes stay on Dataset.map.
    import builtins
    from soup_cli.data import sft_format
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    if type(max_length) is not int or max_length != 512 or not _formatter_reference_valid():
        return False
    if type(format_row) is not FunctionType or format_row.__code__ is not _nested_code(
        _formatter_reference[0]["sft_format._wrap_dropping_arrow_nulls"], "wrapped"
    ):
        return False
    if (format_row.__globals__ is not vars(sft_format)
            or format_row.__builtins__ is not vars(builtins)):
        return False
    inner = _closure(format_row).get("inner")
    if type(inner) is not FunctionType or inner.__code__ is not _nested_code(
        _formatter_reference[0]["sft_format._build_assistant_only_format_row"], "format_row"
    ):
        return False
    if (inner.__globals__ is not vars(sft_format)
            or inner.__builtins__ is not vars(builtins)):
        return False
    captured = _closure(inner)
    tokenizer = captured.get("tokenizer")
    return (
        captured.get("max_length") == max_length
        and captured.get("include_eot") is False
        and captured.get("mask_history") is False
        and type(tokenizer) is Qwen2Tokenizer and tokenizer.is_fast
        and bool(tokenizer.chat_template)
    )


def _supported_schema(dataset):
    from datasets import Value
    if not len(dataset) or "messages" not in dataset.column_names:
        return False
    output_names = {"input_ids", "labels", "attention_mask"}
    if output_names.intersection(dataset.column_names):
        return False  # Dataset.map's overwrite/try_original_type is out of scope.
    for name, feature in dataset.features.items():
        if name != "messages" and not (
            type(feature) is Value and feature.dtype in
            ("null", "bool", "int64", "float64", "string")
        ):
            return False
    # Check all normalized rows, before calling format_row even once. No raw
    # dictionary schema guesses, especially for missing/sparse nested keys.
    for messages in dataset["messages"]:
        if type(messages) is not list or not messages:
            return False
        for message in messages:
            if type(message) is not dict or set(message) != {"role", "content"}:
                return False
            if type(message["content"]) is not str or message["role"] not in (
                "system", "user", "assistant"
            ):
                return False
    return True


def _map_normalized(dataset, *, format_row, split, max_length):
    """Private eligible-only core. Each call receives its own normalized row."""
    from datasets import Dataset, Features, Sequence, Value
    from soup_cli.data.loss_mask import NoCausalLossTargetError, ensure_causal_loss_target

    names = [name for name in dataset.column_names if name != "messages"]
    columns = {name: dataset.data.column(name) for name in names}
    features = {name: dataset.features[name] for name in names}
    # Matches the known formatter's insertion order and Dataset.map's optimized
    # integer types; retained columns keep their initial Arrow feature types.
    for name, dtype in (("input_ids", "int32"), ("labels", "int64"),
                        ("attention_mask", "int8")):
        columns[name] = []
        features[name] = Sequence(Value(dtype))
    for row_index, example in enumerate(dataset):
        try:
            formatted = format_row(example)
            labels = formatted.get("labels")
            if labels is not None:
                ensure_causal_loss_target(labels, max_length=max_length)
        except NoCausalLossTargetError as exc:
            raise ValueError(f"{split} row {row_index + 1}: {exc}") from exc
        # Never retry/replay on errors after formatting has begun.
        for name in ("input_ids", "labels", "attention_mask"):
            columns[name].append(formatted[name])
    return Dataset.from_dict(columns, features=Features(features))


def _make_map(original):
    @wraps(original)
    def map_rows(rows, *, format_row, split, max_length):
        if type(rows) is not list or any(type(row) is not dict for row in rows):
            return original(rows, format_row=format_row, split=split, max_length=max_length)
        if not _supported_formatter(format_row, max_length):
            return original(rows, format_row=format_row, split=split, max_length=max_length)
        from datasets import Dataset
        normalized = Dataset.from_list(rows)  # Required original Arrow normalization.
        if not _supported_schema(normalized):
            return original(rows, format_row=format_row, split=split, max_length=max_length)
        return _map_normalized(normalized, format_row=format_row,
                               split=split, max_length=max_length)
    return map_rows


def patch(*, device_map=None, fast_map=None):
    """Explicit options can be selected independently; apply() uses the constants.

    Installation is idempotent and additive. Use a fresh process to change a
    variant; this module does not silently uninstall another patch.
    """
    device_map = DEVICE_MAP if device_map is None else device_map
    fast_map = FAST_MAP if fast_map is None else fast_map
    if type(device_map) is not bool or type(fast_map) is not bool:
        raise TypeError("device_map and fast_map must be bool")
    if not device_map and not fast_map:
        return
    from importlib.metadata import version
    if version("soup-cli") != "0.77.0" or version("transformers") != "5.16.1":
        raise RuntimeError("startup candidate requires soup-cli 0.77.0 / transformers 5.16.1")
    if fast_map and version("datasets") not in ("4.8.5", "5.1.0"):
        raise RuntimeError("map candidate requires datasets 4.8.5 (gate) or 5.1.0 (CPU only)")
    from soup_cli.trainer import sft
    if device_map and "device_map" not in _installed:
        original = sft.resolve_device_map
        @wraps(original)
        def resolve(device):
            return _resolve(original, device)
        setup = _make_setup(sft.SFTTrainerWrapper._setup_transformers)
        sft.resolve_device_map = resolve
        sft.SFTTrainerWrapper._setup_transformers = setup
        _installed.add("device_map")
    if fast_map and "fast_map" not in _installed:
        sft._map_text_sft_rows = _make_map(sft._map_text_sft_rows)
        _installed.add("fast_map")
