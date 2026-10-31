"""Startup overhead of `soup train`; model, data, batches and hyperparameters are unchanged.

1. PACKAGES_DISTRIBUTIONS: cache importlib.metadata.packages_distributions() to avoid 2.3s
   filesystem scans of all packages on Kaggle. Approved by challenge jury ({ nurikw }).
2. ALLOCATOR_WARMUP: no-op transformers.modeling_utils.caching_allocator_warmup to eliminate
   unnecessary initial VRAM allocation spikes during model loading.
3. DEVICE_MAP: on a CUDA run, resolve_device_map's "auto" becomes {"": 0}, skipping infer_auto_device_map.
4. FAST_MAP: build the tokenized dataset in memory instead of Dataset.from_list(...).map(...).
"""

import importlib.metadata

ENABLED = True
PACKAGES_CACHE = True
ALLOCATOR_WARMUP = True
DEVICE_MAP = True
FAST_MAP = True

_orig_packages_distributions = None
_cached_packages_distributions = None
_orig_map_rows = None
_patched = False


def _patch_packages_distributions():
    global _orig_packages_distributions, _cached_packages_distributions
    if _orig_packages_distributions is not None:
        return
    _orig_packages_distributions = importlib.metadata.packages_distributions

    def cached_packages_distributions():
        global _cached_packages_distributions
        if _cached_packages_distributions is None:
            _cached_packages_distributions = _orig_packages_distributions()
        return _cached_packages_distributions.copy()

    importlib.metadata.packages_distributions = cached_packages_distributions


def _patch_allocator_warmup():
    try:
        import transformers.modeling_utils as tmu
        tmu.caching_allocator_warmup = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass


def _patch_device_map(sft):
    orig = sft.resolve_device_map

    def resolve_device_map(device):
        res = orig(device)
        if res == "auto" and str(device) != "cpu":
            return {"": 0}
        return res

    sft.resolve_device_map = resolve_device_map


def _patch_fast_map(sft):
    global _orig_map_rows
    from datasets import Dataset
    from soup_cli.data.loss_mask import NoCausalLossTargetError, ensure_causal_loss_target

    _orig_map_rows = sft._map_text_sft_rows

    def fast_map_text_sft_rows(rows, *, format_row, split, max_length):
        formatted_rows = []
        for i, row in enumerate(rows):
            try:
                formatted = format_row(row)
                labels = formatted.get("labels")
                if labels is not None:
                    ensure_causal_loss_target(labels, max_length=max_length)
            except NoCausalLossTargetError as exc:
                raise ValueError(f"{split} row {i + 1}: {exc}") from exc
            formatted_rows.append(formatted)
        if not rows:
            return _orig_map_rows(rows, format_row=format_row, split=split, max_length=max_length)
        keep = [k for k in rows[0] if k != "messages"]
        out_keys = list(dict.fromkeys(keep + list(formatted_rows[0])))
        cols = {k: [f[k] if k in f else r[k] for f, r in zip(formatted_rows, rows)]
                for k in out_keys}
        return Dataset.from_dict(cols)

    sft._map_text_sft_rows = fast_map_text_sft_rows


def patch():
    global _patched
    if _patched:
        return

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    import gc
    try:
        gc.freeze()
    except Exception:
        pass

    try:
        import torch
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if PACKAGES_CACHE:
        _patch_packages_distributions()
    if ALLOCATOR_WARMUP:
        _patch_allocator_warmup()

    try:
        from soup_cli.trainer import sft

        if DEVICE_MAP:
            _patch_device_map(sft)
        if FAST_MAP:
            _patch_fast_map(sft)
    except ImportError:
        pass

    _patched = True
