"""Startup overhead of `soup train`; model, data, batches and hyperparameters are unchanged.

1. DEVICE_MAP: on a CUDA run, resolve_device_map's "auto" becomes {"": 0}, which skips accelerate's
   infer_auto_device_map. The harness exposes GPU 0 only, so the placement is the same.
2. FAST_MAP: build the tokenized dataset in memory instead of Dataset.from_list(...).map(...)
   (no Arrow write). Same rows and columns; test_overhead.py compares it with Soup's own function.

Derived from feat/overhead (Alikhan-kai). Each flag is separate; a failed import is an error.
"""

ENABLED = True
DEVICE_MAP = True
FAST_MAP = True
_orig_map_rows = None
_patched = False


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
        results = []
        for i, row in enumerate(rows):
            try:
                formatted = format_row(row)
                labels = formatted.get("labels")
                if labels is not None:
                    ensure_causal_loss_target(labels, max_length=max_length)
            except NoCausalLossTargetError as exc:
                raise ValueError(f"{split} row {i + 1}: {exc}") from exc
            results.append(formatted)
        if not results:
            return Dataset.from_dict({})
        keys = [k for k in results[0] if k != "messages"]
        return Dataset.from_dict({k: [d[k] for d in results] for k in keys})

    sft._map_text_sft_rows = fast_map_text_sft_rows


def patch():
    global _patched
    if _patched:
        return
    from soup_cli.trainer import sft

    if DEVICE_MAP:
        _patch_device_map(sft)
    if FAST_MAP:
        _patch_fast_map(sft)
    _patched = True
