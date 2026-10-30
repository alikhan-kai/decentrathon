"""Overhead optimizations for soup train startup and dataset tokenization.

Reduces non-training overhead without changing model, data, batch order, or hyperparameters:
1. Fast in-memory tokenized dataset construction: replaces Hugging Face Dataset.from_list(rows).map(...)
   (which serializes to Arrow tables on disk and runs single-threaded subprocesses) with direct
   in-memory list comprehension and Dataset.from_dict. Saves ~1.5 - 2.0s of disk I/O.
2. Direct GPU device mapping: replaces device_map="auto" with device_map={"": 0} on single-GPU launches,
   bypassing accelerate's infer_auto_device_map and per-layer dispatch hooks. Saves ~1.5s.
"""

from typing import Any

ENABLED = True
_patched = False


def patch_soup_sft():
    """Patch Soup's SFT trainer wrapper to eliminate dataset mapping and device map overhead."""
    try:
        from soup_cli.trainer import sft
    except (ImportError, ModuleNotFoundError):
        return

    # Patch 1: Fast in-memory dataset mapping
    orig_map_rows = getattr(sft, "_map_text_sft_rows", None)
    if orig_map_rows is not None:
        from datasets import Dataset
        from soup_cli.data.loss_mask import (
            NoCausalLossTargetError,
            ensure_causal_loss_target,
        )

        def fast_map_text_sft_rows(
            rows: list[dict],
            *,
            format_row: Any,
            split: str,
            max_length: int,
        ) -> Any:
            def checked_format_row(example: dict, row_index: int) -> dict:
                try:
                    formatted = format_row(example)
                    labels = formatted.get("labels")
                    if labels is not None:
                        ensure_causal_loss_target(labels, max_length=max_length)
                    return formatted
                except NoCausalLossTargetError as exc:
                    raise ValueError(f"{split} row {row_index + 1}: {exc}") from exc

            # Fast in-memory processing instead of Dataset.from_list().map(...)
            results = [checked_format_row(row, i) for i, row in enumerate(rows)]
            if not results:
                return Dataset.from_dict({})
            keys = [k for k in results[0].keys() if k != "messages"]
            cols = {k: [d[k] for d in results] for k in keys}
            return Dataset.from_dict(cols)

        sft._map_text_sft_rows = fast_map_text_sft_rows

    # Patch 2: Direct GPU device map on single-GPU runs
    orig_resolve_device_map = getattr(sft, "resolve_device_map", None)
    if orig_resolve_device_map is not None:

        def fast_resolve_device_map(device: str):
            res = orig_resolve_device_map(device)
            # When resolve_device_map returns "auto" for a single cuda device,
            # return {"": 0} to bypass accelerate's infer_auto_device_map
            if res == "auto" and str(device) != "cpu":
                return {"": 0}
            return res

        sft.resolve_device_map = fast_resolve_device_map


def patch():
    """Idempotent entry point called by solution.apply()."""
    global _patched
    if not _patched:
        patch_soup_sft()
        _patched = True
