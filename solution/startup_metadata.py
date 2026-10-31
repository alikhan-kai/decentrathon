"""Reuse Transformers' same-process distribution map for the TRL import.

No prepared metadata crosses runs. Every caller receives fresh lists; the
standard function is restored immediately after importing TRL, including errors.
"""
import importlib
import importlib.metadata
import sys


def prepare_trl_import():
    if "trl.import_utils" in sys.modules:
        return
    source = sys.modules.get("transformers.utils.import_utils")
    mapping = getattr(source, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if not isinstance(mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, list)
            and all(isinstance(item, str) for item in v) for k, v in mapping.items()):
        return
    snapshot = {key: list(value) for key, value in mapping.items()}
    original = importlib.metadata.packages_distributions

    def cached_mapping():
        return {key: list(value) for key, value in snapshot.items()}

    importlib.metadata.packages_distributions = cached_mapping
    try:
        importlib.import_module("trl")
    finally:
        importlib.metadata.packages_distributions = original
