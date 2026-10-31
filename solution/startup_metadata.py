"""Default-off reuse for the audited fresh Transformers 5.16.1 harness only.

Assume unchanged distribution metadata, resolver behavior, cwd, and an unpolluted
Transformers mapping SINCE ITS ORIGINAL CONSTRUCTION. The harness's subsequent
repo-root insertion must expose no distributions. Entry guards cannot prove that
history: this is not general stale-map detection. No validation scan is performed.
"""
from contextlib import contextmanager
import functools
import importlib.machinery as machinery
import importlib.metadata as metadata
import os
import sys
import threading
import types

ENABLED = False
_CONTEXT_LOCK = threading.RLock()
_IN_CONTEXT = False
_OWNER = object()


def _copy_mapping(mapping):
    return {key: list(value) for key, value in mapping.items()}


def _stdlib_function(function, name):
    function = getattr(function, "__func__", function)
    return (type(function) is types.FunctionType
            and function.__globals__ is vars(metadata)
            and function.__code__.co_filename == metadata.__file__
            and function.__code__.co_name == name
            and not hasattr(function, "__wrapped__"))


def _standard_resolvers():
    # Reject ordinary replacements, including functools.wraps hooks. This is
    # conservative provenance checking, not a defense against forged code objects.
    for function, name in (
        (metadata.distributions, "distributions"),
        (metadata.Distribution.discover, "discover"),
        (metadata.Distribution._discover_resolvers, "_discover_resolvers"),
        (metadata.MetadataPathFinder.find_distributions, "find_distributions"),
        (metadata.MetadataPathFinder._search_paths, "_search_paths"),
    ):
        if not _stdlib_function(function, name):
            return False
    resolvers = [(finder, getattr(finder, "find_distributions", None))
                 for finder in sys.meta_path
                 if getattr(finder, "find_distributions", None) is not None]
    if len(resolvers) != 1 or resolvers[0][0] is not machinery.PathFinder:
        return False
    function = resolvers[0][1]
    return (type(function) is types.FunctionType
            and function.__module__ == "_frozen_importlib_external"
            and function.__code__.co_filename == "<frozen importlib._bootstrap_external>"
            and function.__code__.co_name == "find_distributions"
            and not hasattr(function, "__wrapped__"))


def _seed_mapping():
    transformers = sys.modules.get("transformers")
    source = sys.modules.get("transformers.utils.import_utils")
    if getattr(transformers, "__version__", None) != "5.16.1" or source is None:
        return None
    mapping = getattr(source, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if type(mapping) is not dict or not mapping:
        return None
    if any(type(key) is not str or type(value) is not list
           or any(type(item) is not str for item in value)
           for key, value in mapping.items()):
        return None
    return _copy_mapping(mapping)


@contextmanager
def reuse_package_mapping():
    """Serialize scopes; nested scopes retain the outer decision and baseline."""
    global _IN_CONTEXT
    with _CONTEXT_LOCK:
        if _IN_CONTEXT:
            yield
            return
        _IN_CONTEXT = True
        try:
            original = metadata.packages_distributions
            seed = (_seed_mapping() if ENABLED
                    and _stdlib_function(original, "packages_distributions")
                    and _standard_resolvers() else None)
            if seed is None:
                yield
                return
            initial_path = tuple(sys.path)
            initial_finders = tuple(sys.meta_path)
            initial_cwd = os.getcwd()
            active = True

            @functools.wraps(original)
            def cached():
                if (active and tuple(sys.path) == initial_path
                        and tuple(sys.meta_path) == initial_finders
                        and os.getcwd() == initial_cwd and _standard_resolvers()):
                    return _copy_mapping(seed)
                return original()

            metadata.packages_distributions = cached
            try:
                yield
            finally:
                active = False
                if metadata.packages_distributions is cached:
                    metadata.packages_distributions = original
                # Do not mutate seed: an unrelated reader can already be copying
                # it. Escaped wrappers delegate; GC owns the snapshot lifetime.
        finally:
            _IN_CONTEXT = False


def wrap_apply(original):
    if getattr(original, "_startup_metadata_owner", None) is _OWNER:
        return original

    @functools.wraps(original)
    def apply(*args, **kwargs):
        with reuse_package_mapping():
            return original(*args, **kwargs)

    apply._startup_metadata_owner = _OWNER
    return apply
