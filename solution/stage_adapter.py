"""Explicit staged registry adapter; original registry list and ordering retained.

No model patch at import. The new stage calls this wrapper only from apply().
All dependencies live inside solution, including when only that folder is copied.
"""
import atexit
import functools
import json
_SAVE_HANDLE = None
_REPORT_REGISTERED = False


def adapt_apply(original):
    if getattr(original,'_combined_r1',False):
        return original
    @functools.wraps(original)
    def apply():
        global _SAVE_HANDLE, _REPORT_REGISTERED
        original()
        from . import packed
        packed.patch()
        from .stage_options import ENABLE_SAVE
        if ENABLE_SAVE:
            from .swiglu_pack import install_saved_activation_packing
            _SAVE_HANDLE=install_saved_activation_packing(enable=True,first_order_training=True)
            if not _SAVE_HANDLE.enabled:
                # Not a crash: the unmodified MLP path trains the same task, only slower / with more memory.
                print('SAVED_ACTIVATION_PACKING DISABLED, ordinary MLP path in use: '+_SAVE_HANDLE.reason,
                      flush=True)
        if not _REPORT_REGISTERED:
            from . import attention_adapter  # load while the staged package is present
            atexit.register(_report_stats)
            _REPORT_REGISTERED = True
    apply._combined_r1=True
    return apply


def install():
    """Optional process-local adapter for an unembedded source variant."""
    import solution
    solution.apply=adapt_apply(solution.apply)


def stats():
    from . import attention_adapter
    return {'attention': attention_adapter.stats(),
            'save': None if _SAVE_HANDLE is None else {
                'enabled': _SAVE_HANDLE.enabled, 'reason': _SAVE_HANDLE.reason,
                'packed_calls': _SAVE_HANDLE.packed_calls,
                'remat_calls': _SAVE_HANDLE.remat_calls,
                'fallback_calls': dict(_SAVE_HANDLE.fallback_calls)}}


def _report_stats():
    # Existing Python counters / metadata only; no tensor contents or GPU sync.
    print('COMBINED_STAGE_STATS '+json.dumps(stats(),separators=(',',':')),flush=True)
