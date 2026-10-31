"""Opt-in packed-row T4 experiment. No patches, registration, or import-time CUDA calls."""
from functools import lru_cache
from importlib.metadata import version

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import repeat_kv


@lru_cache(maxsize=1)
def _pinned_versions():
    return torch.__version__.split('+')[0] == '2.11.0' and version('transformers') == '5.16.1'


@lru_cache(maxsize=8)
def _is_t4(index):
    props = torch.cuda.get_device_properties(index)
    return (props.major, props.minor) == (7, 5) and 'T4' in props.name.upper().split()


def repeat_sdpa(query, key, value, *, dropout=0.0, scaling=None, is_causal=True):
    """Experiment primitive; repeats run INSIDE the caller's autocast context.

    HF's exact expand/reshape implementation is imported above. In the FP32-storage
    case repetition therefore materializes FP32 K/V before SDPA casts to FP16.
    Do not pre-cast or hoist repetition out of a timed region.
    """
    groups = query.shape[1] // key.shape[1]
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=dropout,
        is_causal=is_causal, scale=scaling, enable_gqa=False,
    )


def try_t4_packed_attention(
    module, query, key, value, attention_mask=None, *, enabled=False,
    full_prefill=False, output_attentions=False, past_key_values=None,
    use_cache=False, position_bias=None, sliding_window=None,
    dropout=0.0, scaling=None, is_causal=None,
):
    """Return HF-layout (output, None), or None to call the unchanged HF fallback.

    Caller must explicitly certify a complete independent packed row with
    full_prefill=True, propagate cache/mask/attention flags, and opt in. This helper
    is NOT a general replacement for sdpa_attention_forward. No backend is forced.
    Exceptions after dispatch are not swallowed. This is an unmeasured candidate.
    """
    if not (enabled and full_prefill and module.training and torch.is_grad_enabled()):
        return None
    if (attention_mask is not None or output_attentions or past_key_values is not None
            or use_cache or position_bias is not None or sliding_window is not None
            or getattr(module, 'sliding_window', None) is not None):
        return None
    if getattr(getattr(module, 'config', None), 'output_attentions', False):
        return None
    if not (is_causal if is_causal is not None else getattr(module, 'is_causal', True)):
        return None
    if any(x.ndim != 4 for x in (query, key, value)):
        return None
    length = query.shape[2]
    if (query.shape != (1, 12, length, 128) or length <= 1
            or key.shape != (1, 2, length, 128) or value.shape != key.shape
            or getattr(module, 'num_key_value_groups', None) != 6):
        return None
    if any(x.device != query.device or x.dtype != query.dtype or x.stride(-1) != 1
           for x in (query, key, value)):
        return None
    if query.device.type != 'cuda' or not (query.requires_grad and value.requires_grad):
        return None
    if query.dtype == torch.float32:
        if not (torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.float16):
            return None
    elif query.dtype != torch.float16:
        return None
    elif torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') != torch.float16:
        return None
    if not _pinned_versions() or not _is_t4(query.device.index):
        return None
    out = repeat_sdpa(query, key, value, dropout=dropout, scaling=scaling, is_causal=True)
    return out.transpose(1, 2).contiguous(), None
