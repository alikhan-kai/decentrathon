"""Explicit packed-row adapter. Counters use tensor metadata only, never readback.

The released helper stays default-off; this caller certifies scoped full prefill.
All declined calls execute the supplied HF SDPA fallback once with original args.
"""
from collections import Counter
from .attention_gqa import try_t4_packed_attention

_COUNTS = Counter()
_HELPER_FLAGS = frozenset(('output_attentions', 'past_key_values', 'use_cache',
                           'position_bias', 'sliding_window', 'is_causal'))
# The packing admission/position constructor already establishes these positions.
_SCOPED_METADATA = frozenset(('position_ids', 'output_hidden_states', 'return_dict'))


def reset_stats():
    _COUNTS.clear()


def stats():
    rows=[]
    for (layer, qs, ks, vs, dtype, qg, kg, vg, outcome, reason), count in _COUNTS.items():
        rows.append(dict(layer=layer, query_shape=qs, key_shape=ks, value_shape=vs,
                         dtype=dtype, q_requires_grad=qg, k_requires_grad=kg,
                         v_requires_grad=vg, outcome=outcome, reason=reason, count=count))
    return dict(calls=sum(_COUNTS.values()),
                dispatched=sum(v for k,v in _COUNTS.items() if k[-2]=='dispatched'),
                fallbacks=sum(v for k,v in _COUNTS.items() if k[-2]=='fallback'),
                metadata=rows)


def row_attention(module, query, key, value, attention_mask, *, fallback,
                  dropout=0.0, scaling=None, **kwargs):
    from . import packed
    active=packed._ACTIVE.get()
    in_scope=active is not None and id(module) in active[0]
    unknown=set(kwargs)-_HELPER_FLAGS-_SCOPED_METADATA
    result=None
    if not in_scope:
        reason='outside-packed-scope'
    elif (kwargs.get('output_hidden_states') is not None and kwargs['output_hidden_states'] is not False) or (kwargs.get('return_dict') is not None and kwargs['return_dict'] is not True):
        reason='unsupported-output-metadata'
    elif unknown:
        reason='unsupported-kwargs:' + ','.join(sorted(unknown))
    else:
        flags={k:v for k,v in kwargs.items() if k in _HELPER_FLAGS}
        result=try_t4_packed_attention(module, query, key, value, attention_mask,
                                      enabled=True, full_prefill=True,
                                      dropout=dropout, scaling=scaling, **flags)
        if result is not None:
            reason='accepted'
        elif not key.requires_grad:
            reason='helper-declined:k-no-grad'  # metadata observation, not exclusive cause
        else:
            reason='helper-declined'
    outcome='dispatched' if result is not None else 'fallback'
    _COUNTS[(getattr(module,'layer_idx',None),tuple(query.shape),tuple(key.shape),
             tuple(value.shape),str(query.dtype),query.requires_grad,key.requires_grad,
             value.requires_grad,outcome,reason)]+=1
    if result is not None:
        return result
    return fallback(module,query,key,value,attention_mask,dropout=dropout,
                    scaling=scaling,**kwargs)
