"""Attention of the packed batch [1, N, H] (see packed.py): three ways to keep rows apart (PROTOTYPE).

Packed q, k, v are [1, Hq, N, D] / [1, Hkv, N, D] after rotary; the rows of the batch sit one after the other and must
not attend to each other. VARIANT is switched from code (the ab_attn notebook rewrites the VARIANT line; nothing is read
from the environment):
  "A"  one transformers `sdpa_attention_forward` call per row (the behaviour of commit 0aa53cc / safe-v1). With
       attention_mask=None that function passes enable_gqa=True to SDPA; transformers' own comment on use_gqa_in_sdpa says
       the grouped-query kernel is the flash one, and flash needs sm80, so on a T4 (sm75) SDPA falls back to the math
       backend (bmm + softmax + where ...): measured 5820 kernel launches/step in attention (docs/measurements/
       pack_and_idle_f8948e8/launch_table.txt), GPU busy 0.448.
  "A2" repeat_kv ONCE on the packed k, v (Hkv -> Hq heads), then one F.scaled_dot_product_attention(q_i, k_i, v_i,
       is_causal=True) per row, WITHOUT enable_gqa and without a mask, so the mem-efficient backend can be chosen.
  "B"  one SDPA call for the whole packed batch with a block-diagonal causal additive float mask [1, 1, N, N] in the
       dtype of q (built once per forward and dtype, shared by all layers), repeat_kv once. packed.py appends
       `extra_tokens(N)` filler tokens so that N % 8 == 0 (they form a block of their own, are never read by real tokens,
       carry no loss and are dropped before the final norm).

Backend requirements (from the installed sources: torch 2.11 headers ATen/native/transformers/cuda/mem_eff_attention/
kernel_forward.h, transformers 5.16.1 integrations/sdpa_attention.py; sm75 has no flash kernel):
  * mem-efficient: fp16/fp32 tensors, 4-D [B, H, L, D], q/k/v pointers and strideM aligned (head_dim 128 here), the same
    number of heads in q, k, v (it has no enable_gqa path, hence repeat_kv); is_causal=True without a mask needs no
    sequence-length alignment (q_len == k_len in every call, so top-left causal alignment is exactly the causal mask).
  * with an additive mask the kernel checks `attn_bias` strides (batch/head/row) to be multiples of the kernel alignment
    (8 for fp16): [1, 1, N, N] has row stride N, hence N % 8 == 0 for B (otherwise torch pads the bias, one more copy).
  * a mask of the wrong dtype is rejected, so the mask is built in the dtype of q; masked entries use the dtype's most
    negative finite value (every row keeps its diagonal, no softmax over an all-masked row).
Whether the T4 really leaves the math backend is measured, not assumed: op_profile.py prints ATTN_STATS (calls per step of
_efficient_attention_forward/backward, bmm, _softmax) in the ab_attn run.

No triton import: the CPU test (test_packed_attn.py) imports this file.
"""

import torch
import torch.nn.functional as F

ENABLED = True            # False: always variant A (the original per-row call)
VARIANT = "A"             # "A" | "A2" | "B"
VARIANTS = ("A", "A2", "B")

_MASKS = {}               # dtype -> additive mask of the current forward (variant B)
_ROW_IDS = {"ids": None}


def variant():
    return VARIANT if ENABLED else "A"


def extra_tokens(n_real):
    """Filler tokens appended to the packed sequence: only variant B needs N % 8 == 0."""
    return (-n_real) % 8 if variant() == "B" else 0


def begin(lens, device):
    """Called by packed.py before the decoder layers of one packed forward."""
    _MASKS.clear()
    _ROW_IDS["ids"] = None
    if variant() == "B":
        _ROW_IDS["ids"] = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens)).to(device)


def end():
    _MASKS.clear()
    _ROW_IDS["ids"] = None


def block_causal_mask(dtype):
    mask = _MASKS.get(dtype)
    if mask is None:
        ids = _ROW_IDS["ids"]
        n = ids.shape[0]
        allowed = (ids[:, None] == ids[None, :]) & torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()
        mask = torch.zeros(n, n, dtype=dtype, device=ids.device).masked_fill(~allowed, torch.finfo(dtype).min)
        _MASKS[dtype] = mask = mask.view(1, 1, n, n)
    return mask


def attend(module, query, key, value, lens, dropout):
    """query [1, Hq, N, D], key/value [1, Hkv, N, D], lens = row lengths (sum = N). Returns [1, N, Hq, D]."""
    which = variant()
    if which == "A":
        return _attend_a(module, query, key, value, lens, dropout)
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    groups = module.num_key_value_groups
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)          # once, on the packed tensors
    if which == "B":
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=block_causal_mask(query.dtype),
                                             dropout_p=dropout, scale=module.scaling)
        return out.transpose(1, 2)
    pieces, start = [], 0
    for n in lens:
        end_ = start + n
        pieces.append(F.scaled_dot_product_attention(query[:, :, start:end_], key[:, :, start:end_],
                                                     value[:, :, start:end_], dropout_p=dropout,
                                                     is_causal=True, scale=module.scaling))
        start = end_
    return torch.cat(pieces, dim=2).transpose(1, 2)


def _attend_a(module, query, key, value, lens, dropout):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    pieces, start = [], 0
    for n in lens:
        end_ = start + n
        piece, _ = sdpa_attention_forward(module, query[:, :, start:end_], key[:, :, start:end_],
                                          value[:, :, start:end_], None, dropout=dropout,
                                          scaling=module.scaling, sliding_window=module.sliding_window)
        pieces.append(piece)
        start = end_
    return torch.cat(pieces, dim=1)
