"""Isolated R8 packed decoder, adapted from Ansar's pinned 3e48f6a packed.py.

Only an explicit head request may pack. Plain decoder/LM calls stay padded.
Admission completes before embedding/decoder execution; fallback never replays.
Dropout-off parity is tested; packing changes dropout RNG assignment when on.
No residual dtype casts, activation-storage or recompute integration.
"""

import atexit
import json
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

ENABLED = True

@dataclass
class _Request:
    model: object
    required: torch.Tensor
    consumed: bool = False


_REQUEST = ContextVar("packed_head_request", default=None)
_ACTIVE = ContextVar("packed_attention_rows", default=None)


@contextmanager
def head_request(model, required):
    """Request exactly one decoder call; restore nested state even on failure."""
    token = _REQUEST.set(_Request(model, required))
    try:
        yield
    finally:
        _REQUEST.reset(token)


# Passive counters (no GPU sync: the lengths are already on the CPU): one [real tokens, B * longest row] pair per
# packed step and the number of fallbacks by reason. One line "PACKED_STATS {json}" goes to stdout at exit
# (tools/packed_stats_report.py reads it from train.log); nothing is written to the file system.
_STATS = {"packed_steps": [], "fallbacks": Counter()}
_ORIGINAL = {}
_patched = False


def plan(attention_mask, required=None):
    """Validate binary, nonempty right padding and every shifted predictor first.

    Returns row lengths, row-major flat indices, original per-row positions.
    One small CPU readback for lengths and admission flags; no decoder replay.
    """
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return "not_2d"
    b, t = attention_mask.shape
    if b == 0 or t == 0:
        return "empty_shape"
    if required is not None and (required.shape != attention_mask.shape
                                or required.device != attention_mask.device):
        return "required_shape"
    mask = attention_mask.bool()
    lens = mask.sum(1)
    binary = ((attention_mask == 0) | (attention_mask == 1)).all()
    broken = (mask[:, 1:] & ~mask[:, :-1]).any()
    lost = (required & ~mask).any() if required is not None else mask.new_tensor(False)
    flags = torch.stack((~binary, broken, lost)).long()
    values = torch.cat((lens, flags)).tolist()
    lens, (nonbinary, broken, lost) = values[:-3], values[-3:]
    if nonbinary:
        return "nonbinary"
    if broken:
        return "left_pad_or_hole"
    if min(lens) == 0:
        return "empty_row"
    if lost:
        return "lost_predictor"
    if min(lens) == t:
        return "no_pad"
    positions = torch.cat([torch.arange(n) for n in lens])
    rows = torch.repeat_interleave(torch.arange(b) * t, torch.tensor(lens))
    device = attention_mask.device
    return lens, (positions + rows).to(device), positions.to(device).unsqueeze(0)


def _option(model, kwargs, name, default):
    value = kwargs.get(name)
    return getattr(model.config, name, default) if value is None else value


def _reason(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, kwargs):
    request = _REQUEST.get()
    if request is None or request.model is not model or request.consumed:
        return "no_head_request"
    # Consume even a rejected call. A later/reentrant call cannot inherit permission.
    request.consumed = True
    if not (model.training and torch.is_grad_enabled()):
        return "not_training"
    if input_ids is None or inputs_embeds is not None or input_ids.ndim != 2:
        return "no_input_ids"
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device):
        return "no_2d_mask"
    # Explicit False is required; None may mean config.use_cache=True.
    if position_ids is not None or past_key_values is not None or use_cache is not False:
        return "position_ids_or_cache"
    if getattr(model.config, "_attn_implementation", None) != "sdpa" or model.has_sliding_layers:
        return "not_sdpa_or_sliding"
    if getattr(model.rotary_emb, "rope_type", "default") != "default":
        return "nondefault_rope"
    if getattr(model, "gradient_checkpointing", False):
        return "gradient_checkpointing"
    if (_option(model, kwargs, "output_attentions", False)
            or _option(model, kwargs, "output_hidden_states", False)
            or not _option(model, kwargs, "return_dict", True)):
        return "output_options"
    # Unknown kwargs may affect attention/position semantics. Preserve upstream.
    if set(kwargs) - {"output_attentions", "output_hidden_states", "return_dict"}:
        return "unsupported_kwargs"
    if any(not getattr(layer.self_attn, "is_causal", True) for layer in model.layers):
        return "noncausal"
    return None


def head_rows(outputs, attended, required, width):
    """Validate the output contract before mapping packed hidden to original rows.

    A bad post-decoder contract is a hard error, never a second stochastic forward.
    """
    hidden = outputs.last_hidden_state
    b, t = attended.shape
    keep = getattr(outputs, "packed_keep", None)
    if keep is None:
        if hidden.ndim != 3 or hidden.shape != (b, t, width):
            raise RuntimeError("packed: padded hidden shape mismatch")
        return hidden[required], required
    indices = getattr(outputs, "packed_indices", None)
    positions = getattr(outputs, "packed_positions", None)
    if (not isinstance(keep, torch.Tensor) or keep.dtype != torch.bool
            or keep.shape != (b, t) or keep.device != hidden.device
            or not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
            or indices.ndim != 1 or indices.device != hidden.device
            or not isinstance(positions, torch.Tensor) or positions.dtype != torch.long
            or positions.device != hidden.device or positions.shape != (1, indices.numel())
            or hidden.shape != (1, indices.numel(), width) or indices.numel() == 0):
        raise RuntimeError("packed: invalid row-major layout metadata")
    # Sorted unique in-range indices, N matching population and all selected rows
    # attended establish the exact row-major bijection without another nonzero sync.
    valid = ((keep == attended).all() & ~(required & ~keep).any()
             & (keep.sum() == indices.numel()) & (indices >= 0).all()
             & (indices < b * t).all() & (indices[1:] > indices[:-1]).all()
             & keep.reshape(-1)[indices.clamp(0, b * t - 1)].all()
             & (positions == (indices % t).unsqueeze(0)).all())
    if not bool(valid):
        raise RuntimeError("packed: invalid row-major layout contract")
    return hidden[0], keep


def stats():
    """Counters of this process: packed steps as [real tokens N, padded tokens B * longest row] and fallbacks by reason."""
    steps = _STATS["packed_steps"]
    return {"packed_steps": len(steps), "fallbacks": dict(_STATS["fallbacks"]),
            "n_real": [n for n, _ in steps], "n_padded": [p for _, p in steps],
            "real_over_padded": (sum(n for n, _ in steps) / sum(p for _, p in steps)) if steps else None}


def reset_stats():
    _STATS["packed_steps"].clear()
    _STATS["fallbacks"].clear()


def _report_stats():
    if _STATS["packed_steps"] or _STATS["fallbacks"]:
        print("PACKED_STATS " + json.dumps(stats(), separators=(",", ":")), flush=True)


def _install():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2 as mq

    original_model_forward = mq.Qwen2Model.forward
    original_attention_forward = mq.Qwen2Attention.forward
    _ORIGINAL["model"], _ORIGINAL["attention"] = original_model_forward, original_attention_forward

    def model_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, **kwargs):
        found = _reason(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, kwargs)
        if found is None:
            found = plan(attention_mask, _REQUEST.get().required)
        if isinstance(found, str):
            _STATS["fallbacks"][found] += 1
            token = _ACTIVE.set(None)
            try:
                return original_model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                              position_ids=position_ids, past_key_values=past_key_values,
                                              inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
            finally:
                _ACTIVE.reset(token)
        lens, flat_index, positions = found
        _STATS["packed_steps"].append((sum(lens), len(lens) * attention_mask.shape[1]))
        hidden_states = self.embed_tokens(input_ids.reshape(-1)[flat_index]).unsqueeze(0)        # [1, N, H]
        position_embeddings = self.rotary_emb(hidden_states, positions)
        token = _ACTIVE.set((frozenset(id(layer.self_attn) for layer in self.layers), lens))
        try:
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = decoder_layer(hidden_states, attention_mask=None, position_embeddings=position_embeddings,
                                              position_ids=positions, past_key_values=None, use_cache=False, **kwargs)
        finally:
            _ACTIVE.reset(token)
        output = BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=None)
        output.packed_keep = attention_mask.bool()
        output.packed_indices = flat_index
        output.packed_positions = positions
        return output

    def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        active = _ACTIVE.get()
        lens = active[1] if active is not None and id(self) in active[0] else None
        if lens is None:
            return original_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                                              past_key_values=past_key_values, **kwargs)
        if hidden_states.shape[0] != 1 or sum(lens) != hidden_states.shape[1]:
            raise RuntimeError("packed: attention layout mismatch")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        dropout = 0.0 if not self.training else self.attention_dropout
        from .attention_adapter import row_attention
        pieces, start = [], 0
        for n in lens:
            end = start + n
            # The only attention arithmetic change: explicit per-row helper/fallback.
            # row_attention is imported above
            row_kwargs = dict(kwargs)
            row_kwargs.setdefault("past_key_values", past_key_values)
            row_kwargs.setdefault("sliding_window", self.sliding_window)
            piece, _ = row_attention(self, query_states[:, :, start:end], key_states[:, :, start:end],
                                     value_states[:, :, start:end], attention_mask,
                                     fallback=sdpa_attention_forward, dropout=dropout,
                                     scaling=self.scaling, **row_kwargs)
            pieces.append(piece)
            start = end
        attn_output = torch.cat(pieces, dim=1).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    mq.Qwen2Model.forward = model_forward
    mq.Qwen2Attention.forward = attention_forward


def patch():
    """Idempotent entry point used by solution.apply(); errors are not swallowed."""
    global _patched
    if not _patched:
        _install()
        atexit.register(_report_stats)
        _patched = True


def unpatch():
    """Undo patch() (used by the unit test)."""
    global _patched
    if _patched:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        mq.Qwen2Model.forward, mq.Qwen2Attention.forward = _ORIGINAL["model"], _ORIGINAL["attention"]
        _patched = False
