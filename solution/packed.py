"""Drop pad tokens from the whole decoder, not only before the head (PROTOTYPE, not yet run on a T4).

TRL pads every batch on the right up to its longest row (DataCollatorForLanguageModeling, attention_mask 0 on pad),
so a [4, T] batch spends 28 layers of matmuls, norms, LoRA and SwiGLU on pad rows. Here, in training, the real
tokens of the batch are packed into one sequence [1, N, H] right after the embedding:
  * input_ids are packed (row-major order of the real tokens) before `embed_tokens`; position_ids restart at 0 in every
    row, so rotary sees the same positions as the padded run (right padding: real tokens sit at 0..len-1);
  * norms, MLP, LoRA, q/k/v/o projections run unchanged on [1, N, H] (they only look at the last dimension);
  * attention is done per row: q, k, v are cut to the rows' lengths and each piece goes through transformers' own
    `sdpa_attention_forward` with is_causal (no mask), then the pieces are concatenated. Rows never see each other;
  * the final norm runs on [1, N, H]; the output carries `packed_keep` (the bool attention_mask) so head_loss can use
    the packed hidden directly and scatter token_loss / token_entropy back to [B, T].
Packing is NOT mixing rows of different batches: it only removes pad inside one batch.

Falls back to the original forward (no change at all) when: not training, input_ids absent or inputs_embeds given,
position_ids / past_key_values / use_cache given, attention is not sdpa, output_attentions / output_hidden_states asked,
sliding-window layers are on, the mask is not 2D or not right-padded (a 1 after a 0), a row is empty, or the batch has no pad.

Patched names (transformers 5.16.1, modeling_qwen2): Qwen2Model.forward and Qwen2Attention.forward.
Order in solution.MODULES: after head_loss (head_loss patches Qwen2ForCausalLM.forward and SFTTrainer.compute_loss and
reads `packed_keep` from the output of self.model(...)); no other module patches these two methods (rmsnorm patches
Qwen2RMSNorm.forward, swiglu Qwen2MLP.forward, nf4 a bitsandbytes function, lora a peft method). If a module ever also
wraps Qwen2Model.forward it must be listed after this one so that it sees our wrapper.

LoRA dropout draws a mask of the packed size, so random values differ from the padded run: that is expected. With
dropout off (or on the first step, B = 0) the results agree to float rounding (solution/test_packed.py).
No triton import here: the test runs on CPU.
"""

import torch

ENABLED = True

_STATE = {"lens": None}      # row lengths of the batch currently inside the packed Qwen2Model.forward
_ORIGINAL = {}
_patched = False


def plan(attention_mask):
    """Row lengths and flat indices of the real tokens of a right-padded [B, T] mask, or None if the mask is not
    right-padded, has an empty row, or has no pad at all. One device sync."""
    if attention_mask.dim() != 2:
        return None
    b, t = attention_mask.shape
    mask = attention_mask.to(torch.long)
    lens = mask.sum(1)
    broken = (mask[:, 1:] > mask[:, :-1]).any().to(torch.long).view(1)
    *lens, broken = torch.cat([lens, broken]).tolist()
    if broken or min(lens) == 0 or min(lens) == t and max(lens) == t:
        return None
    positions = torch.cat([torch.arange(n) for n in lens])
    rows = torch.repeat_interleave(torch.arange(b) * t, torch.tensor(lens))
    device = attention_mask.device
    return lens, (positions + rows).to(device), positions.to(device).unsqueeze(0)


def _eligible(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, kwargs):
    return (model.training and torch.is_grad_enabled() and input_ids is not None and inputs_embeds is None
            and attention_mask is not None and attention_mask.shape == input_ids.shape
            and position_ids is None and past_key_values is None and not use_cache
            and getattr(model.config, "_attn_implementation", None) == "sdpa"
            and not model.has_sliding_layers
            and not kwargs.get("output_attentions") and not kwargs.get("output_hidden_states"))


def _install():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2 as mq

    original_model_forward = mq.Qwen2Model.forward
    original_attention_forward = mq.Qwen2Attention.forward
    _ORIGINAL["model"], _ORIGINAL["attention"] = original_model_forward, original_attention_forward

    def model_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, **kwargs):
        found = None
        if _eligible(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, kwargs):
            found = plan(attention_mask)
        if found is None:
            return original_model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                          position_ids=position_ids, past_key_values=past_key_values,
                                          inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        lens, flat_index, positions = found
        hidden_states = self.embed_tokens(input_ids.reshape(-1)[flat_index]).unsqueeze(0)        # [1, N, H]
        position_embeddings = self.rotary_emb(hidden_states, positions)
        _STATE["lens"] = lens
        try:
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = decoder_layer(hidden_states, attention_mask=None, position_embeddings=position_embeddings,
                                              position_ids=positions, past_key_values=None, use_cache=False, **kwargs)
        finally:
            _STATE["lens"] = None
        output = BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=None)
        output.packed_keep = attention_mask.bool()       # read by head_loss: hidden is [1, N, H] in this row-major order
        return output

    def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        lens = _STATE["lens"]
        if lens is None or hidden_states.shape[0] != 1 or sum(lens) != hidden_states.shape[1]:
            return original_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                                              past_key_values=past_key_values, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        dropout = 0.0 if not self.training else self.attention_dropout
        pieces, start = [], 0
        for n in lens:
            end = start + n
            piece, _ = sdpa_attention_forward(self, query_states[:, :, start:end], key_states[:, :, start:end],
                                              value_states[:, :, start:end], None, dropout=dropout,
                                              scaling=self.scaling, sliding_window=self.sliding_window)
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
        _patched = True


def unpatch():
    """Undo patch() (used by the unit test)."""
    global _patched
    if _patched:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        mq.Qwen2Model.forward, mq.Qwen2Attention.forward = _ORIGINAL["model"], _ORIGINAL["attention"]
        _patched = False
