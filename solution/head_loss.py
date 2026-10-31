"""Fused LM-head + cross-entropy + entropy + accuracy, in token chunks (DRAFT, not yet run on a T4).

Base Soup builds the full logits [B, T, 151936], upcasts them to fp32 and keeps several copies for
the loss, TRL's entropy, TRL's accuracy and the rewind log. Here:
  * rows used by attention or a shifted target are retained before the head;
  * the head runs `chunk` rows at a time; one Triton kernel reads each logits chunk and produces
    per-row loss, entropy and argmax-hit;
  * backward reuses bounded FP16 logits (or recomputes chunks) and scales softmax - onehot in fp32 before storing fp16
    dlogits and taking dlogits @ W. The frozen lm_head never needs a weight gradient, and no
    unscaled fp16 gradient is saved across forward/backward. Bounded FP16 logits
    caching avoids projection recomputation; larger inputs retain the remat path.
Design idea (chunking the head + loss, hooking the causal-LM forward and the trainer's
compute_loss) is the same as Soup's own `stream_loss_chunk` (soup_cli/utils/stream_loss_chunk.py,
Apache-2.0); the Triton kernel and this code are new.
"""

from contextvars import ContextVar

import torch
import triton
import triton.language as tl

BLOCK = 2048
CHUNK = 256
IGNORE = -100
CACHE_LOGITS_BYTES = 128 * 1024 * 1024


@triton.jit
def head_loss_kernel(logits_ptr, target_ptr, stride, V, loss_ptr, ent_ptr, hit_ptr, grad_scale_ptr,
                     WITH_GRAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride
    tgt = tl.load(target_ptr + row)
    valid = tgt >= 0
    lane = tl.arange(0, BLOCK)

    # per-lane online softmax state (reduced across lanes once, after the loop)
    m_l = tl.full([BLOCK], -1e30, tl.float32)      # running max
    s_l = tl.zeros([BLOCK], tl.float32)            # sum exp(x - m)
    t_l = tl.zeros([BLOCK], tl.float32)            # sum exp(x - m) * x
    xt_l = tl.zeros([BLOCK], tl.float32)           # logit of the target
    best_l = tl.full([BLOCK], -1e30, tl.float32)   # running argmax value
    bi_l = tl.zeros([BLOCK], tl.int32)             # running argmax index

    for start in range(0, V, BLOCK):
        offs = start + lane
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        xs = tl.where(mask, x, 0.0)
        m_new = tl.maximum(m_l, x)
        e = tl.exp(x - m_new)
        scale = tl.exp(m_l - m_new)
        s_l = s_l * scale + e
        t_l = t_l * scale + e * xs
        m_l = m_new
        xt_l += tl.where(offs == tgt, xs, 0.0)
        better = x > best_l
        best_l = tl.where(better, x, best_l)
        bi_l = tl.where(better, offs, bi_l)

    big = tl.max(m_l, 0)
    w = tl.exp(m_l - big)
    ssum = tl.sum(s_l * w, 0)
    tsum = tl.sum(t_l * w, 0)
    lse = big + tl.log(ssum)
    entropy = lse - tsum / ssum
    xt = tl.sum(xt_l, 0)
    gbest = tl.max(best_l, 0)
    arg = tl.min(tl.where(best_l == gbest, bi_l, 2147483647), 0)

    if not WITH_GRAD:
        tl.store(loss_ptr + row, tl.where(valid, lse - xt, 0.0))
        tl.store(ent_ptr + row, entropy)
        tl.store(hit_ptr + row, tl.where(valid & (arg == tgt), 1, 0))

    if WITH_GRAD:
        grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
        for start in range(0, V, BLOCK):
            offs = start + lane
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            g = tl.exp(x - lse) - tl.where(offs == tgt, 1.0, 0.0)
            # Apply the incoming loss scale before either fp16 rounding boundary.
            g = g * grad_scale
            # Mask last: an ignored row must stay zero even when denom == 0.
            g = tl.where(valid, g, 0.0)
            tl.store(base + offs, g.to(logits_ptr.dtype.element_ty), mask=mask)



@triton.jit
def _cache_labelled_logits(logits, saved, row_map, V: tl.constexpr,
                          COUNT, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // V
    column = offsets % V
    dest_row = tl.load(row_map + row, mask=offsets < COUNT, other=-1)
    value = tl.load(logits + offsets, mask=offsets < COUNT, other=0)
    tl.store(saved + dest_row.to(tl.int64) * V + column, value,
             mask=(offsets < COUNT) & (dest_row >= 0))


class _FusedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, denom, chunk):
        need = ctx.needs_input_grad[0]
        h = hidden.detach().to(torch.float16)
        w = weight.detach().to(torch.float16)
        n, vocab = h.shape[0], w.shape[0]
        loss_row = torch.empty(n, dtype=torch.float32, device=h.device)
        ent = torch.empty_like(loss_row)
        hit = torch.empty(n, dtype=torch.int32, device=h.device)
        # Forward still computes entropy on every retained row. Only labelled
        # rows can contribute to backward, so only those logits need retention.
        rows = (torch.nonzero(target >= 0, as_tuple=False).flatten() if need
                else torch.empty(0, dtype=torch.long, device=h.device))
        labelled = rows.numel()
        cached_rows = min(labelled, max(0, CACHE_LOGITS_BYTES // (vocab * 2))) if need else 0
        cache = cached_rows > 0
        compact = cache and cached_rows < n
        row_map = None
        if compact:
            valid = target >= 0
            rank = valid.to(torch.int64).cumsum(0) - 1
            row_map = torch.where(valid & (rank < cached_rows), rank, -1)
        saved_logits = torch.empty((cached_rows, vocab) if cache else (0,), dtype=torch.float16, device=h.device)
        if n > 0:
            wt = w.t()
            logits_buf = None if cache and not compact else torch.empty((min(chunk, n), vocab), dtype=torch.float16, device=h.device)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                c = e - s
                logits = saved_logits[s:e] if cache and not compact else logits_buf[:c]
                torch.mm(h[s:e], wt, out=logits)          # [c, V] fp16
                head_loss_kernel[(c,)](logits, target[s:e], vocab, vocab,
                                       loss_row[s:e], ent[s:e], hit[s:e], None,
                                       WITH_GRAD=False, BLOCK=BLOCK, num_warps=8)
                if compact and labelled:
                    _cache_labelled_logits[(triton.cdiv(c * vocab, 1024),)](
                        logits, saved_logits, row_map[s:e], vocab, c * vocab, 1024)
            del logits_buf
        denom = torch.as_tensor(denom, dtype=torch.float32, device=h.device)
        ctx.hidden_dtype = hidden.dtype
        ctx.chunk = chunk
        ctx.cached_logits = cache
        ctx.cached_rows = cached_rows
        if need:
            ctx.save_for_backward(h, w, target, denom, saved_logits, rows)
        ctx.mark_non_differentiable(loss_row, ent, hit)
        return loss_row.sum() / denom, loss_row, ent, hit

    @staticmethod
    def backward(ctx, grad_loss, *_):
        h, w, target, denom, saved_logits, rows = ctx.saved_tensors
        grad_scale = grad_loss.float() / denom
        # The forward established the same labelled-row order used by the cache.
        dh = torch.zeros_like(h)
        n, vocab = rows.numel(), w.shape[0]
        if n > 0:
            wt = w.t()
            max_c = min(ctx.chunk, n)
            logits_buf = torch.empty((max_c, vocab), dtype=torch.float16, device=h.device)
            dh_buf = torch.empty((max_c, h.shape[1]), dtype=torch.float16, device=h.device)
            for s in range(0, n, ctx.chunk):
                e = min(s + ctx.chunk, n)
                c = e - s
                selected = rows[s:e]
                target_chunk = target.index_select(0, selected)
                logits = logits_buf[:c]
                cached_c = max(0, min(e, ctx.cached_rows) - s)
                if cached_c:
                    logits[:cached_c].copy_(saved_logits[s:s + cached_c])
                if cached_c < c:
                    h_chunk = h.index_select(0, selected[cached_c:])
                    torch.mm(h_chunk, wt, out=logits[cached_c:])
                head_loss_kernel[(c,)](logits, target_chunk, vocab, vocab,
                                       None, None, None, grad_scale,
                                       WITH_GRAD=True, BLOCK=BLOCK, num_warps=8)
                dh_chunk = dh_buf[:c]
                torch.mm(logits, w, out=dh_chunk)         # already scaled, fp16 dL/dh
                dh.index_copy_(0, selected, dh_chunk)
            del logits_buf, dh_buf
        return dh.to(ctx.hidden_dtype), None, None, None, None


def fused_head_loss(hidden, weight, target, denom, chunk=CHUNK):
    """hidden [N, H], weight [V, H] (frozen), target [N] with -100 = ignore, denom = divisor.
    Returns loss (scalar, grad flows to hidden), per-row loss, per-row entropy, per-row hit."""
    return _FusedHeadLoss.apply(hidden, weight, target.contiguous(), denom, chunk)


# ----------------------------------------------------------------------------------------------
# Hooking into Soup: patch Qwen2ForCausalLM.forward (labelled, training) and SFTTrainer.compute_loss
# ----------------------------------------------------------------------------------------------
_REQUEST = ContextVar("fused_head_request", default=None)


class _Request:
    def __init__(self, model):
        self.model, self.consumed, self.step = model, False, None


def _base(model):
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _bridge_eligible(model, inputs):
    base = _base(model)
    if inputs.get("_prediction_loss_only") is False:
        return False
    if inputs.get("past_key_values") is not None or inputs.get("use_cache") is True:
        return False
    allowed = {"input_ids", "attention_mask", "labels", "inputs_embeds", "position_ids",
               "past_key_values", "use_cache", "logits_to_keep", "output_attentions",
               "output_hidden_states", "return_dict", "_prediction_loss_only"}
    if set(inputs) - allowed:
        return False
    value = inputs.get("return_dict")
    if not (getattr(getattr(base, "config", None), "return_dict", True) if value is None else value):
        return False
    return True
STEP_ATTR = "_soup_chunked_step"   # what Soup's rewind recorder reads (rewind_hf.py)


class _Step:
    def __init__(self, token_entropy, token_loss, correct, labelled):
        self.token_entropy, self.token_loss = token_entropy, token_loss
        self.correct, self.labelled = correct, labelled


def _eligible(trainer, inputs):
    if not (trainer.model.training and "labels" in inputs and inputs.get("attention_mask") is not None):
        return False
    # Trainer removes labels before forward for both of these loss paths.
    if getattr(trainer, "label_smoother", None) is not None or getattr(trainer, "compute_loss_func", None) is not None:
        return False
    if (getattr(trainer.args, "use_liger_kernel", False)
            or getattr(trainer.args, "loss_type", "nll") != "nll"
            or getattr(trainer, "num_virtual_tokens", 0)
            or getattr(trainer, "aux_loss_enabled", False)
            or "shift_labels" in inputs or "ignore_index" in inputs):
        return False
    keep = inputs.get("logits_to_keep", 0)
    if type(keep) is not int or keep != 0:
        return False
    labels, mask = inputs["labels"], inputs["attention_mask"]
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2
            or not isinstance(mask, torch.Tensor) or mask.shape != labels.shape):
        return False
    base = trainer.model
    get_base = getattr(base, "get_base_model", None)
    base = get_base() if callable(get_base) else base
    head = getattr(base, "lm_head", None)
    # Fusion reads only weight, bypassing the head's __call__, bias and adapters.
    if (type(head) is not torch.nn.Linear or head.bias is not None
            or "forward" in head.__dict__ or head._forward_pre_hooks or head._forward_hooks
            or head._backward_pre_hooks or head._backward_hooks):
        return False
    from torch.nn.modules import module as module_hooks
    if (module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
            or module_hooks._global_backward_pre_hooks or module_hooks._global_backward_hooks):
        return False
    weight = head.weight
    if not weight.is_cuda or weight.requires_grad or weight.dtype not in (torch.float16, torch.float32):
        return False
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    if type(base) is not Qwen2ForCausalLM or base.loss_function is not ForCausalLMLoss:
        return False
    # Accelerate's prepared forward enters autocast *after* this predicate.
    # Storage dtype or the currently active context alone cannot identify bf16.
    accelerator = trainer.accelerator
    precision = getattr(accelerator, "mixed_precision", "no")
    if getattr(trainer.args, "bf16", False) or precision not in ("no", "fp16"):
        return False
    autocast = torch.is_autocast_enabled(weight.device.type)
    if autocast and torch.get_autocast_dtype(weight.device.type) != torch.float16:
        return False
    if getattr(accelerator, "native_amp", False):
        # A disabled prepared wrapper also disables an outer autocast context.
        return (precision == "fp16"
                and getattr(getattr(accelerator, "autocast_handler", None), "enabled", True))
    return weight.dtype == torch.float16 or autocast


def _install():
    import torch.nn.functional as F
    from . import packed
    import trl
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2 as mq

    original_forward = mq.Qwen2ForCausalLM.forward
    original_compute_loss = trl.SFTTrainer.compute_loss

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        request = _REQUEST.get()
        if not (request is not None and request.model is self and not request.consumed
                and labels is not None and attention_mask is not None and logits_to_keep == 0):
            return original_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                    logits_to_keep=logits_to_keep, **kwargs)
        request.consumed = True
        from .frozen_weight import compact_tied_weight
        compact_tied_weight(self)
        num_items = kwargs.pop("num_items_in_batch", None)
        # Shift on the ORIGINAL [B,T] layout, before any packed gather.
        b, t = labels.shape
        target_full = F.pad(labels, (0, 1), value=IGNORE)[:, 1:]
        attended = attention_mask.bool()
        required = attended | (target_full != IGNORE)
        with packed.head_request(self.model, required):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        h, keep = packed.head_rows(outputs, attended, required, self.lm_head.weight.shape[1])
        flat_idx = getattr(outputs, "packed_indices", None) if getattr(outputs, "_packed_trusted", False) else None
        if flat_idx is not None:
            tgt = target_full.reshape(-1).index_select(0, flat_idx)
        else:
            tgt = target_full[keep]
        labelled = (tgt != IGNORE).sum()
        denom = labelled if num_items is None else num_items
        loss, loss_row, ent, hit = fused_head_loss(h, self.lm_head.weight, tgt, denom)
        tok_loss = torch.zeros(b * t, dtype=torch.float32, device=hidden.device)
        tok_ent = torch.zeros_like(tok_loss)
        if flat_idx is not None:
            tok_loss.index_copy_(0, flat_idx, loss_row)
            tok_ent.index_copy_(0, flat_idx, ent)
        else:
            flat = keep.reshape(-1)
            tok_loss[flat], tok_ent[flat] = loss_row, ent
            tok_ent.masked_fill_(~attended.reshape(-1), 0.0)  # original attention population
        request.step = _Step(tok_ent.view(b, t), tok_loss.view(b, t), hit.sum(), labelled)
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Clear stale Rewind data before either route; eligibility remains the reviewed guard.
        setattr(self, STEP_ATTR, None)
        if not (_base(model) is _base(self.model)
                and _bridge_eligible(model, inputs) and _eligible(self, inputs)):
            token = _REQUEST.set(None)
            try:
                return original_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                             num_items_in_batch=num_items_in_batch)
            finally:
                _REQUEST.reset(token)
        inputs = dict(inputs)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        request = _Request(_base(model))
        token = _REQUEST.set(request)
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            step = request.step
        finally:
            _REQUEST.reset(token)
        if step is None:
            raise RuntimeError("head_loss: fast path was armed but the forward did not take it")
        setattr(self, STEP_ATTR, step)                     # read by Soup's rewind recorder
        # entropy, num_tokens, mean_token_accuracy: the same formulas as TRL's SFTTrainer
        metrics = self._metrics["train"]
        am = inputs["attention_mask"]
        am_sum = am.sum()
        entropy = (step.token_entropy * am).sum() / am_sum
        if getattr(self.accelerator, "num_processes", None) == 1:
            acc_t = step.correct / step.labelled.clamp_min(1)
            ent_v, am_v, lab_v, acc_v = torch.stack((
                entropy.float(), am_sum.float(), step.labelled.float(), acc_t.float()
            )).tolist()
            metrics["entropy"].append(ent_v)
            self._total_train_tokens += int(am_v)
            metrics["num_tokens"] = [self._total_train_tokens]
            metrics["mean_token_accuracy"].append(acc_v if lab_v > 0 else 0.0)
        else:
            gather = self.accelerator.gather_for_metrics
            metrics["entropy"].append(gather(entropy).mean().item())
            self._total_train_tokens += gather(am_sum).sum().item()
            metrics["num_tokens"] = [self._total_train_tokens]
            total = gather(step.labelled).sum()
            acc = (gather(step.correct).sum() / total).item() if total > 0 else 0.0
            metrics["mean_token_accuracy"].append(acc)
        return (loss, outputs) if return_outputs else loss

    mq.Qwen2ForCausalLM.forward = forward
    trl.SFTTrainer.compute_loss = compute_loss


ENABLED = True
_patched = False


def patch():
    """Idempotent entry point used by solution.apply()."""
    global _patched
    if not _patched:
        _install()
        _patched = True
