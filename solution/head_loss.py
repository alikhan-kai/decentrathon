"""Fused LM-head + cross-entropy + entropy + accuracy, in token chunks (DRAFT, not yet run on a T4).

Base Soup builds the full logits [B, T, 151936], upcasts them to fp32 and keeps several copies for
the loss, TRL's entropy, TRL's accuracy and the rewind log. Here:
  * rows used by attention or a shifted target are retained before the head;
  * the head runs `chunk` rows at a time; one Triton kernel reads each logits chunk and produces
    per-row loss, entropy and argmax-hit;
  * backward recomputes logits chunks and scales softmax - onehot in fp32 before storing fp16
    dlogits and taking dlogits @ W. The frozen lm_head never needs a weight gradient, and no
    full logits or unscaled fp16 gradient is saved across forward/backward.
Design idea (chunking the head + loss, hooking the causal-LM forward and the trainer's
compute_loss) is the same as Soup's own `stream_loss_chunk` (soup_cli/utils/stream_loss_chunk.py,
Apache-2.0); the Triton kernel and this code are new.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048
CHUNK = 512
IGNORE = -100


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
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            logits = h[s:e] @ w.t()                       # [c, V] fp16
            head_loss_kernel[(e - s,)](logits, target[s:e], logits.stride(0), vocab,
                                       loss_row[s:e], ent[s:e], hit[s:e], None,
                                       WITH_GRAD=False, BLOCK=BLOCK, num_warps=8)
            del logits
        denom = torch.as_tensor(denom, dtype=torch.float32, device=h.device)
        ctx.hidden_dtype = hidden.dtype
        ctx.chunk = chunk
        if need:
            ctx.save_for_backward(h, w, target, denom)
        ctx.mark_non_differentiable(loss_row, ent, hit)
        return loss_row.sum() / denom, loss_row, ent, hit

    @staticmethod
    def backward(ctx, grad_loss, *_):
        h, w, target, denom = ctx.saved_tensors
        grad_scale = grad_loss.float() / denom
        # Ignored targets have identically zero dlogits. Compact only backward;
        # forward still needs every retained row for entropy and other metrics.
        # CUDA nonzero synchronizes to determine the compacted length once per
        # backward. No explicit tensor-content readback is used.
        rows = torch.nonzero(target >= 0, as_tuple=False).flatten()
        dh = torch.zeros_like(h)
        n, vocab = rows.numel(), w.shape[0]
        for s in range(0, n, ctx.chunk):
            e = min(s + ctx.chunk, n)
            selected = rows[s:e]
            h_chunk = h.index_select(0, selected)
            target_chunk = target.index_select(0, selected)
            logits = h_chunk @ w.t()                    # same fp16 projection as forward
            head_loss_kernel[(e - s,)](logits, target_chunk, logits.stride(0), vocab,
                                       None, None, None, grad_scale,
                                       WITH_GRAD=True, BLOCK=BLOCK, num_warps=8)
            dh.index_copy_(0, selected, torch.mm(logits, w))  # already scaled, fp16 dL/dh
            del logits
        return dh.to(ctx.hidden_dtype), None, None, None, None


def fused_head_loss(hidden, weight, target, denom, chunk=CHUNK):
    """hidden [N, H], weight [V, H] (frozen), target [N] with -100 = ignore, denom = divisor.
    Returns loss (scalar, grad flows to hidden), per-row loss, per-row entropy, per-row hit."""
    return _FusedHeadLoss.apply(hidden, weight, target.contiguous(), denom, chunk)


# ----------------------------------------------------------------------------------------------
# Hooking into Soup: patch Qwen2ForCausalLM.forward (labelled, training) and SFTTrainer.compute_loss
# ----------------------------------------------------------------------------------------------
_STATE = {"armed": False, "step": None}
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
    import trl
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2 as mq

    original_forward = mq.Qwen2ForCausalLM.forward
    original_compute_loss = trl.SFTTrainer.compute_loss

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if not (_STATE["armed"] and labels is not None and attention_mask is not None
                and logits_to_keep == 0):
            return original_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                    logits_to_keep=logits_to_keep, **kwargs)
        _STATE["armed"] = False
        num_items = kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        b, t, _ = hidden.shape
        target_full = F.pad(labels, (0, 1), value=IGNORE)[:, 1:]
        # A left-pad row can predict the first supervised token after shifting.
        attended = attention_mask.bool()
        keep = attended | (target_full != IGNORE)
        h, tgt = hidden[keep], target_full[keep]
        labelled = (tgt != IGNORE).sum()
        denom = labelled if num_items is None else num_items
        loss, loss_row, ent, hit = fused_head_loss(h, self.lm_head.weight, tgt, denom)
        flat = keep.reshape(-1)
        tok_loss = torch.zeros(b * t, dtype=torch.float32, device=hidden.device)
        tok_ent = torch.zeros_like(tok_loss)
        tok_loss[flat], tok_ent[flat] = loss_row, ent
        tok_ent.masked_fill_(~attended.reshape(-1), 0.0)  # original attention population
        _STATE["step"] = _Step(tok_ent.view(b, t), tok_loss.view(b, t), hit.sum(), labelled)
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _eligible(self, inputs):
            return original_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        _STATE["armed"], _STATE["step"] = True, None
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        _STATE["armed"] = False
        step, _STATE["step"] = _STATE["step"], None
        if step is None:
            raise RuntimeError("head_loss: fast path was armed but the forward did not take it")
        setattr(self, STEP_ATTR, step)                     # read by Soup's rewind recorder
        # entropy, num_tokens, mean_token_accuracy: the same formulas as TRL's SFTTrainer
        gather, metrics = self.accelerator.gather_for_metrics, self._metrics["train"]
        am = inputs["attention_mask"]
        entropy = (step.token_entropy * am).sum() / am.sum()
        metrics["entropy"].append(gather(entropy).mean().item())
        self._total_train_tokens += gather(am.sum()).sum().item()
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
