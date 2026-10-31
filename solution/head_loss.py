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

from contextvars import ContextVar

import torch
import triton
import triton.language as tl

BLOCK = 2048
CHUNK = 128
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

    Computes sum(cross_entropy(logits, target)) / denom, row entropy, and argmax hits in chunks.
    No [N, V] tensor or unscaled gradient is ever materialized.
    """
    return _FusedHeadLoss.apply(hidden, weight, target, denom, chunk)


# Dynamic patch: hooked callers read _CURRENT_MODEL to bridge causal-LM forward to compute_loss.
# ContextVar keeps the pair thread-safe if evaluation or a future task runs in a worker thread.
_CURRENT_MODEL = ContextVar("current_model", default=None)
_PATCHED = False
_ORIGINAL = {}


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
    return True


def _compute_loss_hook(trainer, orig_compute_loss, model, inputs, return_outputs=False, num_items_in_batch=None):
    if not _eligible(trainer, inputs):
        return orig_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    # Shifted target matching CausalLM forward: row [b, t] predicts labels[b, t+1].
    # row [b, -1] has no target and is ignored; row [b, t] is ignored if labels[b, t+1] == -100.
    labels, mask = inputs["labels"], inputs["attention_mask"]
    target = torch.full_like(labels, IGNORE)
    target[:, :-1] = labels[:, 1:]
    target = torch.where(mask == 1, target, IGNORE)
    # Rows with no target are excluded from denom and from backward logits projection.
    # Same denominator as ForCausalLMLoss: num_items_in_batch if given, else non-ignored targets.
    if num_items_in_batch is not None:
        denom = num_items_in_batch
    else:
        denom = (target != IGNORE).sum()
    denom = max(1, int(denom))

    # Request the model hook to capture and compact the hidden states on the very next forward.
    # If the forward falls back, the request is marked consumed and compute_loss follows suit.
    req = _Request(model)
    tok = _CURRENT_MODEL.set(req)
    try:
        # Pass shifted target and attended mask down so packed / selective forwards can retain only
        # rows that matter for loss or for attention to a loss row (e.g. prompt prefix).
        attended = mask == 1
        required = (target != IGNORE) | attended
        from . import packed
        with packed.head_request(model, required):
            outputs = model(**inputs)
        step = req.step
        if not req.consumed or step is None:
            # Model hook did not run (e.g. an unexpected inner wrapper swallowed it). Fall back.
            return orig_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
    finally:
        _CURRENT_MODEL.reset(tok)

    # Attach the Step recorder to the model where Soup's RewindSFTTrainer reads it.
    setattr(trainer.model, STEP_ATTR, step)

    loss = step.loss
    return (loss, outputs) if return_outputs else loss


def _model_forward_hook(model, orig_forward, *args, **kwargs):
    req = _CURRENT_MODEL.get()
    if req is None or req.model is not model or req.consumed:
        return orig_forward(*args, **kwargs)
    req.consumed = True

    # Call the original forward with logits_to_keep=0 (Transformers 4.45+). Qwen2 skips the lm_head
    # entirely and sets logits=None.
    kwargs["logits_to_keep"] = 0
    outputs = orig_forward(*args, **kwargs)
    if outputs.logits is not None:
        # Pre-4.45 or custom forward that does not honor logits_to_keep. Fall back.
        return outputs

    # Compact hidden states to non-ignored rows.
    # Note: caller (_compute_loss_hook) already verified the architecture, head and autocast precision.
    base = _base(model)
    head = base.lm_head
    labels, mask = kwargs["labels"], kwargs["attention_mask"]
    target = torch.full_like(labels, IGNORE)
    target[:, :-1] = labels[:, 1:]
    target = torch.where(mask == 1, target, IGNORE)
    attended = mask == 1
    required = (target != IGNORE) | attended
    width = base.config.hidden_size
    from . import packed
    hidden, keep = packed.head_rows(outputs, attended, required, width)
    target = target[keep]

    denom = (target != IGNORE).sum()
    denom = max(1, int(denom))

    loss, loss_row, ent, hit = fused_head_loss(hidden, head.weight, target, denom)

    # Unpack per-row metrics back to the [B, T] grid for Soup's rewind log and accuracy metric.
    grid_ent = torch.zeros_like(labels, dtype=torch.float32)
    grid_loss = torch.zeros_like(labels, dtype=torch.float32)
    grid_hit = torch.zeros_like(labels, dtype=torch.int32)
    grid_ent[keep] = ent
    grid_loss[keep] = loss_row
    grid_hit[keep] = hit

    # Only attended positions with a real target contribute to correct / labelled counts.
    valid = (target != IGNORE)
    correct = hit[valid].sum()
    labelled = valid.sum()

    step = _Step(grid_ent.detach(), grid_loss.detach(), correct.detach(), labelled.detach())
    step.loss = loss
    req.step = step

    outputs.loss = loss
    return outputs


def patch():
    """Idempotent: install the fused chunked head into Soup's training path."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from trl import SFTTrainer
    except ImportError:
        return

    _ORIGINAL["trainer_compute_loss"] = SFTTrainer.compute_loss
    _ORIGINAL["model_forward"] = Qwen2ForCausalLM.forward

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return _compute_loss_hook(self, _ORIGINAL["trainer_compute_loss"], model, inputs,
                                  return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def forward(self, *args, **kwargs):
        return _model_forward_hook(self, _ORIGINAL["model_forward"], *args, **kwargs)

    SFTTrainer.compute_loss = compute_loss
    Qwen2ForCausalLM.forward = forward
    _PATCHED = True


def unpatch():
    """Undo patch() (for tests)."""
    global _PATCHED
    if not _PATCHED:
        return
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from trl import SFTTrainer

    SFTTrainer.compute_loss = _ORIGINAL["trainer_compute_loss"]
    Qwen2ForCausalLM.forward = _ORIGINAL["model_forward"]
    _PATCHED = False
