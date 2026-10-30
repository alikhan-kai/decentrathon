"""Narrow Soup/TRL bridge for the frozen Qwen2 output head.

Normal model calls still return logits. The alternative route is scoped to a
supported training compute_loss call, preserving PEFT and Accelerate wrappers.
Soup's existing per-token Rewind interface is used without enabling its
stream_loss_chunk switch or invoking its head implementation.
"""

import contextvars
import functools
from dataclasses import dataclass

import torch

from .head import frozen_head_loss


_ACTIVE = contextvars.ContextVar("ember_head_call", default=None)
_STEP_ATTRIBUTE = "_soup_chunked_step"


@dataclass
class _HeadCall:
    model: object
    result: object = None


def _base_model(model):
    """Unwrap only known transparent container conventions, with a cycle guard."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    seen = set()
    while isinstance(model, torch.nn.Module) and id(model) not in seen:
        seen.add(id(model))
        if type(model) is Qwen2ForCausalLM:
            return model
        if callable(getattr(model, "get_base_model", None)):
            model = model.get_base_model()
        elif hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return None
    return None


def attach_model(model):
    """Mark a supported frozen head after Soup has finished preparing PEFT."""
    if not isinstance(model, torch.nn.Module):
        return False
    base = _base_model(model)
    if base is None or type(base.lm_head) is not torch.nn.Linear:
        return False
    if base.lm_head.weight.requires_grad or base.lm_head.bias is not None:
        return False
    base._ember_head_ready = True
    return True


def _eligible(trainer, model, inputs):
    if not isinstance(model, torch.nn.Module) or not model.training:
        return None
    base = _base_model(model)
    if base is None or not getattr(base, "_ember_head_ready", False):
        return None
    if base.lm_head.weight.requires_grad or base.lm_head.bias is not None:
        return None
    from transformers.loss.loss_utils import ForCausalLMLoss
    if base.loss_function is not ForCausalLMLoss:
        return None
    if getattr(trainer, "label_smoother", None) is not None or getattr(trainer, "compute_loss_func", None) is not None:
        return None
    if any(getattr(trainer, key, False) for key in ("aux_loss_enabled", "padding_free", "num_virtual_tokens")):
        return None
    args = trainer.args
    if getattr(args, "use_liger_kernel", False) or getattr(args, "loss_type", "nll") != "nll":
        return None
    # Accelerate can enter autocast *inside* model.forward, after this guard.
    # Check the configured mode too, so a bf16 wrapper keeps the dense route.
    if getattr(args, "bf16", False) or getattr(trainer.accelerator, "mixed_precision", "no") == "bf16":
        return None
    if getattr(trainer.accelerator, "num_processes", 1) != 1 or getattr(args, "n_gpu", 0) > 1:
        return None
    if any(key in inputs for key in ("shift_labels", "past_key_values", "inputs_embeds")):
        return None
    if inputs.get("return_dict") is False:
        return None
    keep = inputs.get("logits_to_keep", 0)
    if not isinstance(keep, int) or keep != 0:
        return None
    labels = inputs.get("labels")
    mask = inputs.get("attention_mask")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        return None
    if not isinstance(mask, torch.Tensor) or mask.shape != labels.shape:
        return None
    weight = base.lm_head.weight
    if weight.device.type not in ("cuda", "cpu") or weight.dtype not in (torch.float16, torch.float32):
        return None
    if torch.is_autocast_enabled(weight.device.type):
        if torch.get_autocast_dtype(weight.device.type) not in (torch.float16, torch.float32):
            return None
    return base


def _record_metrics(trainer, result, inputs):
    """The same token populations and distributed reductions used by TRL."""
    mask = inputs["attention_mask"]
    labels = inputs["labels"][:, 1:]
    valid = labels != -100
    gather = trainer.accelerator.gather_for_metrics
    with torch.no_grad():
        entropy = (result.token_entropy * mask).sum() / mask.sum()
        entropy = gather(entropy).mean().item()
        correct = gather(((result.predictions[:, :-1] == labels) & valid).sum()).sum()
        count = gather(valid.sum()).sum()
        accuracy = (correct / count).item() if count > 0 else 0.0
        trainer._total_train_tokens += gather(mask.sum()).sum().item()
    metrics = trainer._metrics["train"]
    metrics["entropy"].append(entropy)
    metrics["mean_token_accuracy"].append(accuracy)
    metrics["num_tokens"] = [trainer._total_train_tokens]
    # Rewind consumes only token_loss; no built-in chunking implementation runs.
    setattr(trainer, _STEP_ATTRIBUTE, result)


def patch_training():
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from trl import SFTTrainer
    from soup_cli.trainer.sft import SFTTrainerWrapper

    original_forward = Qwen2ForCausalLM.forward
    if not getattr(original_forward, "_ember_patched", False):
        @functools.wraps(original_forward)
        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    past_key_values=None, inputs_embeds=None, labels=None,
                    use_cache=None, logits_to_keep=0, **kwargs):
            call = _ACTIVE.get()
            if call is None or call.model is not self:
                return original_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                        position_ids=position_ids, past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                        logits_to_keep=logits_to_keep, **kwargs)
            decoder_kwargs = dict(kwargs)
            denominator = decoder_kwargs.pop("num_items_in_batch", None)
            decoder_kwargs.pop("return_dict", None)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds, use_cache=use_cache,
                                return_dict=True, **decoder_kwargs)
            result = frozen_head_loss(output.last_hidden_state, self.lm_head.weight,
                                      labels, denominator)
            call.result = result
            return CausalLMOutputWithPast(loss=result.loss, logits=None,
                                         past_key_values=output.past_key_values,
                                         hidden_states=output.hidden_states,
                                         attentions=output.attentions)
        forward._ember_patched = True
        Qwen2ForCausalLM.forward = forward

    original_compute = SFTTrainer.compute_loss
    if not getattr(original_compute, "_ember_patched", False):
        @functools.wraps(original_compute)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            setattr(self, _STEP_ATTRIBUTE, None)
            base = _eligible(self, model, inputs)
            if base is None:
                return original_compute(self, model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False
            call = _HeadCall(base)
            token = _ACTIVE.set(call)
            try:
                loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                    num_items_in_batch=num_items_in_batch)
                if call.result is None:
                    raise RuntimeError("the supported Qwen head did not produce diagnostics")
                _record_metrics(self, call.result, inputs)
                return (loss, outputs) if return_outputs else loss
            finally:
                _ACTIVE.reset(token)
        compute_loss._ember_patched = True
        SFTTrainer.compute_loss = compute_loss

    original_setup = SFTTrainerWrapper._setup_transformers
    if not getattr(original_setup, "_ember_patched", False):
        @functools.wraps(original_setup)
        def setup(self, *args, **kwargs):
            result = original_setup(self, *args, **kwargs)
            attach_model(getattr(self, "model", None))
            return result
        setup._ember_patched = True
        SFTTrainerWrapper._setup_transformers = setup
