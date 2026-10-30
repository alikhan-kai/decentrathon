"""CPU dispatch regressions using the real Qwen2, SFTTrainer, PEFT and Accelerate.

Only Tensor.is_cuda is overridden, and only while the eligibility predicate runs.
Fallbacks execute upstream model/loss/backward code. The supported-path test uses
a dense CPU stand-in for the fused operator to test dispatch/metrics, not Triton
arithmetic or CUDA correctness. No production device override is introduced.
"""

import copy
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import AutocastKwargs
from datasets import Dataset
from peft import LoraConfig, PromptTuningConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from trl import SFTConfig, SFTTrainer

from solution import head_loss


@pytest.fixture(autouse=True)
def isolated_runtime():
    # Accelerate shares state between instances, including the precision mode.
    accelerator_state = AcceleratorState._shared_state.copy()
    partial_state = PartialState._shared_state.copy()
    threads = torch.get_num_threads()
    AcceleratorState._reset_state(reset_partial_state=True)
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    AcceleratorState._reset_state(reset_partial_state=True)
    AcceleratorState._shared_state.update(accelerator_state)
    PartialState._shared_state.update(partial_state)
    torch.set_num_threads(threads)


@pytest.fixture
def installed(monkeypatch):
    original_loss = SFTTrainer.compute_loss
    # Register restoration before installing the real class patches.
    monkeypatch.setattr(Qwen2ForCausalLM, "forward", Qwen2ForCausalLM.forward)
    monkeypatch.setattr(SFTTrainer, "compute_loss", original_loss)
    monkeypatch.setattr(head_loss, "_STATE", {"armed": False, "step": None})
    head_loss._install()
    eligible = head_loss._eligible

    def cpu_eligible(trainer, inputs):
        with monkeypatch.context() as device_override:
            device_override.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
            return eligible(trainer, inputs)

    monkeypatch.setattr(head_loss, "_eligible", cpu_eligible)

    def unexpected_fusion(*args, **kwargs):
        pytest.fail("unsupported configuration reached the fused operator")

    monkeypatch.setattr(head_loss, "fused_head_loss", unexpected_fusion)
    return original_loss


def tiny_model(dtype=torch.float16):
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        attention_dropout=0.0, pad_token_id=0, use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).to(dtype)
    model.lm_head.requires_grad_(False)
    return model


def batch():
    ids = torch.tensor([[0, 0, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7]])
    labels = ids.clone()
    labels[0, :2] = -100
    labels[1, :3] = -100
    return {"input_ids": ids, "attention_mask": ids.ne(0).long(), "labels": labels}


def trainer_for(tmp_path, model, **kwargs):
    compute_loss_func = kwargs.pop("compute_loss_func", None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>")),
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    args = SFTConfig(
        output_dir=str(tmp_path), use_cpu=True, fp16=False,
        bf16=kwargs.pop("bf16", False), gradient_checkpointing=False,
        report_to="none", save_strategy="no", disable_tqdm=True,
        dataset_kwargs={"skip_prepare_dataset": True}, **kwargs,
    )
    return SFTTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"input_ids": [[2, 3, 4]]}),
        compute_loss_func=compute_loss_func,
    )


def assert_fallback(trainer, original_loss, inputs=None, context=nullcontext):
    """Compare loss, logits, gradients and all TRL metrics to unpatched TRL."""
    model = trainer.accelerator.prepare_model(trainer.model)
    inputs = batch() if inputs is None else inputs
    initial_metrics = copy.deepcopy(trainer._metrics)
    initial_tokens = trainer._total_train_tokens
    sentinel = object()
    setattr(trainer, head_loss.STEP_ATTR, sentinel)

    def run(loss_fn):
        model.zero_grad(set_to_none=True)
        with context():
            loss, outputs = loss_fn(
                trainer, model, copy.deepcopy(inputs), return_outputs=True,
                num_items_in_batch=11,
            )
            loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        assert grads
        return loss.detach(), outputs.logits.detach(), grads, copy.deepcopy(trainer._metrics)

    expected = run(original_loss)
    trainer._metrics = initial_metrics
    trainer._total_train_tokens = initial_tokens
    actual = run(SFTTrainer.compute_loss)
    torch.testing.assert_close(actual[:3], expected[:3], rtol=0, atol=0)
    assert actual[3] == expected[3]
    assert getattr(trainer, head_loss.STEP_ATTR) is sentinel
    assert head_loss._STATE == {"armed": False, "step": None}
    return actual


@pytest.mark.parametrize("kind", ["smoothing", "custom", "dft"])
def test_trainer_loss_that_removes_labels_falls_back(tmp_path, installed, kind):
    calls = []

    def custom(outputs, labels, num_items_in_batch=None):
        calls.append(labels)
        return 1.7 * ForCausalLMLoss(outputs.logits, labels, 32, num_items_in_batch)

    options = {"smoothing": {"label_smoothing_factor": 0.2},
               "custom": {"compute_loss_func": custom}, "dft": {"loss_type": "dft"}}[kind]
    trainer = trainer_for(tmp_path, tiny_model(), **options)
    assert_fallback(trainer, installed)
    if kind == "custom":
        assert len(calls) == 2 and all(torch.equal(x, batch()["labels"]) for x in calls)


@pytest.mark.parametrize("selection", [6, torch.arange(6)], ids=["nonzero-int", "tensor"])
def test_full_length_logit_selection_falls_back(tmp_path, installed, selection):
    # These selections keep all tokens, so upstream TRL's loss/metrics are valid.
    # The old guard arms fusion, then the forward declines or bool(Tensor) raises.
    inputs = {**batch(), "logits_to_keep": selection}
    assert_fallback(trainer_for(tmp_path, tiny_model()), installed, inputs)


@pytest.mark.parametrize("kind", ["bias", "subclass", "forward", "pre-hook", "post-hook", "backward-pre-hook", "backward-hook", "lora"])
def test_custom_head_behavior_and_gradients_are_preserved(tmp_path, installed, kind):
    model = tiny_model()
    head = model.lm_head
    if kind == "bias":
        head.bias = torch.nn.Parameter(torch.linspace(-1, 1, 32, dtype=head.weight.dtype))
    elif kind == "subclass":
        class ScaledLinear(torch.nn.Linear):
            def forward(self, inputs):
                return super().forward(inputs) * 1.7
        head.__class__ = ScaledLinear
    elif kind == "forward":
        original_forward = head.forward
        head.forward = lambda inputs: original_forward(inputs) * 1.7
    elif kind == "pre-hook":
        head.register_forward_pre_hook(lambda module, inputs: (inputs[0] * 1.7,))
    elif kind == "post-hook":
        head.register_forward_hook(lambda module, inputs, output: output * 1.7)
    elif kind == "backward-pre-hook":
        head.register_full_backward_pre_hook(lambda module, grad_out: (grad_out[0] * 1.7,))
    elif kind == "backward-hook":
        head.register_full_backward_hook(lambda module, grad_in, grad_out: (grad_in[0] * 1.7,))
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules=["lm_head"], r=2, lora_alpha=4,
        ))
        with torch.no_grad():
            model.get_base_model().lm_head.lora_B["default"].weight.fill_(0.2)
        assert not model.get_base_model().lm_head.weight.requires_grad
    _, _, grads, _ = assert_fallback(trainer_for(tmp_path, model), installed)
    if kind == "bias":
        assert grads["lm_head.bias"].abs().sum() > 0
    if kind == "lora":
        assert any("lora_B" in name and grad.abs().sum() > 0 for name, grad in grads.items())


def test_custom_model_loss_falls_back(tmp_path, installed):
    model = tiny_model()
    model.loss_function = lambda **kwargs: 1.7 * ForCausalLMLoss(**kwargs)
    assert_fallback(trainer_for(tmp_path, model), installed)


@pytest.mark.parametrize("kind", ["forward-pre", "forward", "backward-pre", "backward"])
def test_global_head_hooks_preserve_outputs_and_gradients(tmp_path, installed, monkeypatch, kind):
    from torch.nn.modules import module as module_hooks

    model = tiny_model()
    head = model.lm_head
    calls = []
    # Registration changes this global flag even after the handle is removed.
    monkeypatch.setattr(module_hooks, "_global_is_full_backward_hook",
                        module_hooks._global_is_full_backward_hook)

    def transform(module, values, *unused):
        if module is head:
            calls.append(kind)
            return (values[0] * 1.7,)

    def transform_output(module, inputs, output):
        if module is head:
            calls.append(kind)
            return output * 1.7

    register, callback = {
        "forward-pre": (module_hooks.register_module_forward_pre_hook, transform),
        "forward": (module_hooks.register_module_forward_hook, transform_output),
        "backward-pre": (module_hooks.register_module_full_backward_pre_hook, transform),
        "backward": (module_hooks.register_module_full_backward_hook, transform),
    }[kind]
    handle = register(callback)
    try:
        assert_fallback(trainer_for(tmp_path, model), installed)
        assert calls == [kind, kind]
    finally:
        handle.remove()


def test_prompt_tuning_keeps_virtual_token_metric_handling(tmp_path, installed):
    model = get_peft_model(tiny_model(), PromptTuningConfig(
        task_type="CAUSAL_LM", num_virtual_tokens=2,
    ))
    trainer = trainer_for(tmp_path, model)
    assert trainer.num_virtual_tokens == 2
    assert_fallback(trainer, installed)


def test_custom_ignore_index_falls_back(tmp_path, installed):
    inputs = batch()
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["ignore_index"] = 4
    assert_fallback(trainer_for(tmp_path, tiny_model()), installed, inputs)


def test_prepared_bf16_is_detected_outside_autocast(tmp_path, installed):
    trainer = trainer_for(tmp_path, tiny_model(torch.float32), bf16=True)
    # The effective Accelerate mode is decisive even if the args no longer agree.
    trainer.args.bf16 = False
    assert trainer.accelerator.native_amp
    assert trainer.accelerator.mixed_precision == "bf16"
    seen = []
    trainer.model.model.register_forward_pre_hook(
        lambda module, inputs: seen.append(torch.is_autocast_enabled("cpu")))
    assert not torch.is_autocast_enabled("cpu")
    assert_fallback(trainer, installed)
    assert seen == [True, True]


def test_ambient_bf16_falls_back(tmp_path, installed):
    trainer = trainer_for(tmp_path, tiny_model())
    assert_fallback(trainer, installed, context=lambda: torch.autocast("cpu", dtype=torch.bfloat16))


def test_prepared_wrapper_disabling_ambient_amp_falls_back(tmp_path, installed):
    trainer = trainer_for(tmp_path, tiny_model(torch.float32))
    trainer.accelerator.state._mixed_precision = "fp16"
    trainer.accelerator.native_amp = True
    trainer.accelerator.autocast_handler = AutocastKwargs(enabled=False)
    seen = []
    trainer.model.model.register_forward_pre_hook(
        lambda module, inputs: seen.append(torch.is_autocast_enabled("cpu")))
    assert_fallback(trainer, installed, context=lambda: torch.autocast("cpu", dtype=torch.float16))
    assert seen == [False, False]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_non_fp16_projection_without_amp_falls_back(tmp_path, installed, dtype):
    assert_fallback(trainer_for(tmp_path, tiny_model(dtype)), installed)


@pytest.mark.parametrize("prepared_amp", [False, True], ids=["fp16-storage", "prepared-fp16-amp"])
def test_supported_peft_path_keeps_normalization_metrics_and_rewind(tmp_path, installed, monkeypatch, prepared_amp):
    model = get_peft_model(tiny_model(torch.float32 if prepared_amp else torch.float16), LoraConfig(
        task_type="CAUSAL_LM", target_modules=["q_proj"], r=2, lora_alpha=4,
    ))
    trainer = trainer_for(tmp_path, model)
    if prepared_amp:
        # CPU Accelerate does not enable fp16 automatically. Exercise its actual
        # prepare_model autocast wrapper, configured as it is on an fp16 T4.
        trainer.accelerator.state._mixed_precision = "fp16"
        trainer.accelerator.native_amp = True
        trainer.args.fp16 = True
    prepared = trainer.accelerator.prepare_model(model)
    reference_loss, reference = installed(trainer, prepared, batch(), return_outputs=True, num_items_in_batch=11)
    reference_metrics = copy.deepcopy(trainer._metrics)
    trainer._metrics = {mode: type(metrics)(list) for mode, metrics in trainer._metrics.items()}
    trainer._total_train_tokens = 0
    calls = []

    def dense_operator(hidden, weight, target, denom):
        calls.append((hidden.shape[0], denom, torch.is_autocast_enabled("cpu")))
        logits = F.linear(hidden.half(), weight.half()).float()
        row = F.cross_entropy(logits, target, reduction="none", ignore_index=-100)
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        hit = logits.argmax(-1).eq(target).int()
        return row.sum() / denom, row.detach(), entropy.detach(), hit

    monkeypatch.setattr(head_loss, "fused_head_loss", dense_operator)
    loss, outputs = trainer.compute_loss(prepared, batch(), return_outputs=True, num_items_in_batch=11)
    loss.backward()
    assert calls == [(11, 11, prepared_amp)]  # includes left-pad row predicting first supervised token
    assert outputs.logits is None
    torch.testing.assert_close(loss, reference_loss)
    for metric in ("entropy", "mean_token_accuracy", "num_tokens"):
        # Without Accelerate's fp32 output conversion, TRL reduces entropy in
        # fp16; the fused path has always accumulated it in fp32.
        tolerance = 1e-6 if prepared_amp or metric != "entropy" else 2e-3
        assert trainer._metrics["train"][metric] == pytest.approx(reference_metrics["train"][metric], abs=tolerance)
    step = getattr(trainer, head_loss.STEP_ATTR)
    assert step.token_loss.shape == step.token_entropy.shape == (2, 6)
    assert step.labelled.item() == 7
    assert step.token_entropy[0, :2].eq(0).all()
    assert step.token_loss[0, 1] > 0
    assert head_loss._STATE == {"armed": False, "step": None}


def test_cpu_device_is_rejected_without_override(tmp_path):
    trainer = trainer_for(tmp_path, tiny_model())
    assert not head_loss._eligible(trainer, batch())
