"""Exercise actual pinned Qwen2, PEFT, TRL, HF loss and Soup Rewind APIs."""

import copy

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import SFTConfig, SFTTrainer

import solution
from solution.integration import _ACTIVE, attach_model
from soup_cli.trainer.rewind_hf import (
    attach_rewind_state, make_rewind_trainer_class, row_losses,
    row_losses_from_token_losses,
)


@pytest.fixture
def trainer(tmp_path, request):
    solution.apply()
    torch.manual_seed(812)
    config = Qwen2Config(vocab_size=47, hidden_size=24, intermediate_size=40,
                        num_hidden_layers=2, num_attention_heads=3,
                        num_key_value_heads=1, max_position_embeddings=64,
                        tie_word_embeddings=True, pad_token_id=0, eos_token_id=1)
    model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=.03)
    assert attach_model(model)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>")),
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    precision = getattr(request, "param", "cpu_fp32")
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=precision != "cuda_fp16", bf16=precision == "cpu_bf16",
                     fp16=precision == "cuda_fp16", report_to="none", max_steps=1, save_strategy="no",
                     per_device_train_batch_size=2, dataloader_pin_memory=False,
                     dataset_kwargs={"skip_prepare_dataset": True})
    data = Dataset.from_dict({"input_ids": [[3, 4, 5, 1], [6, 7, 8, 1]],
                              "attention_mask": [[1]*4, [1]*4],
                              "labels": [[-100, -100, 5, 1], [-100, 7, 8, 1]]})
    cls = make_rewind_trainer_class(SFTTrainer)
    instance = cls(model=model, args=args, train_dataset=data, processing_class=tokenizer)
    instance.model.train()
    return instance


def batch():
    ids = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 0, 0], [13, 14, 15, 0, 0, 0]])
    labels = ids.clone()
    labels[0, :3] = -100
    labels[1, :2] = -100
    labels[1, 4:] = -100
    labels[2] = -100
    return {"input_ids": ids, "labels": labels, "attention_mask": (ids != 0).long()}


@pytest.mark.parametrize("denominator", [None, 19])
def test_trl_peft_outputs_metrics_and_adapter_gradients(trainer, denominator):
    reference_model = copy.deepcopy(trainer.model)
    original_compute = SFTTrainer.compute_loss.__wrapped__
    torch.manual_seed(44)
    baseline_loss, baseline_output = original_compute(
        trainer, reference_model, batch(), return_outputs=True, num_items_in_batch=denominator)
    before = {key: values[-1] for key, values in trainer._metrics["train"].items()}
    torch.manual_seed(44)
    loss, output = trainer.compute_loss(trainer.model, batch(), return_outputs=True,
                                        num_items_in_batch=denominator)
    assert output.logits is None
    assert baseline_output.logits.shape == (3, 6, 47)
    torch.testing.assert_close(loss, baseline_loss, rtol=3e-6, atol=1e-6)
    after = {key: values[-1] for key, values in trainer._metrics["train"].items()}
    assert after["entropy"] == pytest.approx(before["entropy"], abs=2e-6)
    assert after["mean_token_accuracy"] == before["mean_token_accuracy"]
    assert after["num_tokens"] == 2 * before["num_tokens"] == 26
    expected_rows, expected_counts = row_losses(baseline_output.logits, batch()["labels"])
    actual_rows, actual_counts = row_losses_from_token_losses(trainer._soup_chunked_step.token_loss, batch()["labels"])
    assert actual_counts == expected_counts == [3, 2, 0]
    assert actual_rows == pytest.approx(expected_rows, abs=2e-6)
    (loss * 65536).backward()
    (baseline_loss * 65536).backward()
    for (name, param), (other_name, expected) in zip(trainer.model.named_parameters(), reference_model.named_parameters()):
        assert name == other_name
        if param.requires_grad:
            assert param.grad is not None
            torch.testing.assert_close(param.grad / 65536, expected.grad / 65536, rtol=2e-4, atol=1e-7)
    assert _ACTIVE.get() is None


def test_eval_returns_real_logits_and_clears_rewind_state(trainer):
    trainer.compute_loss(trainer.model, batch())
    assert trainer._soup_chunked_step is not None
    trainer.model.eval()
    with torch.no_grad():
        _, output = trainer.compute_loss(trainer.model, batch(), return_outputs=True)
    assert output.logits.shape == (3, 6, 47)
    assert trainer._soup_chunked_step is None
    assert _ACTIVE.get() is None


def test_exception_restores_normal_model_calls(trainer):
    invalid = batch()
    invalid["labels"][0, 3] = 47
    with pytest.raises(IndexError):
        trainer.compute_loss(trainer.model, invalid)
    assert _ACTIVE.get() is None
    assert trainer._soup_chunked_step is None
    normal = trainer.model(**batch())
    assert normal.logits.shape == (3, 6, 47)


def test_rewind_one_record_per_microbatch(trainer):
    class Sink:
        def __init__(self):
            self.records = []
        def record_batch(self, **record):
            self.records.append(record)
    sink = Sink()
    state = attach_rewind_state(trainer, sink)
    for idx in [31, 7, 10, 5, 2, 4]:
        state.on_index(idx)
    trainer.compute_loss(trainer.model, batch())
    trainer.compute_loss(trainer.model, batch())
    assert not state.failed
    assert len(sink.records) == 2
    assert list(sink.records[0]["rows"]) == [31, 7, 10]
    assert list(sink.records[1]["rows"]) == [5, 2, 4]
    assert [r["micro"] for r in sink.records] == [0, 1]
    assert all(list(r["row_tokens"]) == [3, 2, 0] for r in sink.records)


def test_actual_training_step_and_adapter_save(trainer, tmp_path):
    trainer.train()
    assert trainer.state.global_step == 1
    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()


def test_apply_idempotent():
    solution.apply()
    original = SFTTrainer.compute_loss
    solution.apply()
    assert SFTTrainer.compute_loss is original
    assert attach_model(object()) is False


@pytest.mark.parametrize("trainer", ["cpu_bf16"], indirect=True)
def test_prepared_bf16_training_uses_original_loss(trainer):
    trainer.train()
    assert trainer.state.global_step == 1
    assert trainer._soup_chunked_step is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA AMP and GradScaler")
@pytest.mark.parametrize("trainer", ["cuda_fp16"], indirect=True)
def test_prepared_cuda_fp16_training_and_scaler(trainer):
    trainer.train()
    assert trainer.state.global_step == 1
    assert trainer._soup_chunked_step is not None
    assert trainer.accelerator.scaler is not None
    assert torch.isfinite(trainer._soup_chunked_step.loss)
