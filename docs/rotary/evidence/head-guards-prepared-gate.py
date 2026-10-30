"""Actual prepared NF4/LoRA trainer gate; run in a fresh process, one visible T4."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

import torch
import bitsandbytes as bnb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BitsAndBytesConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import SFTConfig, SFTTrainer

ROOT = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from solution import head_loss
from soup_cli.trainer.rewind_hf import row_losses_from_token_losses
from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

assert torch.cuda.device_count() == 1
assert 'T4' in torch.cuda.get_device_name(0)
torch.manual_seed(83)
original_compute_loss = SFTTrainer.compute_loss
work = pathlib.Path(tempfile.mkdtemp(prefix='prepared-guard-'))
config = Qwen2Config(vocab_size=4096, hidden_size=128, intermediate_size=384,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=512, attention_dropout=0., pad_token_id=0, use_cache=False)
Qwen2ForCausalLM(config).save_pretrained(work / 'model')
model = Qwen2ForCausalLM.from_pretrained(
    work / 'model', device_map={'': 0}, attn_implementation='sdpa',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj'],
                                      r=16, lora_alpha=32, lora_dropout=0.05))
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if 'lora_B' in name:
            parameter.normal_(0, .001)
assert model.get_base_model().lm_head.weight.dtype == torch.float32
assert not model.get_base_model().lm_head.weight.requires_grad
# PEFT exposes the base weight through its wrapper; count physical BNB modules once.
quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
assert len(quantized) == 14
assert all(m.weight.quant_state.nested for m in quantized)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
    {'<pad>': 0, '<eos>': 1, '<unk>': 2}, unk_token='<unk>')),
    pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
trainer = SFTTrainer(model=model, processing_class=tokenizer,
    args=SFTConfig(output_dir=str(work / 'output'), fp16=True, bf16=False,
        gradient_checkpointing=False, report_to='none', save_strategy='no', disable_tqdm=True,
        dataset_kwargs={'skip_prepare_dataset': True}),
    train_dataset=Dataset.from_dict({'input_ids': [[2, 3, 4]]}))
# Match Soup's pre-training correction after TRL's QLoRA BF16 conversion.
align_trainable_dtype_for_fp16(trainer.model, fp16=trainer.args.fp16, bf16=trainer.args.bf16)
assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.02)
prepared = trainer.accelerator.prepare_model(model)
assert trainer.accelerator.mixed_precision == 'fp16'
assert trainer.accelerator.native_amp and trainer.accelerator.scaler is not None
prepared.train()
import solution
solution.apply()
parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
initial = {n: p.detach().clone() for n, p in parameters.items()}
metrics0 = copy.deepcopy(trainer._metrics)
tokens0 = trainer._total_train_tokens
ids = torch.randint(2, 4096, (2, 300), device='cuda')
mask = torch.ones_like(ids)
mask[0, :17] = 0
ids[0, :17] = 0
labels = ids.clone()
labels[0, :17] = -100
labels[1, :23] = -100
inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
target = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
assert (mask.bool() | target.ne(-100)).sum() > 512
assert target.ne(-100).sum() > 512
assert head_loss._eligible(trainer, inputs)
spec = importlib.util.spec_from_file_location("r8_reference", ROOT.parent / "baseline/solution/head_loss.py")
r8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r8)
original_operator = head_loss.fused_head_loss
launches = []
def observed(*args, **kwargs):
    launches.append({'rows': args[0].shape[0], 'dtype': str(args[0].dtype),
                     'amp': torch.is_autocast_enabled('cuda')})
    return original_operator(*args, **kwargs)
head_loss.fused_head_loss = observed
rng_cpu = torch.get_rng_state()
rng_cuda = torch.cuda.get_rng_state()


def run(fused, scale):
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(initial[name])
    optimizer.zero_grad(set_to_none=True)
    trainer._metrics = copy.deepcopy(metrics0)
    trainer._total_train_tokens = tokens0
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_cuda)
    trainer.accelerator.scaler = torch.amp.GradScaler('cuda', init_scale=scale, growth_interval=1000)
    loss_fn = SFTTrainer.compute_loss if fused else original_compute_loss
    loss, output = loss_fn(trainer, prepared, copy.deepcopy(inputs), return_outputs=True,
                           num_items_in_batch=4096)
    if fused:
        assert output.logits is None
        step = getattr(trainer, head_loss.STEP_ATTR)
        rows, counts = row_losses_from_token_losses(step.token_loss, labels)
        entropy = step.token_entropy.detach().clone()
        assert entropy[0, :17].eq(0).all()
        assert step.token_loss[0, 16] > 0
    else:
        logits = output.logits.detach().float()
        row_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4096), target.flatten(),
                                                     ignore_index=-100, reduction='none').reshape_as(labels)
        rows, counts = row_losses_from_token_losses(row_loss, labels)
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1) * mask
    trainer.accelerator.backward(loss)
    scaled = {n: p.grad.detach().clone() for n, p in parameters.items()}
    assert all(torch.isfinite(g).all() for g in scaled.values())
    trainer.accelerator.scaler.unscale_(optimizer)
    grads = {n: p.grad.detach().clone() for n, p in parameters.items()}
    trainer.accelerator.scaler.step(optimizer)
    trainer.accelerator.scaler.update()
    assert trainer.accelerator.scaler.get_scale() == scale
    updates = {n: p.detach() - initial[n] for n, p in parameters.items()}
    return dict(loss=loss.detach(), grads=grads, scaled=scaled, updates=updates,
                weights={n:p.detach().clone() for n,p in parameters.items()},
                rows=torch.tensor(rows), counts=counts, entropy=entropy,
                metrics=copy.deepcopy(trainer._metrics), rng=torch.cuda.get_rng_state())


dense_failures = []
for scale in (65536., 1.):
    reference = run(False, scale)
    actual = run(True, scale)
    head_loss.fused_head_loss = r8.fused_head_loss
    r8_actual = run(True, scale)
    head_loss.fused_head_loss = observed
    for key in ('loss', 'grads', 'scaled', 'updates', 'weights', 'rows', 'entropy'):
        torch.testing.assert_close(actual[key], r8_actual[key], rtol=0, atol=0,
                                   msg=lambda msg: 'Guard revision differs from R8: ' + msg)
    assert actual['counts'] == r8_actual['counts']
    assert actual['metrics'] == r8_actual['metrics']
    assert torch.equal(actual['rng'], r8_actual['rng'])
    print(json.dumps(dict(event='r8_equivalence', scale=scale, status='exact')), flush=True)
    assert actual['counts'] == reference['counts']
    assert torch.equal(actual['rng'], reference['rng'])
    for field in ('loss', 'rows', 'entropy'):
        torch.testing.assert_close(actual[field], reference[field], rtol=2e-4, atol=2e-4)
    max_relative = 0.
    for field in ('grads', 'scaled', 'updates'):
        for name, expected in reference[field].items():
            observed_value = actual[field][name]
            magnitude = expected.abs().max().item()
            print(json.dumps(dict(event='comparison', scale=scale, field=field, parameter=name,
                                  reference_max=magnitude,
                                  max_error=(observed_value-expected).abs().max().item(),
                                  dtype=str(expected.dtype))), flush=True)
            try:
                if field == 'updates':
                    # Subtracting two rounded stored weights magnifies relative
                    # error for tiny updates. Bound SGD gradient error plus the
                    # rounding of each stored weight; gradient gates stay strict.
                    def spacing(value):
                        return torch.maximum(
                            (torch.nextafter(value, torch.full_like(value, float('inf'))) - value).abs(),
                            (torch.nextafter(value, torch.full_like(value, -float('inf'))) - value).abs())
                    bound = (.02 * (actual['grads'][name] - reference['grads'][name]).abs()
                             + spacing(actual['weights'][name]) + spacing(reference['weights'][name])
                             + spacing(observed_value) + spacing(expected))
                    assert torch.all((observed_value - expected).abs() <= bound), \
                        f'Update exceeds SGD gradient-error plus storage-ULP bound: {name}'
                else:
                    torch.testing.assert_close(observed_value, expected, rtol=.02, atol=magnitude * .002,
                                               msg=lambda msg: f'{field}: {name}, scale={scale}\n{msg}')
            except AssertionError as exc:
                dense_failures.append(dict(scale=scale, field=field, parameter=name,
                                           message=str(exc)))
                print(json.dumps(dict(event='dense_mismatch', scale=scale, field=field,
                                      parameter=name, message=str(exc))), flush=True)
            if magnitude:
                max_relative = max(max_relative, ((observed_value - expected).norm() / expected.norm()).item())
    for metric in ('entropy', 'mean_token_accuracy', 'num_tokens'):
        a = torch.tensor(actual['metrics']['train'][metric])
        b = torch.tensor(reference['metrics']['train'][metric])
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    print(json.dumps(dict(event='prepared_gate', scale=scale, status='dense_mismatch_recorded' if any(x['scale']==scale for x in dense_failures) else 'passed',
                         loss=actual['loss'].item(), parameters=len(parameters),
                         rewind_counts=actual['counts'], max_relative_l2=max_relative)), flush=True)
assert len(launches) == 2 and all(x['amp'] and x['rows'] > 512 for x in launches)
assert head_loss._STATE == {'armed': False, 'step': None}
assert not any(x['scale'] == 65536. for x in dense_failures), 'Scaled training oracle mismatch; final gate remains blocked'
print(json.dumps(dict(event='summary', status='passed_with_recorded_unscaled_limit' if dense_failures else 'passed', dense_mismatches=dense_failures, quantized_modules=len(quantized),
                     dispatch=launches, source_sha256=hashlib.sha256((ROOT/'solution/head_loss.py').read_bytes()).hexdigest())), flush=True)
