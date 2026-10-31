#!/usr/bin/env python3
"""Standalone numerical regression tests for a candidate head_loss.py.

CPU: TRITON_INTERPRET=1 is set BEFORE torch/Triton/candidate imports. Every
candidate launch executes the actual Triton kernel; projection and dH use CPU
fp16 matmul. This establishes numerical/semantic evidence, not CUDA codegen,
performance, memory, or AMP evidence. CUDA: interpreter disabled, real kernels
and a real torch.amp.GradScaler + fp32 parameter are exercised.

The oracle is dense fp16 projection followed by fp32 cross_entropy and PyTorch
autograd. It is never substituted for candidate kernel math. The existing
exp(logits - lse) gradient convention is intentionally preserved; reduction
order can differ from CE. Random-gradient tolerance allows small fp16/reduction
rounding differences, while the two underflow witnesses use tight tolerances
and explicit nonzero assertions. Only CPU's hardware eligibility check is
bypassed in the installed trainer-hook test; all kernel and model math is real.

Example:
  python -B test_head_loss_regression.py --head-loss ./head_loss.py --device cpu
Use --device cuda on a CUDA host; no CUDA evidence is implied by CPU passes.
"""
import argparse
import ast
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from collections import defaultdict
from unittest.mock import patch

sys.dont_write_bytecode = True
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--head-loss', required=True, type=Path)
PARSER.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
ARGS, UNITTEST_ARGS = PARSER.parse_known_args()
os.environ['TRITON_INTERPRET'] = '1' if ARGS.device == 'cpu' else '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import torch
import torch.nn.functional as F

if ARGS.device == 'cpu':
    # NumPy >= 2.4 rejects int(array([value])). Triton's interpreter currently
    # uses that conversion for scalar loop bounds. Repair only Python scalar
    # extraction; the candidate source, tl operations, stores and GEMMs stay real.
    import numpy as np
    import triton.runtime.interpreter as interpreter
    try:
        int(np.array([1]))
    except TypeError:
        _original_patch_lang_tensor = interpreter._patch_lang_tensor
        def _numpy_compatible_patch_lang_tensor(tensor_class, scope):
            _original_patch_lang_tensor(tensor_class, scope)
            scope.set_attr(tensor_class, '__index__', lambda self: int(self.handle.data.item()))
        interpreter._patch_lang_tensor = _numpy_compatible_patch_lang_tensor
        print('CPU HARNESS: NumPy scalar-index compatibility fix; kernel math unchanged', flush=True)

if ARGS.device == 'cuda' and not torch.cuda.is_available():
    PARSER.error('--device cuda requires an available CUDA device')
torch.set_num_threads(1)
DEVICE = torch.device(ARGS.device)
spec = importlib.util.spec_from_file_location('head_loss_candidate', ARGS.head_loss.resolve())
if spec is None or spec.loader is None:
    PARSER.error('cannot load --head-loss module')
HEAD = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = HEAD
spec.loader.exec_module(HEAD)
# Use small blocks only in the interpreter. CUDA retains production launch geometry.
if ARGS.device == 'cpu':
    HEAD.BLOCK = 32
TEST_VOCAB = 73 if ARGS.device == 'cpu' else 5003

# Retain a separately imported released implementation for behavior comparisons.
BASE_PATH = Path(__file__).with_name('head_loss.correct-r6.py')
base_spec = importlib.util.spec_from_file_location('head_loss_released_r6', BASE_PATH)
BASE = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = BASE
base_spec.loader.exec_module(BASE)
BASE.BLOCK = HEAD.BLOCK


def dense_oracle(hidden, weight, target, denom, upstream=1.0):
    h = hidden.detach().clone().requires_grad_(True)
    logits = (h.half() @ weight.detach().half().t()).float()
    row = F.cross_entropy(logits, target, ignore_index=-100, reduction='none')
    loss = row.sum() / torch.as_tensor(denom, device=DEVICE, dtype=torch.float32)
    grad, = torch.autograd.grad(loss, h, grad_outputs=loss.new_tensor(upstream))
    logp = logits.log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    hit = (logits.argmax(-1) == target) & target.ne(-100)
    return loss.detach(), row.detach(), entropy.detach(), hit, grad


def tensor(data, dtype=torch.float32):
    return torch.tensor(data, device=DEVICE, dtype=dtype)


@contextlib.contextmanager
def hook_dependencies():
    """Real lightweight Qwen2 when installed; dependency-only fallback otherwise."""
    try:
        import trl
        from transformers import Qwen2Config, Trainer
        from transformers.models.qwen2 import modeling_qwen2 as mq
        # Resolve lazy imports here so missing optional dependencies use fallback.
        trainer_class = trl.SFTTrainer
        model_class = mq.Qwen2ForCausalLM
    except ImportError as exc:
        print(f'HOOK EVIDENCE: dependency stubs ({exc}); real candidate hooks/kernel', flush=True)
        class Output(dict):
            def __init__(self, **kw):
                super().__init__(kw)
                self.__dict__.update(kw)
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(73, 16)
            def forward(self, input_ids=None, inputs_embeds=None, **kw):
                h = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
                return Output(last_hidden_state=h, past_key_values=None,
                              hidden_states=None, attentions=None)
        class Qwen2ForCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()
                self.lm_head = torch.nn.Linear(16, 73, bias=False)
            def forward(self, **kw):
                raise AssertionError('installed candidate forward was not taken')
        class Trainer:
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                out = model(**inputs, num_items_in_batch=num_items_in_batch)
                return (out.loss, out) if return_outputs else out.loss
        class SFTTrainer(Trainer):
            pass
        trl = types.ModuleType('trl')
        trl.SFTTrainer = SFTTrainer
        transformers = types.ModuleType('transformers')
        transformers.Trainer = Trainer
        outputs = types.ModuleType('transformers.modeling_outputs')
        outputs.CausalLMOutputWithPast = Output
        mq = types.ModuleType('transformers.models.qwen2.modeling_qwen2')
        mq.Qwen2ForCausalLM = Qwen2ForCausalLM
        qwen2 = types.ModuleType('transformers.models.qwen2')
        qwen2.modeling_qwen2 = mq
        models = types.ModuleType('transformers.models')
        models.qwen2 = qwen2
        replacements = {'trl': trl, 'transformers': transformers,
                        'transformers.modeling_outputs': outputs,
                        'transformers.models': models, 'transformers.models.qwen2': qwen2,
                        'transformers.models.qwen2.modeling_qwen2': mq}
        with patch.dict(sys.modules, replacements):
            yield trl, mq, Qwen2ForCausalLM().to(DEVICE), False
    else:
        config = Qwen2Config(vocab_size=73, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=2, max_position_embeddings=32,
                            attention_dropout=0.0, use_cache=False, pad_token_id=0)
        config._attn_implementation = 'eager'
        model = model_class(config).to(DEVICE)
        print('HOOK EVIDENCE: real tiny Qwen2, real Trainer.compute_loss', flush=True)
        yield trl, mq, model, True


class HeadLossRegression(unittest.TestCase):
    def assert_close(self, actual, expected, *, atol=1e-3, rtol=1e-3):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, check_dtype=False)

    def test_only_backward_changed_from_released_source(self):
        def without_backward(path):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == '_FusedHeadLoss':
                    node.body = [x for x in node.body if getattr(x, 'name', '') != 'backward']
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(without_backward(ARGS.head_loss), without_backward(BASE_PATH))

    def test_mask_densities_random_rows_tails_scales_and_noncontiguous_hidden(self):
        torch.manual_seed(8308)
        n, width, chunk = 101, 16, 7
        weight = torch.randn(TEST_VOCAB, width, device=DEVICE) * .01
        for fraction in (0., .01, .25, 1.):
            count = round(n * fraction)
            indices = torch.randperm(n, device=DEVICE)[:count]
            target = torch.full((n,), -100, dtype=torch.long, device=DEVICE)
            target[indices] = torch.randint(TEST_VOCAB, (count,), device=DEVICE)
            denom = count + 3
            for dtype in (torch.float16, torch.float32):
                # Transpose keeps a non-contiguous fp16 saved hidden as well.
                hidden = torch.randn(width, n, device=DEVICE, dtype=dtype).t().requires_grad_()
                self.assertFalse(hidden.is_contiguous())
                loss, row, entropy, hit = HEAD.fused_head_loss(hidden, weight, target, denom, chunk)
                for scale in (1., 65536.):
                    with self.subTest(fraction=fraction, dtype=dtype, scale=scale):
                        grad, = torch.autograd.grad(loss, hidden, grad_outputs=loss.new_tensor(scale),
                                                    retain_graph=scale == 1.)
                        expected = dense_oracle(hidden, weight, target, denom, scale)
                        for actual, ref in zip((loss, row, entropy), expected[:3]):
                            self.assert_close(actual, ref)
                        self.assertTrue(torch.equal(hit.bool(), expected[3]))
                        self.assertEqual(grad.shape, hidden.shape)
                        self.assertEqual(grad.dtype, dtype)
                        self.assertTrue(torch.isfinite(grad).all().item())
                        self.assertEqual(torch.count_nonzero(grad[target < 0]).item(), 0)
                        self.assert_close(grad, expected[-1], rtol=.02,
                                          atol=max(2e-6, expected[-1].abs().max().item() * .002))

    def test_backward_kernel_rows_are_only_supervised_and_bounded(self):
        original = HEAD.head_loss_kernel
        launches = []
        class ObservedKernel:
            def __getitem__(self, grid):
                real_launch = original[grid]
                def launch(*args, **kwargs):
                    if kwargs['WITH_GRAD']:
                        launches.append((grid[0], args[1].detach().clone()))
                    return real_launch(*args, **kwargs)
                return launch
        torch.manual_seed(82)
        target = torch.full((23,), -100, device=DEVICE, dtype=torch.long)
        indices = tensor([22, 1, 15, 5, 9, 17, 11], torch.long)
        target[indices] = torch.arange(7, device=DEVICE)
        hidden = torch.randn(23, 8, device=DEVICE, requires_grad=True)
        weight = torch.randn(TEST_VOCAB, 8, device=DEVICE) * .01
        with patch.object(HEAD, 'head_loss_kernel', ObservedKernel()):
            loss, *_ = HEAD.fused_head_loss(hidden, weight, target, 7, chunk=3)
            loss.backward()
            self.assertEqual([size for size, _ in launches], [3, 3, 1])
            self.assertTrue(torch.equal(torch.cat([t for _, t in launches]), target[target >= 0]))
            launches.clear()
            zero_loss, *_ = HEAD.fused_head_loss(hidden, weight, target * 0 - 100, 0, chunk=3)
            hidden.grad = None
            zero_loss.backward()
            self.assertEqual(launches, [])
            self.assertEqual(torch.count_nonzero(hidden.grad).item(), 0)

    def test_all_negative_kernel_validity_matches_released_and_normalized_oracle(self):
        hidden = tensor([[1., .3], [.2, -.2], [.4, 1.], [.3, .8]]).requires_grad_()
        weight = tensor([[.001, .002], [-.002, -.001], [.003, -.004]])
        target = tensor([-1, 0, -7, -100], torch.long)
        for scale in (1., 65536., -3., 0.):
            with self.subTest(scale=scale):
                loss, *_ = HEAD.fused_head_loss(hidden, weight, target, 3, chunk=2)
                grad, = torch.autograd.grad(loss, hidden, grad_outputs=loss.new_tensor(scale))
                old_loss, *_ = BASE.fused_head_loss(hidden, weight, target, 3, chunk=2)
                old_grad, = torch.autograd.grad(old_loss, hidden, grad_outputs=loss.new_tensor(scale))
                normalized = target.masked_fill(target < 0, -100)
                ref = dense_oracle(hidden, weight, normalized, 3, scale)
                self.assert_close(grad, old_grad, atol=2e-7, rtol=.002)
                self.assert_close(grad, ref[-1], atol=2e-7, rtol=.002)
                self.assertEqual(torch.count_nonzero(grad[target < 0]).item(), 0)

    def test_target_mutation_saved_version_semantics_match_released(self):
        for module in (BASE, HEAD):
            for all_ignored in (False, True):
                with self.subTest(module=module.__name__, all_ignored=all_ignored):
                    hidden = tensor([[.1, .2], [.3, .4]]).requires_grad_()
                    weight = tensor([[.001, .002], [-.002, .001]])
                    target = tensor([-100, -100] if all_ignored else [0, -100], torch.long)
                    loss, *_ = module.fused_head_loss(hidden, weight, target, 2, chunk=1)
                    loss.backward(retain_graph=True)
                    target[0] = 1
                    with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
                        loss.backward()
            # Public wrapper copies non-contiguous targets, so mutating the
            # original view after forward must leave its saved snapshot intact.
            hidden = tensor([[.1, .2], [.3, .4]]).requires_grad_()
            weight = tensor([[.001, .002], [-.002, .001]])
            storage = tensor([0, 99, -100, 99], torch.long)
            target = storage[::2]
            self.assertFalse(target.is_contiguous())
            snapshot = target.clone()
            loss, *_ = module.fused_head_loss(hidden, weight, target, 2, chunk=1)
            target.fill_(1)
            loss.backward()
            self.assert_close(hidden.grad, dense_oracle(hidden, weight, snapshot, 2)[-1],
                              atol=2e-7, rtol=.002)

    def test_selected_chunk_boundaries_strided_hidden_repeated_backward_vs_released(self):
        torch.manual_seed(813)
        n, chunk = 21, 7
        weight = torch.randn(TEST_VOCAB, 8, device=DEVICE) * .01
        for count in (0, 1, chunk - 1, chunk, chunk + 1, 2 * chunk + 1):
            with self.subTest(supervised=count):
                hidden = torch.randn(n, 16, device=DEVICE, dtype=torch.float16)[:, ::2].requires_grad_()
                self.assertFalse(hidden.is_contiguous())
                target = torch.full((n,), -100, dtype=torch.long, device=DEVICE)
                indices = torch.randperm(n, device=DEVICE)[:count]
                target[indices] = torch.randint(TEST_VOCAB, (count,), device=DEVICE)
                actual = HEAD.fused_head_loss(hidden, weight, target, count + 3, chunk=chunk)
                released = BASE.fused_head_loss(hidden, weight, target, count + 3, chunk=chunk)
                for a, b in zip(actual, released):
                    self.assert_close(a, b, rtol=0., atol=0.)
                for scale in (1., 65536., -3., 0.):
                    grad, = torch.autograd.grad(actual[0], hidden, retain_graph=scale != 0.,
                                                grad_outputs=actual[0].new_tensor(scale))
                    ref, = torch.autograd.grad(released[0], hidden, retain_graph=scale != 0.,
                                               grad_outputs=released[0].new_tensor(scale))
                    self.assertEqual(torch.count_nonzero(grad[target < 0]).item(), 0)
                    self.assert_close(grad, ref, rtol=.02,
                                      atol=max(2e-6, ref.abs().max().item() * .002))

    def test_contiguous_target_mutation_before_first_backward_raises(self):
        for all_ignored in (False, True):
            hidden = tensor([[1., 0.], [0., 1.]]).requires_grad_()
            target = tensor([-100, -100] if all_ignored else [0, -100], torch.long)
            loss, *_ = HEAD.fused_head_loss(hidden, torch.eye(2, device=DEVICE), target, 2, chunk=1)
            target[1] = 0
            with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
                loss.backward()

    def test_empty_rows_preserve_shape_dtype_and_zero_gradient(self):
        for dtype in (torch.float16, torch.float32):
            hidden = torch.empty((0, 8), dtype=dtype, device=DEVICE, requires_grad=True)
            weight = torch.zeros((TEST_VOCAB, 8), device=DEVICE)
            target = torch.empty((0,), dtype=torch.long, device=DEVICE)
            loss, row, entropy, hit = HEAD.fused_head_loss(hidden, weight, target, 0, chunk=3)
            loss.backward()
            self.assertTrue(torch.isnan(loss).item())
            self.assertEqual(hidden.grad.shape, hidden.shape)
            self.assertEqual(hidden.grad.dtype, dtype)
            self.assertTrue(all(x.numel() == 0 for x in (row, entropy, hit)))

    def test_actual_kernel_and_nondifferentiable_metrics(self):
        self.assertTrue(hasattr(HEAD.head_loss_kernel, '__getitem__'))
        if ARGS.device == 'cpu':
            self.assertEqual(type(HEAD.head_loss_kernel).__name__, 'InterpretedFunction',
                             'CPU must run the actual Triton interpreter kernel')
        h = tensor([[1., 0.], [0., 1.]]).requires_grad_()
        w = tensor([[0., 1.], [-2., 0.], [1., -1.]])
        loss, row, ent, hit = HEAD.fused_head_loss(h, w, tensor([0, -100], torch.long), 1, chunk=1)
        self.assertTrue(loss.requires_grad)
        for metric in (row, ent, hit):
            self.assertFalse(metric.requires_grad)
        self.assertEqual(row[1].item(), 0)
        self.assertEqual(hit[1].item(), 0)

    def test_large_upstream_recovers_exp_minus20_before_fp16_store(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                h = tensor([[1., 0.]], dtype).requires_grad_()
                w = tensor([[0., 0.], [-20., 1.]])
                tgt = tensor([0], torch.long)
                loss, *_ = HEAD.fused_head_loss(h, w, tgt, 1, chunk=1)
                (loss * 65536.).backward()
                ref = dense_oracle(h, w, tgt, 1, 65536.)[-1]
                self.assertEqual(h.grad.dtype, dtype)
                self.assertTrue(torch.isfinite(h.grad).all().item())
                self.assertGreater(h.grad[0, 1].item(), 1e-4)
                self.assert_close(h.grad, ref, atol=2e-7, rtol=2e-3)

    def test_small_weight_recovers_dh_before_fp16_gemm_rounding(self):
        h = tensor([[1., 0.]]).requires_grad_()
        w = tensor([[0., 0.], [-10., 1e-4]])
        tgt = tensor([0], torch.long)
        loss, *_ = HEAD.fused_head_loss(h, w, tgt, 1, chunk=1)
        (loss * 65536.).backward()
        ref = dense_oracle(h, w, tgt, 1, 65536.)[-1]
        self.assertGreater(h.grad[0, 1].item(), 2e-4)
        self.assert_close(h.grad, ref, atol=2e-7, rtol=2e-3)

    def test_random_multichunk_fp16_projection_fp32_ce(self):
        torch.manual_seed(4901)
        w = torch.randn(TEST_VOCAB, 16, device=DEVICE) * .2
        tgt = torch.randint(0, TEST_VOCAB, (13,), device=DEVICE)
        tgt[::4] = -100
        denom = int(tgt.ne(-100).sum()) + 3  # Exercise externally supplied normalization.
        for dtype in (torch.float16, torch.float32):
            for upstream in (.125, 3., 65536.):
                with self.subTest(dtype=dtype, upstream=upstream):
                    h = torch.randn(13, 16, device=DEVICE, dtype=dtype).requires_grad_()
                    loss, row, ent, hit = HEAD.fused_head_loss(h, w, tgt, denom, chunk=4)
                    (loss * upstream).backward()
                    ref_loss, ref_row, ref_ent, ref_hit, ref_grad = dense_oracle(h, w, tgt, denom, upstream)
                    self.assert_close(loss, ref_loss)
                    self.assert_close(row, ref_row)
                    self.assert_close(ent, ref_ent)
                    self.assertTrue(torch.equal(hit.bool(), ref_hit))
                    self.assertEqual(h.grad.dtype, dtype)
                    self.assertTrue(torch.isfinite(h.grad).all().item())
                    self.assertEqual(torch.count_nonzero(h.grad[tgt == -100]).item(), 0)
                    # FP16 GEMM plus different fp32 logsumexp reduction order: up
                    # to 0.2% of max magnitude near cancellation, 2% relative.
                    atol = max(2e-6, ref_grad.abs().max().item() * .002)
                    self.assert_close(h.grad, ref_grad, atol=atol, rtol=.02)

    def test_all_ignored_zero_denom_nan_loss_zero_gradients(self):
        for upstream in (1., 65536., 0., float('nan'), float('inf')):
            with self.subTest(upstream=upstream):
                h = tensor([[1., 2.], [-1., 3.], [0., 1.]]).requires_grad_()
                w = tensor([[0., 1.], [-1., 0.], [1., -1.]])
                tgt = torch.full((3,), -100, device=DEVICE, dtype=torch.long)
                loss, row, ent, hit = HEAD.fused_head_loss(h, w, tgt, 0, chunk=2)
                self.assertTrue(torch.isnan(loss).item())
                loss.backward(gradient=loss.new_tensor(upstream))
                self.assertTrue(torch.isfinite(h.grad).all().item())
                self.assertEqual(torch.count_nonzero(h.grad).item(), 0)
                self.assertEqual(torch.count_nonzero(row).item(), 0)
                self.assertEqual(torch.count_nonzero(hit).item(), 0)
                self.assertTrue(torch.isfinite(ent).all().item())

    def test_repeated_backward_uses_each_upstream_with_retain_graph(self):
        h = tensor([[1., 0.], [.5, 0.]]).requires_grad_()
        w = tensor([[0., 0.], [-20., 1e-4]])
        tgt = tensor([0, 0], torch.long)
        loss, *_ = HEAD.fused_head_loss(h, w, tgt, 2, chunk=1)
        scales = [65536., 1., 17., 0., -3.]
        for index, upstream in enumerate(scales):
            with self.subTest(upstream=upstream):
                h.grad = None
                loss.backward(loss.new_tensor(upstream), retain_graph=index < len(scales)-1)
                ref = dense_oracle(h, w, tgt, 2, upstream)[-1]
                self.assert_close(h.grad, ref, atol=2e-7, rtol=3e-3)

    def test_no_grad_multichunk_metrics(self):
        torch.manual_seed(13)
        h = torch.randn(7, 16, device=DEVICE, requires_grad=True)
        w = torch.randn(TEST_VOCAB, 16, device=DEVICE) * .1
        tgt = tensor([3, -100, 72, 0, 31, -100, 4], torch.long)
        ref_loss, ref_row, ref_ent, ref_hit, _ = dense_oracle(h, w, tgt, 8)
        with torch.no_grad():
            loss, row, ent, hit = HEAD.fused_head_loss(h, w, tgt, 8, chunk=3)
        self.assertFalse(loss.requires_grad)
        self.assert_close(loss, ref_loss)
        self.assert_close(row, ref_row)
        self.assert_close(ent, ref_ent)
        self.assertTrue(torch.equal(hit.bool(), ref_hit))

    def test_installed_hooks_shifted_labels_left_padding_metrics(self):
        torch.manual_seed(31)
        with hook_dependencies() as (trl, mq, model, real_dependencies):
            model.train()
            model.lm_head.weight.requires_grad_(False)
            ids = tensor([[0, 0, 7, 8, 9], [0, 4, 5, 6, 0]], torch.long)
            am = tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]], torch.long)
            labels = tensor([[-100, -100, 7, 8, 9], [-100, 4, 5, 6, -100]], torch.long)
            target = F.pad(labels, (0, 1), value=-100)[:, 1:]
            with torch.no_grad():
                hidden = model.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
                logits = (hidden.half() @ model.lm_head.weight.half().t()).float()
                # Guarantee a correct prediction at the supervised pad boundary,
                # so attention-only hit counting cannot pass with zero accuracy.
                labels[0, 2] = logits[0, 1].argmax()
                target = F.pad(labels, (0, 1), value=-100)[:, 1:]
                row = F.cross_entropy(logits.reshape(-1, 73), target.reshape(-1),
                                      ignore_index=-100, reduction='none').view_as(labels)
                logp = logits.log_softmax(-1)
                entropy = -(logp.exp() * logp).sum(-1)
                hits = ((logits.argmax(-1) == target) & target.ne(-100)).sum()
            original_forward = mq.Qwen2ForCausalLM.forward
            original_compute = trl.SFTTrainer.compute_loss
            original_eligible = HEAD._eligible
            original_state = HEAD._STATE.copy()
            trainer = types.SimpleNamespace(
                model=model, args=types.SimpleNamespace(use_liger_kernel=False,
                    average_tokens_across_devices=False, n_gpu=0, past_index=-1),
                accelerator=types.SimpleNamespace(gather_for_metrics=lambda x: x,
                    parallelism_config=None, num_processes=1),
                label_smoother=None, compute_loss_func=None, model_accepts_loss_kwargs=True,
                _metrics={'train': defaultdict(list)}, _total_train_tokens=0)
            parameters = [p for p in model.parameters() if p.requires_grad]
            self.assertGreater(len(parameters), 0)
            ref_hidden = model.model(input_ids=ids, attention_mask=am,
                                     use_cache=False).last_hidden_state
            ref_logits = (ref_hidden.half() @ model.lm_head.weight.half().t()).float()
            ref_loss = F.cross_entropy(ref_logits.reshape(-1, 73), target.reshape(-1),
                                       ignore_index=-100, reduction='sum') / 9.
            ref_gradients = torch.autograd.grad(ref_loss * 65536., parameters, allow_unused=True)
            try:
                HEAD._install()  # Exercise the actual install and installed closures.
                if ARGS.device == 'cpu':
                    self.assertFalse(original_eligible(trainer, {'labels': labels, 'attention_mask': am}))
                    # CPU is excluded by production design. Bypass ONLY this gate
                    # to exercise the exact installed trainer/forward code path.
                    HEAD._eligible = lambda self, inputs: True
                loss, out = trl.SFTTrainer.compute_loss(
                    trainer, model, {'input_ids': ids, 'attention_mask': am, 'labels': labels},
                    return_outputs=True, num_items_in_batch=tensor(9.))
                step = getattr(trainer, HEAD.STEP_ATTR)
                self.assertTrue(loss.requires_grad)
                self.assertIsNone(out.logits)
                self.assert_close(loss, row.sum() / 9.)
                self.assert_close(step.token_loss, row)
                self.assertGreater(step.token_loss[0, 1].item(), 0.)
                self.assertGreater(step.token_loss[1, 0].item(), 0.)
                self.assert_close(step.token_entropy, entropy * am.bool())
                self.assertEqual(step.token_entropy[0, 1].item(), 0.)
                self.assertEqual(step.labelled.item(), target.ne(-100).sum().item())
                self.assertEqual(step.correct.item(), hits.item())
                metrics = trainer._metrics['train']
                self.assertAlmostEqual(metrics['entropy'][0], (entropy * am).sum().item()/am.sum().item(), places=4)
                self.assertEqual(metrics['num_tokens'], [am.sum().item()])
                self.assertAlmostEqual(metrics['mean_token_accuracy'][0], hits.item()/target.ne(-100).sum().item(), places=6)
                self.assertFalse(HEAD._STATE['armed'])
                self.assertIsNone(HEAD._STATE['step'])
                from rewind_reference import RewindState, row_losses_from_token_losses
                row_means, row_counts = row_losses_from_token_losses(step.token_loss, labels)
                self.assertEqual(row_counts, [3, 3])
                self.assert_close(tensor(row_means), row.sum(1) / tensor(row_counts))
                records = []
                rewind = RewindState(types.SimpleNamespace(record_batch=lambda **kw: records.append(kw)))
                for row_id in (701, 702, 703, 704):
                    rewind.on_index(row_id)
                rewind.on_micro_batch(step=8, row_loss=row_means, row_tokens=row_counts)
                (loss * 65536.).backward()
                for parameter, reference in zip(parameters, ref_gradients):
                    if reference is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())
                        self.assert_close(parameter.grad, reference, rtol=.02,
                                          atol=max(2e-6, reference.abs().max().item() * .002))
                self.assertTrue(any(p.grad is not None and p.grad.abs().max().item() > 0 for p in parameters))
                self.assertIsNone(model.lm_head.weight.grad)
                after_means, after_counts = row_losses_from_token_losses(step.token_loss, labels)
                self.assertEqual(after_means, row_means)
                self.assertEqual(after_counts, row_counts)
                rewind.on_micro_batch(step=8, row_loss=after_means, row_tokens=after_counts)
                self.assertEqual([r['rows'] for r in records], [[701, 702], [703, 704]])
                self.assertEqual([r['row_tokens'] for r in records], [[3, 3], [3, 3]])
                self.assertEqual([r['micro'] for r in records], [0, 1])
                self.assertEqual(rewind.dropped, 0)
                self.assertFalse(rewind.failed)
            finally:
                mq.Qwen2ForCausalLM.forward = original_forward
                trl.SFTTrainer.compute_loss = original_compute
                HEAD._eligible = original_eligible
                HEAD._STATE.clear()
                HEAD._STATE.update(original_state)

    @unittest.skipUnless(ARGS.device == 'cuda', 'CUDA-only: real GradScaler evidence unavailable on CPU')
    def test_cuda_real_gradscaler_fp32_parameter_fp32_hidden(self):
        parameter = torch.nn.Parameter(tensor([[1., 0.]]))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler = torch.amp.GradScaler('cuda', init_scale=65536., growth_interval=1000)
        weight = tensor([[0., 0.], [-20., 1.]])
        target = tensor([0], torch.long)
        # A real fp32 parameter produces a nonleaf fp32 hidden tensor; no fp16
        # leaf optimizer gradients, which GradScaler correctly refuses to unscale.
        hidden = parameter * 1.
        hidden.retain_grad()
        self.assertEqual(hidden.dtype, torch.float32)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, *_ = HEAD.fused_head_loss(hidden, weight, target, 1, chunk=1)
        reference = dense_oracle(hidden, weight, target, 1, scaler.get_scale())[-1]
        scaler.scale(loss).backward()
        self.assertEqual(parameter.grad.dtype, torch.float32)
        self.assertGreater(parameter.grad[0, 1].item(), 1e-4)
        self.assert_close(hidden.grad, reference, atol=2e-7, rtol=2e-3)
        scaler.unscale_(optimizer)
        self.assert_close(parameter.grad, reference / scaler.get_scale(), atol=1e-11, rtol=2e-3)
        before = parameter.detach().clone()
        expected = before - .1 * parameter.grad
        scaler.step(optimizer)
        scaler.update()
        self.assert_close(parameter, expected, atol=1e-12, rtol=1e-6)
        self.assertLess(parameter[0, 1].item(), 0.)
        self.assertEqual(scaler.get_scale(), 65536.)

    @unittest.skipUnless(ARGS.device == 'cuda', 'CUDA-only: full training vocabulary')
    def test_cuda_training_vocabulary_tail_and_multichunk(self):
        torch.manual_seed(404)
        h = torch.randn(5, 32, device=DEVICE, dtype=torch.float32).requires_grad_()
        weight = torch.randn(151936, 32, device=DEVICE) * .1
        target = tensor([0, 151935, -100, 2048, 151934], torch.long)
        loss, row, entropy, hit = HEAD.fused_head_loss(h, weight, target, 6, chunk=2)
        (loss * 65536.).backward()
        expected = dense_oracle(h, weight, target, 6, 65536.)
        self.assert_close(loss, expected[0])
        self.assert_close(row, expected[1])
        self.assert_close(entropy, expected[2])
        self.assertTrue(torch.equal(hit.bool(), expected[3]))
        self.assertTrue(torch.isfinite(h.grad).all().item())
        self.assertEqual(torch.count_nonzero(h.grad[2]).item(), 0)
        self.assert_close(h.grad, expected[4],
                          atol=max(2e-6, expected[4].abs().max().item() * .002), rtol=.02)


if __name__ == '__main__':
    print(f'EVIDENCE MODE: {ARGS.device}; candidate={ARGS.head_loss.resolve()}; '
          f'TRITON_INTERPRET={os.environ["TRITON_INTERPRET"]}; BLOCK={HEAD.BLOCK}', flush=True)
    unittest.main(argv=[sys.argv[0]] + UNITTEST_ARGS, verbosity=2)
