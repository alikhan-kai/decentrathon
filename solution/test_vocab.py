"""Numerical API tests; CUDA cases execute the kernels when a GPU is present."""

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from solution.vocab import (
    vocab_backward,
    vocab_backward_reference,
    vocab_stats,
    vocab_stats_reference,
)


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")),
]


def _dense_reference(logits, targets, ignore_index=-100, scale=1.0):
    """Independent dense float64/autograd oracle using the rounded inputs."""
    values = logits.detach().double().requires_grad_(True)
    labels = targets.long()
    loss = F.cross_entropy(values, labels, ignore_index=ignore_index, reduction="none")
    log_probability = F.log_softmax(values, dim=-1)
    entropy = -(log_probability.exp() * log_probability).sum(dim=-1)
    prediction = values.argmax(dim=-1)
    lse = torch.logsumexp(values, dim=-1)
    (gradient,) = torch.autograd.grad(loss.sum() * scale, values)
    return loss.detach(), entropy.detach(), prediction, lse.detach(), gradient.detach()


def _assert_stats(actual, expected, *, atol=5e-6):
    loss, entropy, prediction, lse = actual
    assert loss.dtype == entropy.dtype == lse.dtype == torch.float32
    assert prediction.dtype == torch.int64
    for output in actual:
        assert output.shape == expected[0].shape
        assert not output.requires_grad
    torch.testing.assert_close(loss, expected[0].float(), atol=atol, rtol=3e-6)
    torch.testing.assert_close(entropy, expected[1].float(), atol=atol, rtol=3e-6)
    torch.testing.assert_close(prediction, expected[2], atol=0, rtol=0)
    # Allow two fp32 ULPs, including at large offsets, rather than a broad
    # relative tolerance that would obscure a missing normalization term.
    reference_lse = expected[3].float()
    ulp = (torch.nextafter(reference_lse, torch.full_like(reference_lse, math.inf)) - reference_lse).abs()
    assert bool(((lse - reference_lse).abs() <= torch.maximum(ulp * 2, torch.full_like(ulp, atol))).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,vocab", [(1, 1), (5, 17), (7, 2049), (3, 4097), (2, 151936)])
def test_stats_and_scaled_gradient_match_dense_autograd(device, dtype, rows, vocab):
    generator = torch.Generator().manual_seed(84 + vocab)
    logits = (torch.randn(rows, vocab, generator=generator) * 3).to(device=device, dtype=dtype)
    targets = (torch.arange(rows) * 71 % vocab).to(device)
    if rows > 1:
        targets[1] = -100
    scale = torch.tensor(0.375, device=device)
    actual = vocab_stats(logits, targets)
    expected = _dense_reference(logits, targets, scale=0.375)
    _assert_stats(actual, expected)
    gradient = vocab_backward(logits, targets, actual[3], scale)
    assert gradient.shape == logits.shape and gradient.dtype == logits.dtype
    assert not gradient.requires_grad
    # fp16 rounds once at the final store; fp32 is compared to float64 AD.
    tolerance = dict(atol=6e-8, rtol=0.002) if dtype == torch.float16 else dict(atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(gradient, expected[4].to(dtype), **tolerance)
    if rows > 1:
        assert torch.count_nonzero(gradient[1]) == 0
        assert actual[0][1] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [-1000000.0, -10000.0, 10000.0, 1000000.0])
def test_large_common_offsets_preserve_entropy_loss_and_gradient(device, offset):
    generator = torch.Generator().manual_seed(411)
    logits = ((torch.randn(4, 4099, generator=generator) * 2) + offset).to(device)
    targets = torch.tensor([0, 2048, 4098, -100], device=device)
    outputs = vocab_stats(logits, targets)
    expected = _dense_reference(logits, targets, scale=0.75)
    _assert_stats(outputs, expected)
    gradient = vocab_backward(logits, targets, outputs[3], torch.tensor(0.75, device=device))
    torch.testing.assert_close(gradient, expected[4].float(), atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_lse_rounding_requires_backward_renormalization(device):
    # All logits are equal: exact p=1/V. At offset 1e6, fp32 LSE has a
    # 0.0625 ULP, making exp(x-LSE) over 0.5 percent wrong. This is the reason
    # backward deliberately pays for another full logit read and partial sums.
    vocab = 4099
    logits = torch.full((1, vocab), 1_000_000.0, device=device)
    targets = torch.tensor([3], device=device)
    outputs = vocab_stats(logits, targets)
    naive_probability = torch.exp(logits[0, 0] - outputs[3][0])
    assert abs(naive_probability.item() * vocab - 1.0) > 0.005
    gradient = vocab_backward(logits, targets, outputs[3], torch.tensor(1.0, device=device))
    torch.testing.assert_close(gradient[0, 0], torch.tensor(1 / vocab, device=device), rtol=2e-6, atol=1e-9)
    assert abs(gradient.float().sum().item()) < 2e-6
    torch.testing.assert_close(outputs[1], torch.tensor([math.log(vocab)], device=device), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [-60000.0, 60000.0])
def test_fp16_large_offsets_use_the_given_rounded_logits(device, offset):
    logits = (torch.linspace(-64, 64, 4101) + offset).repeat(2, 1).to(device=device, dtype=torch.float16)
    targets = torch.tensor([0, 4100], device=device)
    expected = _dense_reference(logits, targets)
    actual = vocab_stats(logits, targets)
    _assert_stats(actual, expected)
    gradient = vocab_backward(logits, targets, actual[3], torch.tensor(1.0, device=device))
    torch.testing.assert_close(gradient, expected[4].half(), atol=6e-8, rtol=0.002)


@pytest.mark.parametrize("device", DEVICES)
def test_lowest_index_ties_across_tiles_and_custom_ignore(device):
    logits = torch.full((4, 4101), -12.0, device=device)
    logits[0, [2048, 4096]] = 9.0
    logits[1, [17, 2049, 4100]] = 7.0
    logits[2] = 0
    logits[3, [2047, 2048]] = 5.0
    targets = torch.tensor([2048, -7, 4100, -7], device=device, dtype=torch.int32)
    outputs = vocab_stats(logits, targets, ignore_index=-7)
    _assert_stats(outputs, _dense_reference(logits, targets, ignore_index=-7))
    torch.testing.assert_close(outputs[2], torch.tensor([2048, 17, 0, 2047], device=device))
    assert torch.all(outputs[1] > 0)
    gradient = vocab_backward(logits, targets, outputs[3], torch.tensor(1.0, device=device), ignore_index=-7)
    assert torch.count_nonzero(gradient[[1, 3]]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_fp16_upstream_scaling_precedes_narrowing(device):
    vocab = 151936
    denominator = 1024
    logits = torch.zeros((2, vocab), device=device, dtype=torch.float16)
    targets = torch.tensor([7, -100], device=device)
    outputs = vocab_stats(logits, targets)
    # Represents GradScaler's 65536 upstream gradient / the batch denominator.
    scale = torch.tensor(65536.0, device=device) / denominator
    actual = vocab_backward(logits, targets, outputs[3], scale)
    expected = _dense_reference(logits, targets, scale=65536.0 / denominator)[4].half()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual[0, 0] > 0
    assert torch.tensor(1 / (vocab * denominator)).half() == 0
    assert torch.count_nonzero(actual[1]) == 0
    # Also exercise the actual 65536 scalar, which overflows if narrowed first.
    direct = vocab_backward(logits, targets, outputs[3], torch.tensor(65536.0, device=device))
    assert torch.isfinite(direct[0, 0]) and direct[0, 0] > 0
    torch.testing.assert_close(direct[0, 0], torch.tensor(65536.0 / vocab, device=device).half(), atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_inputs_and_chunked_rows(device):
    generator = torch.Generator().manual_seed(301)
    base = torch.randn(13, 2 * 2051, generator=generator).to(device)
    logits = base[:, ::2]
    labels = torch.arange(26, device=device, dtype=torch.int64)
    labels[::6] = -100
    targets = labels[::2]
    assert not logits.is_contiguous() and not targets.is_contiguous()
    expected = _dense_reference(logits, targets, scale=-1.25)
    pieces = [vocab_stats(logits[start:end], targets[start:end]) for start, end in [(0, 1), (1, 6), (6, 13)]]
    outputs = tuple(torch.cat([piece[index] for piece in pieces]) for index in range(4))
    _assert_stats(outputs, expected)
    lse_storage = torch.empty(26, device=device)
    lse_storage[::2] = outputs[3]
    gradient = vocab_backward(logits, targets, lse_storage[::2], torch.tensor(-1.25))
    torch.testing.assert_close(gradient, expected[4].float(), atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_all_ignored_still_reports_statistics(device):
    logits = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 8.0, 8.0]], device=device, requires_grad=True)
    targets = torch.full((2,), -100, device=device, dtype=torch.int64)
    outputs = vocab_stats(logits, targets)
    _assert_stats(outputs, _dense_reference(logits, targets))
    assert torch.count_nonzero(outputs[0]) == 0
    assert torch.all(outputs[1] > 0)
    gradient = vocab_backward(logits, targets, outputs[3], torch.tensor(65536.0, device=device))
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_zero_rows_and_zero_scale(device):
    logits = torch.empty((0, 2051), device=device)
    targets = torch.empty((0,), dtype=torch.int64, device=device)
    outputs = vocab_stats(logits, targets)
    assert all(item.shape == (0,) for item in outputs)
    assert vocab_backward(logits, targets, outputs[3], torch.tensor(1.0, device=device)).shape == logits.shape
    logits = torch.tensor([[2.0, -3.0]], device=device)
    targets = torch.tensor([1], device=device)
    outputs = vocab_stats(logits, targets)
    assert torch.count_nonzero(vocab_backward(logits, targets, outputs[3], torch.tensor(0.0, device=device))) == 0


@pytest.mark.parametrize("label", [-101, -1, 4, 2**40])
def test_cpu_invalid_labels_fail_before_indexing(label):
    logits = torch.zeros(1, 4)
    targets = torch.tensor([label])
    with pytest.raises(IndexError, match="target"):
        vocab_stats(logits, targets)
    with pytest.raises(IndexError, match="target"):
        vocab_backward(logits, targets, torch.zeros(1), torch.tensor(1.0))


def test_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="shape"):
        vocab_stats(torch.empty(2, 0), torch.tensor([0, 0]))
    with pytest.raises(TypeError, match="fp16 or fp32"):
        vocab_stats(torch.zeros(2, 4, dtype=torch.float64), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="targets"):
        vocab_stats(torch.zeros(2, 4), torch.tensor([0]))
    with pytest.raises(TypeError, match="int32 or int64"):
        vocab_stats(torch.zeros(2, 4), torch.zeros(2))
    with pytest.raises(ValueError, match="lse"):
        vocab_backward(torch.zeros(2, 4), torch.tensor([0, 0]), torch.zeros(2, dtype=torch.float64), torch.tensor(1.0))
    with pytest.raises(ValueError, match="scale"):
        vocab_backward(torch.zeros(2, 4), torch.tensor([0, 0]), torch.zeros(2), torch.ones(2))


def test_explicit_cpu_reference_entrypoints():
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float16)
    targets = torch.tensor([1])
    outputs = vocab_stats_reference(logits, targets)
    _assert_stats(outputs, _dense_reference(logits, targets))
    gradient = vocab_backward_reference(logits, targets, outputs[3], torch.tensor(2.0))
    torch.testing.assert_close(gradient, _dense_reference(logits, targets, scale=2.0)[4].half(), atol=0, rtol=0)


def test_module_imports_lazily_and_cpu_works_without_triton():
    # Isolate import state, block Triton even on a machine where it is installed,
    # and verify neither torch nor Triton is required just to import the module.
    module_path = Path(__file__).with_name("vocab.py")
    script = """
import importlib.abc
import importlib.util
import sys
class NoTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'triton' or fullname.startswith('triton.'):
            raise ImportError('Triton intentionally unavailable')
sys.meta_path.insert(0, NoTriton())
spec = importlib.util.spec_from_file_location('isolated_vocab', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
import torch
x = torch.tensor([[0., 0., 0.]])
y = torch.tensor([1])
outputs = module.vocab_stats(x, y)
torch.testing.assert_close(outputs[0], torch.tensor([3.]).log())
gradient = module.vocab_backward(x, y, outputs[3], torch.tensor(2.))
torch.testing.assert_close(gradient, torch.tensor([[2./3, -4./3, 2./3]]))
"""
    result = subprocess.run([sys.executable, "-c", script, str(module_path)], text=True, capture_output=True,
                            timeout=60, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("logit_type,target_type", [("fp16", "i64"), ("fp32", "i32")])
@pytest.mark.parametrize("vocab", [4099, 151936])
def test_triton_kernels_compile_offline_for_sm75(logit_type, target_type, vocab):
    """Code generation only: explicitly does not require/claim GPU execution."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from solution import vocab as implementation

    splits = triton.cdiv(vocab, 2048)
    reduce = triton.next_power_of_2(splits)
    cases = [
        (implementation.ember_vocab_partials,
         dict(Logits=f"*{logit_type}", Maxima="*fp32", Sums="*fp32", Moments="*fp32", Winners="*i32"),
         dict(V=vocab, SPLITS=splits, BLOCK=2048)),
        (implementation.ember_vocab_merge,
         dict(Logits=f"*{logit_type}", Targets=f"*{target_type}", Maxima="*fp32", Sums="*fp32",
              Moments="*fp32", Winners="*i32", Loss="*fp32", Entropy="*fp32", Predictions="*i64", Lse="*fp32"),
         dict(V=vocab, SPLITS=splits, IGNORE=-100, REDUCE=reduce)),
        (implementation.ember_vocab_backward_mass,
         dict(Logits=f"*{logit_type}", Lse="*fp32", Masses="*fp32"),
         dict(V=vocab, SPLITS=splits, BLOCK=2048)),
        (implementation.ember_vocab_backward_write,
         dict(Logits=f"*{logit_type}", Targets=f"*{target_type}", Lse="*fp32", Scale="*fp32",
              Masses="*fp32", Gradient=f"*{logit_type}"),
         dict(V=vocab, SPLITS=splits, IGNORE=-100, BLOCK=2048, REDUCE=reduce)),
    ]
    for function, signature, constants in cases:
        compiled = triton.compile(ASTSource(function, signature, constexprs=constants),
                                  target=GPUTarget("cuda", 75, 32),
                                  options={"num_warps": 4, "debug": True})
        assert compiled.asm["cubin"]
        assert ".target sm_75" in compiled.asm["ptx"]
