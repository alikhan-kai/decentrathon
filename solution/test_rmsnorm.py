"""Independent eager comparisons; CUDA tests require an actual CUDA runtime."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from solution import rmsnorm


def reference(x, weight, eps):
    # The graph, including both dtype transitions, is the Qwen2 contract.
    dtype = x.dtype
    fp32 = x.to(torch.float32)
    mean_square = fp32.pow(2).mean(-1, keepdim=True)
    scaled = fp32 * torch.rsqrt(mean_square + eps)
    return weight * scaled.to(dtype)


DTYPES = [
    (torch.float32, torch.float32),
    (torch.float16, torch.float32),
    (torch.float16, torch.float16),
    (torch.float32, torch.float16),
]
SHAPES = [(1536,), (2, 3, 1536), (3, 37), (2, 1, 257), (2, 1)]
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="requires CUDA and Triton; CPU execution is not GPU validation",
)


def compare_reference(shape, dtypes, eps, device, *, exact=False, scale=1.0):
    torch.manual_seed(23)
    input_dtype, weight_dtype = dtypes
    x = torch.randn(shape, device=device, dtype=input_dtype).requires_grad_()
    other = x.detach().clone().requires_grad_()
    weight = torch.randn(shape[-1], device=device, dtype=weight_dtype)
    output = rmsnorm.rms_norm(x, weight, eps)
    expected = reference(other, weight, eps)
    upstream = torch.randn_like(expected) * scale
    dx, = torch.autograd.grad(output, x, upstream)
    expected_dx, = torch.autograd.grad(expected, other, upstream)
    assert output.dtype == expected.dtype == torch.promote_types(input_dtype, weight_dtype)
    assert dx.dtype == input_dtype
    if device == "cuda":
        assert isinstance(output.grad_fn, torch.autograd.function.BackwardCFunction)
        assert type(output.grad_fn).__name__ == "_FrozenRMSNormBackward"
    tolerance = {"rtol": 0, "atol": 0} if exact else (
        {"rtol": 2e-3, "atol": 2e-3 * scale}
        if input_dtype == torch.float16
        else {"rtol": 2e-5, "atol": 2e-6 * scale}
    )
    torch.testing.assert_close(output, expected, **tolerance)
    torch.testing.assert_close(dx, expected_dx, **tolerance)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_cpu_forward_and_input_gradient(shape, dtypes, eps):
    compare_reference(shape, dtypes, eps, "cpu", exact=True)


@CUDA
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_cuda_forward_and_input_gradient(shape, dtypes, eps):
    compare_reference(shape, dtypes, eps, "cuda")


@CUDA
@pytest.mark.parametrize("scale", [2.0**-16, 4096.0])
def test_cuda_backward_upstream_scale(scale):
    compare_reference((2, 1536), (torch.float16, torch.float32), 1e-6, "cuda", scale=scale)


@CUDA
def test_cuda_rounds_at_both_half_precision_boundaries():
    x = torch.ones(2, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    weight = torch.linspace(0.71, 1.31, 1536, device="cuda")
    upstream = torch.linspace(-2.1, 1.9, x.numel(), device="cuda").reshape_as(x)
    output = rmsnorm.rms_norm(x, weight, 1e-6)
    expected = reference(other, weight, 1e-6)
    # Normalized fp16 values round to exactly one; output must retain fp32 W.
    torch.testing.assert_close(output, weight.expand_as(x), rtol=0, atol=0)
    actual_dx, = torch.autograd.grad(output, x, upstream)
    expected_dx, = torch.autograd.grad(expected, other, upstream)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=2e-6)

    # An apparently equivalent fused derivative drops the internal half cast.
    hidden = other.detach().float()
    inv = (hidden.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    wrong_grad = upstream * weight
    wrong_dx = (
        wrong_grad * inv
        - hidden * inv.pow(3) * (wrong_grad * hidden).mean(-1, keepdim=True)
    ).half()
    assert (expected_dx - wrong_dx).abs().max().item() > 1e-4


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("case", ["trainable", "strided_x", "strided_weight", "bf16", "double", "wide", "empty", "broadcast"])
def test_eager_fallback_preserves_both_gradients(monkeypatch, device, case):
    torch.manual_seed(41)
    x = torch.randn(2, 37, device=device)
    weight = torch.randn(37, device=device)
    if case == "strided_x":
        x = torch.randn(37, 2, device=device).t()
    elif case == "strided_weight":
        weight = torch.randn(74, device=device)[::2]
    elif case == "bf16":
        x = x.bfloat16()
    elif case == "double":
        x, weight = x.double(), weight.double()
    elif case == "wide":
        x, weight = torch.randn(1, 8193, device=device), torch.randn(8193, device=device)
    elif case == "empty":
        x = torch.empty(0, 37, device=device)
    elif case == "broadcast":
        weight = weight.unsqueeze(0)
    x.requires_grad_()
    weight.requires_grad_(case == "trainable")
    other = x.detach().clone().requires_grad_()
    other_weight = weight.detach().clone().requires_grad_(weight.requires_grad)

    def no_kernel_load():
        pytest.fail("unsupported input attempted to load Triton")

    monkeypatch.setattr(rmsnorm, "_triton_kernels", no_kernel_load)
    actual = rmsnorm.rms_norm(x, weight, 1e-6)
    expected = reference(other, other_weight, 1e-6)
    upstream = torch.randn_like(expected)
    inputs = (x, weight) if weight.requires_grad else (x,)
    refs = (other, other_weight) if weight.requires_grad else (other,)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, refs, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_import_and_cpu_execution_never_import_triton():
    script = r'''
import importlib.abc
import importlib.util
import sys
import torch
class RejectTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise AssertionError("CPU RMSNorm imported Triton")
sys.meta_path.insert(0, RejectTriton())
spec = importlib.util.spec_from_file_location("isolated_rmsnorm", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
x = torch.ones(2, 3, requires_grad=True)
module.rms_norm(x, torch.ones(3), 1e-6).sum().backward()
assert x.grad is not None
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(rmsnorm.__file__).resolve())],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_patch_is_idempotent_and_retains_original_fallback(monkeypatch):
    calls = []

    class FakeQwen2RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(37), requires_grad=False)
            self.variance_epsilon = 1e-6

        def forward(self, hidden_states):
            calls.append(hidden_states)
            return reference(hidden_states, self.weight, self.variance_epsilon)

    module_name = "transformers.models.qwen2.modeling_qwen2"
    fake_module = types.ModuleType(module_name)
    fake_module.Qwen2RMSNorm = FakeQwen2RMSNorm
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    norm = FakeQwen2RMSNorm()
    original = FakeQwen2RMSNorm.forward
    parameter = norm.weight
    assert rmsnorm.patch_rmsnorm() is True
    patched = FakeQwen2RMSNorm.forward
    assert patched._soup_rmsnorm_original_forward is original
    assert rmsnorm.patch_rmsnorm() is True
    assert FakeQwen2RMSNorm.forward is patched
    x = torch.randn(2, 37, requires_grad=True)
    torch.testing.assert_close(norm(x), reference(x, parameter, 1e-6), rtol=0, atol=0)
    assert len(calls) == 1 and calls[0] is x
    assert norm.weight is parameter
    assert norm.variance_epsilon == 1e-6
    assert not norm.weight.requires_grad


def test_patch_missing_optional_class(monkeypatch):
    name = "transformers.models.qwen2.modeling_qwen2"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    assert rmsnorm.patch_rmsnorm() is False


@CUDA
def test_cuda_noncontiguous_upstream_and_second_derivative():
    torch.manual_seed(7)
    x = torch.randn(2, 37, device="cuda", requires_grad=True)
    other = x.detach().clone().requires_grad_()
    weight = torch.randn(37, device="cuda")
    upstream = torch.randn(37, 2, device="cuda").t()
    actual = rmsnorm.rms_norm(x, weight, 1e-6)
    expected = reference(other, weight, 1e-6)
    first, = torch.autograd.grad(actual, x, upstream, create_graph=True)
    ref_first, = torch.autograd.grad(expected, other, upstream, create_graph=True)
    second, = torch.autograd.grad(first.square().sum(), x)
    ref_second, = torch.autograd.grad(ref_first.square().sum(), other)
    torch.testing.assert_close(first, ref_first, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(second, ref_second, rtol=1e-5, atol=1e-6)
    # Exercise the kernel as well as the higher-order eager backward.
    first_kernel, = torch.autograd.grad(rmsnorm.rms_norm(x, weight, 1e-6), x, upstream)
    torch.testing.assert_close(first_kernel, ref_first, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_qwen_decoder_output_and_gradient_parity(monkeypatch, dtype, device):
    pytest.importorskip("transformers")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2RMSNorm

    # Keep the real decoder stack (residuals, attention and MLP) small and offline.
    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=101, hidden_size=64, intermediate_size=112,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen2Model(config).to(device=device, dtype=dtype).eval().requires_grad_(False)
    for module in model.modules():
        if isinstance(module, Qwen2RMSNorm):
            module.float()
    forward = Qwen2RMSNorm.forward
    original = getattr(forward, "_soup_rmsnorm_original_forward", forward)
    monkeypatch.setattr(Qwen2RMSNorm, "forward", original)
    x = torch.randn(2, 7, 64, device=device, dtype=dtype, requires_grad=True)
    other = x.detach().clone().requires_grad_()

    def run(inputs):
        with torch.autocast(device, dtype=torch.float16, enabled=dtype == torch.float16):
            return model(inputs_embeds=inputs, use_cache=False).last_hidden_state

    expected = run(other)
    upstream = torch.randn_like(expected)
    expected_dx, = torch.autograd.grad(expected, other, upstream)
    assert rmsnorm.patch_rmsnorm() is True
    actual = run(x)
    if device == "cuda":
        assert type(actual.grad_fn).__name__ == "_FrozenRMSNormBackward"
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    assert actual.dtype == expected.dtype == torch.float32
    tolerance = {"rtol": 0, "atol": 0} if device == "cpu" else (
        {"rtol": 3e-3, "atol": 3e-3} if dtype == torch.float16
        else {"rtol": 3e-5, "atol": 3e-6}
    )
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(actual_dx, expected_dx, **tolerance)
