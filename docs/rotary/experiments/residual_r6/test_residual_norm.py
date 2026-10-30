"""Focused residual/RMSNorm checks; interpretation/SM75 compilation are not GPU evidence.

Run normally (CPU dispatch + offline SM75), then with TRITON_INTERPRET=1.
On NVIDIA CUDA, run -k cuda separately for production-kernel correctness.
Separate processes keep Triton's interpreter language patches out of its compiler.
"""

import contextlib
import copy
import functools
import inspect
import types

import pytest
import torch
import triton

from candidate import residual_norm as rn
from baseline import rmsnorm as team
import sys
sys.modules["solution.rmsnorm"] = team  # Production team-patch identity guard.


def qwen_reference(residual, attention, weight, eps=1e-6):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    norm = Qwen2RMSNorm(weight.numel(), eps=eps)
    norm.weight = weight
    summed = residual + attention
    return summed, norm(summed)


@pytest.fixture
def interpreted(monkeypatch):
    if not triton.knobs.runtime.interpret:
        pytest.skip("run separately with TRITON_INTERPRET=1; no GPU correctness claim")
    # Only the device gate is overridden: production kernels and all other
    # guards run unchanged. Production behavior never depends on this fixture.
    monkeypatch.setattr(rn, "_cuda_supported", lambda x: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())


@pytest.fixture
def patched():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    original = Qwen2DecoderLayer.forward
    assert rn.patch()
    assert rn.patch()
    yield
    rn.unpatch()
    assert Qwen2DecoderLayer.forward is original


def make_inputs(shape, dtype=torch.float16, residual_grad=True, attention_grad=True, device="cpu"):
    torch.manual_seed(930)
    r = torch.randn(shape, device=device).requires_grad_(residual_grad)
    a = (torch.randn(shape, device=device) / 4).to(dtype).requires_grad_(attention_grad)
    w = torch.nn.Parameter(torch.rand(shape[-1], device=device) + .5, requires_grad=False)
    return r, a, w


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 17), (1, 3, 1001), (2, 3, 1536)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1, 65536])
def test_interpreted_matches_plain_qwen_and_current_team(interpreted, shape, dtype, scale):
    r, a, w = make_inputs(shape, dtype)
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    assert type(actual[0].grad_fn).__name__ == "_ResidualNormBackward"
    # Noncontiguous and broadcast upstream gradients exercise runtime strides.
    gs = torch.randn(*shape, 2)[..., 0] * (scale / 32)
    gy = torch.randn(1).expand(shape) * (scale / 32)
    gradients = torch.autograd.grad(actual, (r, a), (gs, gy))
    for use_team in (False, True):
        rr, aa = (x.detach().clone().requires_grad_() for x in (r, a))
        if use_team:
            summed = rr + aa
            expected = summed, team.rmsnorm(summed, w, 1e-6)
        else:
            expected = qwen_reference(rr, aa, w)
        expected_gradients = torch.autograd.grad(expected, (rr, aa), (gs, gy))
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
        for got, want in zip(gradients, expected_gradients):
            tol = 2e-3 if got.dtype == torch.float16 else 1e-4
            torch.testing.assert_close(got / scale, want / scale, atol=tol, rtol=tol)
    assert actual[0].dtype == actual[1].dtype == gradients[0].dtype == torch.float32
    assert gradients[1].dtype == dtype


@pytest.mark.parametrize("which", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("needs", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("hidden", [7, 1536])
def test_missing_outputs_and_frozen_first_residual(interpreted, which, needs, hidden):
    r, a, w = make_inputs((2, 3, hidden), residual_grad=needs[0], attention_grad=needs[1])
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    expected = qwen_reference(r, a, w)
    variables = tuple(x for x in (r, a) if x.requires_grad)
    upstream = tuple(torch.randn_like(actual[i]) for i in which)
    got = torch.autograd.grad(tuple(actual[i] for i in which), variables, upstream)
    want = torch.autograd.grad(tuple(expected[i] for i in which), variables, upstream)
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("hidden", [17, 1536])
def test_scaled_branches_cancel_before_only_half_cast(interpreted, hidden):
    # Each branch exceeds fp16's finite range; their SUM is exactly zero.
    r = torch.zeros(2, hidden, requires_grad=True)
    a = torch.zeros_like(r, dtype=torch.float16, requires_grad=True)
    w = torch.ones(hidden)
    summed, normalized = rn.fused_residual_norm(r, a, w, 1.)
    gradients = torch.autograd.grad((summed, normalized), (r, a),
                                    (torch.full_like(r, -65536), torch.full_like(r, 65536)))
    for gradient in gradients:
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_saved_state_and_higher_derivatives(interpreted):
    r, a, w = make_inputs((2, 17), torch.float32)
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    saved = actual[0].grad_fn.saved_tensors
    assert len(saved) == 3
    assert saved[0].data_ptr() == actual[0].data_ptr()
    assert all(x.data_ptr() not in (r.data_ptr(), a.data_ptr()) for x in saved)
    for result in (actual, qwen_reference(r, a, w)):
        first = torch.autograd.grad(sum(x.square().sum() for x in result), r,
                                    create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), r)[0]
        if result is actual:
            got = first, second
        else:
            for x, y in zip(got, (first, second)):
                torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("case", ["cpu", "empty", "strided", "weight_grad", "half_residual",
                                  "broadcast", "subclass"])
def test_eager_fallback_preserves_outputs_and_all_gradients(interpreted, monkeypatch, case):
    r, a, w = make_inputs((2, 3, 17))
    if case == "cpu":
        monkeypatch.setattr(rn, "_cuda_supported", lambda x: False)
    elif case == "empty":
        r, a, w = make_inputs((2, 0, 17))
    elif case == "strided":
        r, a = r.transpose(0, 1), a.transpose(0, 1)
    elif case == "weight_grad":
        w.requires_grad_()
    elif case == "half_residual":
        r = r.half()
    elif case == "broadcast":
        a = a[:1]
    elif case == "subclass":
        class TensorSubclass(torch.Tensor):
            pass
        r = r.as_subclass(TensorSubclass)
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    expected = qwen_reference(r, a, w)
    assert type(actual[0].grad_fn).__name__ != "_ResidualNormBackward"
    variables = (r, a, w) if w.requires_grad else (r, a)
    gradients = [torch.autograd.grad(sum(x.sum() for x in outputs), variables)
                 for outputs in (actual, expected)]
    for got, want in zip(actual + gradients[0], expected + gradients[1]):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def make_layer():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    cfg = Qwen2Config(hidden_size=16, intermediate_size=32, num_attention_heads=2,
                      num_key_value_heads=1, num_hidden_layers=1, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    layer = Qwen2DecoderLayer(cfg, 0).train()
    layer.input_layernorm.weight.requires_grad_(False)
    layer.post_attention_layernorm.weight.requires_grad_(False)
    return layer


def layer_arguments():
    return dict(position_embeddings=(torch.ones(1, 3, 8), torch.zeros(1, 3, 8)))


def test_patch_signature_cpu_and_idempotence(patched):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    assert inspect.signature(Qwen2DecoderLayer.forward) == inspect.signature(rn._original)
    layer = make_layer()
    x = torch.randn(1, 3, 16, requires_grad=True)
    got = layer(x, **layer_arguments())
    want = rn._original(layer, x, **layer_arguments())
    assert isinstance(got, torch.Tensor)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("attention_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1, 65536])
def test_tiny_qwen_model_and_decoder_attention_hooks(interpreted, patched, monkeypatch,
                                                    attention_dtype, scale):
    from transformers import Qwen2Config, Qwen2Model

    torch.manual_seed(41)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2,
                         attention_dropout=0., use_cache=False)
    config._attn_implementation = "eager"
    candidate = Qwen2Model(config).train()
    for module in candidate.modules():
        if "RMSNorm" in type(module).__name__:
            module.weight.requires_grad_(False)
    candidate.embed_tokens.weight.requires_grad_(False)  # first Q input frozen
    reference = copy.deepcopy(candidate)
    for layer in reference.layers:
        layer.forward = types.MethodType(rn._original, layer)
    events = []
    handles = []
    calls = []
    original_fused = rn._fused_residual_norm_validated
    def record(residual, attention, weight, eps):
        calls.append((residual.requires_grad, attention.dtype))
        return original_fused(residual, attention, weight, eps)
    monkeypatch.setattr(rn, "_fused_residual_norm_validated", record)
    def cast_attention(module, args, output):
        return output[0].to(attention_dtype), output[1]
    for layer in list(candidate.layers) + list(reference.layers):
        handles.append(layer.self_attn.register_forward_hook(cast_attention))
    for index, layer in enumerate(candidate.layers):
        handles.append(layer.register_forward_hook(lambda *args, i=index: events.append(("layer", i))))
        handles.append(layer.self_attn.register_forward_hook(
            lambda *args, i=index: events.append(("attention", i))))
    try:
        ids = torch.tensor([[1, 2, 3], [4, 2, 0]])
        actual = candidate(ids).last_hidden_state
        expected = reference(ids).last_hidden_state
        # Small coefficients keep parameter gradients finite at scale 65536.
        upstream = torch.randn_like(actual) * (scale / 128)
        actual.backward(upstream)
        expected.backward(upstream)
        assert calls == [(False, attention_dtype), (True, attention_dtype)]
        assert events == [("attention", 0), ("layer", 0), ("attention", 1), ("layer", 1)]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for (name, a), (_, b) in zip(candidate.named_parameters(), reference.named_parameters()):
            if a.grad is None or b.grad is None:
                assert a.grad is b.grad is None, name
            else:
                torch.testing.assert_close(a.grad / scale, b.grad / scale,
                                           atol=2e-4, rtol=2e-3, msg=name)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("case", ["forward_hook", "pre_hook", "backward_hook", "backward_pre_hook",
                                  "instance_forward", "accelerate", "global_forward", "global_pre",
                                  "global_backward", "global_backward_pre", "trainable", "subclass",
                                  "unknown_class_forward"])
def test_norm_observers_and_overrides_are_not_bypassed(interpreted, patched, monkeypatch, case):
    layer = make_layer()
    norm = layer.post_attention_layernorm
    events = []
    handles = []
    hook = lambda *args: events.append(case)
    if case in ("forward_hook", "pre_hook", "backward_hook", "backward_pre_hook"):
        register = {"forward_hook": "register_forward_hook", "pre_hook": "register_forward_pre_hook",
                    "backward_hook": "register_full_backward_hook",
                    "backward_pre_hook": "register_full_backward_pre_hook"}[case]
        handles.append(getattr(norm, register)(hook))
    elif case.startswith("global_"):
        register = {"global_forward": "register_module_forward_hook",
                    "global_pre": "register_module_forward_pre_hook",
                    "global_backward": "register_module_full_backward_hook",
                    "global_backward_pre": "register_module_full_backward_pre_hook"}[case]
        handles.append(getattr(torch.nn.modules.module, register)(hook))
    elif case == "instance_forward":
        original = norm.forward
        norm.forward = lambda x: original(x) * 2
    elif case == "accelerate":
        norm._hf_hook = object()
    elif case == "trainable":
        norm.weight.requires_grad_(True)
    elif case == "subclass":
        class CustomNorm(type(norm)):
            pass
        norm.__class__ = CustomNorm
    else:
        original = type(norm).forward
        monkeypatch.setattr(type(norm), "forward", lambda self, x: original(self, x) * 2)
    def forbidden(*args):
        pytest.fail("observable/unknown norm must run through its module call")
    monkeypatch.setattr(rn, "_fused_residual_norm_validated", forbidden)
    try:
        x = torch.randn(1, 3, 16, requires_grad=True)
        actual = layer(x, **layer_arguments())
        expected = rn._original(layer, x, **layer_arguments())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        if "hook" in case or case.startswith("global_"):
            assert events
        if case == "trainable":
            assert norm.weight.grad is not None
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("reentrant", [True, False])
def test_checkpoint_falls_back(interpreted, patched, monkeypatch, reentrant):
    from functools import partial
    from torch.utils.checkpoint import checkpoint

    layer = make_layer()
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=reentrant)
    monkeypatch.setattr(rn, "_fused_residual_norm_validated", lambda *args: pytest.fail("checkpoint fast path"))
    x = torch.randn(1, 3, 16, requires_grad=True)
    actual = layer(x, **layer_arguments())
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_unexpected_attention_dtype_calls_attention_once(interpreted, patched, monkeypatch):
    layer = make_layer()
    calls = []
    original = layer.self_attn.forward
    def changed(*args, **kwargs):
        calls.append(1)
        x, weights = original(*args, **kwargs)
        return x.double(), weights
    layer.self_attn.forward = changed
    # Observe the real sum/norm, while making double input legal for this tiny MLP.
    layer.mlp.double()
    monkeypatch.setattr(rn, "_fused_residual_norm_validated", lambda *args: pytest.fail("dtype fast path"))
    out = layer(torch.randn(1, 3, 16), **layer_arguments())
    assert len(calls) == 1
    assert out.dtype == torch.float64


def test_team_patch_is_known_without_replacing_other_methods(interpreted, patched):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    original = Qwen2RMSNorm.forward
    team.patch()
    try:
        layer = make_layer()
        assert rn._safe_norm(layer.post_attention_layernorm)
        out = layer(torch.randn(1, 3, 16), **layer_arguments())
        out.sum().backward()
        assert layer.self_attn.q_proj.weight.grad is not None
    finally:
        team.unpatch()
    assert Qwen2RMSNorm.forward is original


@pytest.mark.parametrize("mode", ["eval", "no_grad", "inference", "cache", "past", "decoder_subclass"])
def test_decoder_guard_modes(interpreted, patched, monkeypatch, mode):
    layer = make_layer()
    r = torch.randn(1, 3, 16)
    if mode == "eval":
        layer.eval()
    if mode == "decoder_subclass":
        class CustomDecoder(type(layer)):
            pass
        layer.__class__ = CustomDecoder
    context = {"no_grad": torch.no_grad, "inference": torch.inference_mode}.get(mode,
                contextlib.nullcontext)
    with context():
        assert not rn._can_patch_call(layer, r, object() if mode == "past" else None,
                                      mode == "cache", {})
        if mode not in ("cache", "past"):
            monkeypatch.setattr(rn, "_fused_residual_norm_validated", lambda *args: pytest.fail("guard bypass"))
            actual = layer(r, **layer_arguments())
            expected = rn._original(layer, r, **layer_arguments())
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fake_forward_ad_and_tracing_use_eager(interpreted, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    r, a, w = make_inputs((2, 17), torch.float32)
    monkeypatch.setattr(rn, "_launch_forward", lambda *args: pytest.fail("unsupported transform"))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(r, torch.ones_like(r))
        actual = rn.fused_residual_norm(dual, a, w, 1e-6)
        expected = qwen_reference(dual, a, w)
        for x, y in zip(actual, expected):
            for got, want in zip(torch.autograd.forward_ad.unpack_dual(x),
                                 torch.autograd.forward_ad.unpack_dual(y)):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(x) for x in (r, a, w)]
        outputs = rn.fused_residual_norm(*fake, 1e-6)
        assert all(type(x).__name__ == "FakeTensor" and x.shape == r.shape for x in outputs)
    traced = torch.jit.trace(lambda r, a, w: rn.fused_residual_norm(r, a, w, 1e-6),
                             (r, a, w), check_trace=False)
    for got, want in zip(traced(r, a, w), qwen_reference(r, a, w)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_batched_backward_keeps_differentiable_fallback(interpreted):
    r, a, w = make_inputs((2, 17), torch.float32)
    actual = rn.fused_residual_norm(r, a, w, 1e-6)
    expected = qwen_reference(r, a, w)
    gradients = tuple(torch.randn(3, *r.shape) for _ in range(2))
    got = torch.autograd.grad(actual, (r, a), gradients, is_grads_batched=True)
    want = torch.autograd.grad(expected, (r, a), gradients, is_grads_batched=True)
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(triton.knobs.runtime.interpret, reason="ordinary JIT binder required")
@pytest.mark.parametrize("hidden", [17, 1536])
def test_dynamic_rows_and_gradient_strides_do_not_inflate_jit_keys(monkeypatch, hidden):
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    from triton.runtime.jit import compute_cache_key, create_function_from_signature

    keys = [set(), set()]
    for index, name in enumerate(("residual_norm_fwd_kernel", "residual_norm_bwd_kernel")):
        kernel = getattr(rn, name)
        binder = create_function_from_signature(kernel.signature, kernel.params,
                                                 make_backend(GPUTarget("cuda", 75, 32)))
        class Capture:
            def __init__(self, bind, found):
                self.bind, self.found = bind, found
            def __getitem__(self, grid):
                def capture(*args, **kwargs):
                    _, specialization, options = self.bind(*args, **kwargs)
                    self.found.add(compute_cache_key({}, specialization, options))
                return capture
        monkeypatch.setattr(rn, name, Capture(binder, keys[index]))
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    for rows in (1, 3, 16, 17, 65, 4 * 127, 4 * 256, 4 * 512):
        r, a, w = make_inputs((rows, hidden))
        s, y, inv = rn._launch_forward(r, a, w, 1e-6)
        for gs, gy in ((torch.ones_like(s), torch.randn(rows, hidden, 2)[..., 0]),
                       (torch.ones(1).expand_as(s), None), (None, torch.ones_like(s))):
            rn._launch_backward(s, w, inv, gs, gy, True, True, a.dtype)
    assert [len(found) for found in keys] == [1, 1]


@pytest.mark.parametrize("attention_dtype", ["fp16", "fp32"])
@pytest.mark.skipif(triton.knobs.runtime.interpret, reason="ordinary JIT compiler required")
@pytest.mark.parametrize("backward,need_r,need_a", [(False, True, True), (True, True, True),
                                                   (True, False, True), (True, True, False)])
def test_offline_sm75_compile(attention_dtype, backward, need_r, need_a):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if backward:
        kernel = rn.residual_norm_bwd_kernel
        signature = {name: "*fp32" for name in ("S", "W", "INV", "GS", "GY", "DR", "DA")}
        signature["DA"] = f"*{attention_dtype}" if need_a else "*fp32"
        signature.update({name: "i32" for name in ("GS0", "GS1", "GY0", "GY1", "HAS_GS", "HAS_GY")})
        constants = dict(H=1536, BLOCK=2048, NEED_R=need_r, NEED_A=need_a)
    else:
        kernel = rn.residual_norm_fwd_kernel
        signature = {name: "*fp32" for name in ("R", "A", "W", "S", "Y", "INV")}
        signature.update(A=f"*{attention_dtype}", EPS="fp32")
        constants = dict(H=1536, BLOCK=2048)
    compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants),
                              target=GPUTarget("cuda", 75, 32),
                              options={"num_warps": 4, "enable_fp_fusion": False})
    assert compiled.asm["cubin"]
    assert ".target sm_75" in compiled.asm["ptx"]
    assert "fma." not in compiled.asm["ptx"]


@pytest.mark.parametrize("weight_grad", [False, True])
def test_cpu_primitive_without_device_override(weight_grad):
    r, a, w = make_inputs((2, 17))
    w.requires_grad_(weight_grad)
    got = rn.fused_residual_norm(r, a, w, 1e-6)
    want = qwen_reference(r, a, w)
    assert type(got[0].grad_fn).__name__ != "_ResidualNormBackward"
    variables = (r, a, w) if weight_grad else (r, a)
    got_grads = torch.autograd.grad(sum(x.sum() for x in got), variables)
    want_grads = torch.autograd.grad(sum(x.sum() for x in want), variables)
    for x, y in zip(got + got_grads, want + want_grads):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.mark.parametrize("class_name", ["Qwen2RMSNorm", "Qwen2DecoderLayer"])
def test_patch_rejects_unknown_wrapped_forward(monkeypatch, class_name):
    from transformers.models.qwen2 import modeling_qwen2

    cls = getattr(modeling_qwen2, class_name)
    original = cls.forward
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs) * 2
    monkeypatch.setattr(cls, "forward", wrapped)
    try:
        assert rn.patch() is False
        assert cls.forward is wrapped
    finally:
        rn.unpatch()


CUDA = pytest.mark.skipif(
    triton.knobs.runtime.interpret or not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA with the real JIT; CPU/interpreter results are not GPU evidence",
)


def check_cuda_residual(shape, dtype, scale, needs=(True, True), which=(0, 1), cancel=False):
    r, a, w = make_inputs(shape, dtype, *needs, device="cuda")
    eps = 1. if cancel else 1e-6
    if cancel:
        with torch.no_grad():
            r.zero_(), a.zero_(), w.fill_(1)
    actual = rn.fused_residual_norm(r, a, w, eps)
    assert type(actual[0].grad_fn).__name__ == "_ResidualNormBackward"
    assert all(x.dtype == torch.float32 and x.is_cuda for x in actual)
    gs = (torch.randn(*shape, 2, device="cuda") * (scale / 32))[..., 0]
    gy = (torch.randn((), device="cuda") * (scale / 32)).expand(shape)
    upstream = (torch.full_like(r, -65536), torch.full_like(r, 65536)) if cancel else (gs, gy)
    def gradients(outputs, inputs):
        return torch.autograd.grad(tuple(outputs[i] for i in which),
                                   tuple(x for x in inputs if x.requires_grad),
                                   tuple(upstream[i] for i in which))
    got = gradients(actual, (r, a))
    for use_team in (False, True):
        rr, aa = (x.detach().clone().requires_grad_(need) for x, need in zip((r, a), needs))
        summed = rr + aa
        expected = (summed, team.rmsnorm(summed, w, eps)) if use_team else qwen_reference(rr, aa, w, eps)
        want = gradients(expected, (rr, aa))
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
        for x, y in zip(got, want):
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            tol = 2e-3 if x.dtype == torch.float16 else 1e-4
            torch.testing.assert_close(x.float() / scale, y.float() / scale, atol=tol, rtol=tol)
    if cancel:
        assert all(torch.count_nonzero(x) == 0 for x in got)


@CUDA
@pytest.mark.parametrize("shape", [(2, 3, 17), (1, 3, 1001), (4, 127, 1536), (4, 256, 1536), (4, 512, 1536)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [1, 65536])
def test_cuda_outputs_and_input_gradients(shape, dtype, scale):
    check_cuda_residual(shape, dtype, scale)


@CUDA
@pytest.mark.parametrize("needs", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("which", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("hidden", [17, 1536])
def test_cuda_frozen_inputs_and_missing_output_gradients(needs, which, hidden):
    check_cuda_residual((2, 3, hidden), torch.float16, 65536, needs, which)


@CUDA
@pytest.mark.parametrize("hidden", [17, 1536])
def test_cuda_scaled_branches_cancel_before_half_cast(hidden):
    check_cuda_residual((2, hidden), torch.float16, 65536, cancel=True)


@CUDA
def test_cuda_varying_rows_reuse_actual_compiled_kernels():
    kernels = (rn.residual_norm_fwd_kernel, rn.residual_norm_bwd_kernel)
    previous = None
    for rows in (1, 3, 16, 17, 65):
        r, a, w = make_inputs((rows, 17), device="cuda")
        out = rn.fused_residual_norm(r, a, w, 1e-6)
        assert type(out[0].grad_fn).__name__ == "_ResidualNormBackward"
        for which in ((0, 1), (0,), (1,)):
            upstream = (torch.randn(rows, 17, 2, device="cuda")[..., 0],
                        torch.ones((), device="cuda").expand_as(r))
            torch.autograd.grad(tuple(out[i] for i in which), (r, a),
                                tuple(upstream[i] for i in which), retain_graph=True)
        torch.cuda.synchronize()
        caches = [kernel.device_caches[torch.cuda.current_device()][0] for kernel in kernels]
        assert all(cache and all(k.asm["cubin"] for k in cache.values()) for cache in caches)
        current = [set(cache) for cache in caches]
        if previous is not None:
            assert current == previous
        previous = current


@CUDA
@pytest.mark.parametrize("scale", [1, 65536])
def test_cuda_qwen_peft_sdpa_autocast_gradscaler(patched, monkeypatch, scale):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2Model

    # The patched fixture explicitly calls raw rn.patch(), irrespective of ENABLED.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(41)
        config = Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
                             num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                             attention_dropout=.1, use_cache=False)
        config._attn_implementation = "sdpa"
        candidate = get_peft_model(Qwen2Model(config), LoraConfig(
            task_type="FEATURE_EXTRACTION", r=4, lora_alpha=8, lora_dropout=.1,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )).cuda().train()
        with torch.no_grad():
            for name, parameter in candidate.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=.01)  # Exercise A and B adapter gradients on this step.
        for module in candidate.modules():
            if "RMSNorm" in type(module).__name__:
                assert module.weight.dtype == torch.float32 and not module.weight.requires_grad
        reference = copy.deepcopy(candidate)
        for layer in reference.get_base_model().layers:
            layer.forward = types.MethodType(rn._original, layer)
        calls, results = [], []
        launch_forward = rn._launch_forward
        def candidate_launch_forward(residual, attention, weight, eps):
            assert residual.dtype == weight.dtype == torch.float32 and not weight.requires_grad
            calls.append((residual.requires_grad, attention.dtype))
            return launch_forward(residual, attention, weight, eps)
        monkeypatch.setattr(rn, "_launch_forward", candidate_launch_forward)
        ids = torch.randint(0, config.vocab_size, (2, 17), device="cuda")
        probe = torch.randn(2, 17, config.hidden_size, device="cuda")
        rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        for model in (reference, candidate):
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
            scaler = torch.amp.GradScaler("cuda", init_scale=scale)
            assert scaler.is_enabled()
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(input_ids=ids, use_cache=False).last_hidden_state
                loss = (output.float() * probe).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            results.append((output.detach(), torch.get_rng_state(), torch.cuda.get_rng_state()))
            if model is reference:
                assert not calls
        assert calls == [(False, torch.float16), (True, torch.float16)]
        assert results[0][0].dtype == results[1][0].dtype == torch.float32
        torch.testing.assert_close(results[1][0], results[0][0], atol=2e-4, rtol=2e-3)
        assert all(torch.equal(a, b) for a, b in zip(results[0][1:], results[1][1:]))
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            expected = reference_parameters[name]
            if not parameter.requires_grad:
                assert parameter.grad is expected.grad is None, name
                continue
            assert "lora_" in name and parameter.grad is not None and expected.grad is not None, name
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(expected.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-5, rtol=2e-3, msg=name)
        for adapter in ("lora_A", "lora_B"):
            assert any(torch.count_nonzero(p.grad) > 0 for n, p in candidate.named_parameters()
                       if adapter in n)


def test_real_fx_tracing_uses_eager_fallback(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx
    monkeypatch.setattr(rn,'_cuda_supported',lambda x:True)
    def forbidden(*args):
        raise AssertionError('custom kernel launched during real-mode FX tracing')
    monkeypatch.setattr(rn,'_launch_forward',forbidden)
    r=torch.randn(2,17,requires_grad=True)
    a=torch.randn(2,17,dtype=torch.float16,requires_grad=True)
    w=torch.ones(17)
    trace=make_fx(lambda r,a,w:rn.fused_residual_norm(r,a,w,1e-6),tracing_mode='real')(r,a,w)
    got=trace(r,a,w);want=rn._eager(r,a,w,1e-6)
    for x,y in zip(got,want):torch.testing.assert_close(x,y,atol=0,rtol=0)
