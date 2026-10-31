"""RoPE oracles, conservative dispatch, and opt-in kernel checks.

CPU/interpreter/offline compilation results are not evidence of GPU correctness.
Run the interpreter separately with TRITON_INTERPRET=1 and -k interpreter.
"""

import builtins
import contextlib
import importlib.util
import inspect
import io
import os
import sys
import types

import pytest
import torch

from solution import rotary as rope


def reference(q, k, cos, sin, unsqueeze_dim=1):
    def rotate(x):
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat((-second, first), dim=-1)

    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * c + rotate(q) * s, k * c + rotate(k) * s


def transpose_reference(g, cos, sin):
    # Multiply BEFORE exchanging halves: sine halves need not be equal.
    direct, rotated = g * cos.unsqueeze(1), g * sin.unsqueeze(1)
    first, second = rotated.chunk(2, dim=-1)
    return direct + torch.cat((second, -first), dim=-1)


CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="requires CUDA and Triton; CPU checks are not GPU evidence",
)
INTERPRETER = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",  # lint: ok -- test-only interpreter/full-shape opt-in
    reason="requires opt-in TRITON_INTERPRET=1; not GPU validation",
)
BACKENDS = ["cpu", pytest.param("cuda", marks=CUDA),
            pytest.param("interpreter", marks=INTERPRETER)]
DTYPES = [(torch.float32, torch.float32), (torch.float16, torch.float16),
          (torch.float32, torch.float16)]


@pytest.fixture
def backend(request, monkeypatch):
    value = request.param
    if value == "interpreter":
        pytest.importorskip("triton")
        # Explicitly opt into CPU interpretation of the CUDA-only candidate.
        # Production dispatch is tested separately and never has this override.
        monkeypatch.setattr(rope, "_supports_triton", lambda *args: True)
        monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    return value


def inputs(shape=(2, 17, 12, 2, 128), dtype=torch.float32,
           trig_dtype=None, device="cpu", layout="transposed", trig_batch=None):
    b, t, hq, hk, d = shape
    trig_dtype = dtype if trig_dtype is None else trig_dtype
    trig_batch = b if trig_batch is None else trig_batch
    torch.manual_seed(173)

    def activation(heads):
        if layout == "contiguous":
            return torch.randn(b, heads, t, d, dtype=dtype, device=device)
        # Nonzero storage offset, optional gaps, and independent Q/K strides.
        step = 2 if layout == "gapped" else 1
        return torch.randn(b, t * step + 1, heads, d, dtype=dtype,
                           device=device)[:, 1::step].transpose(1, 2)

    q, k = activation(hq), activation(hk)
    # These intentionally have different halves, arbitrary values and strides.
    c = torch.randn(trig_batch, t, d * 2, dtype=trig_dtype, device=device)[..., ::2]
    s = torch.randn(t, trig_batch, d, dtype=trig_dtype, device=device).transpose(0, 1)
    return q, k, c, s


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("shape,layout,trig_batch", [
    ((2, 17, 12, 2, 128), "transposed", 1),
    ((2, 5, 3, 1, 66), "gapped", 2),
    ((1, 1, 1, 1, 2), "contiguous", 1),
    ((1, 3, 5, 2, 130), "contiguous", 1),
])
def test_forward_backward_parity(backend, dtypes, shape, layout, trig_batch):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs(shape, *dtypes, device, layout, trig_batch)
    q.requires_grad_()
    k.requires_grad_()
    qr, kr = (x.detach().clone().requires_grad_() for x in (q, k))
    before = [x.clone() for x in (q, k, c, s)]
    actual = rope.apply_rotary_pos_emb(q, k, c, s)
    expected = reference(qr, kr, c, s)
    upstream = [torch.randn(*x.shape, 2, dtype=x.dtype, device=device)[..., 0]
                for x in actual]
    if backend != "cpu":
        assert type(actual[0].grad_fn).__name__ == "_FrozenRoPEBackward"
    gradients = torch.autograd.grad(actual, (q, k), upstream)
    reference_gradients = torch.autograd.grad(expected, (qr, kr), upstream)
    for value, oracle in zip(actual + gradients, expected + reference_gradients):
        exact(value, oracle)
    for value, original in zip((q, k, c, s), before):
        exact(value, original)


def varying_shape_inputs(dtypes, device, geometry):
    # Fixed model geometry, changing padding/batch/layout. Revisit lengths with
    # different contents and strides. Keep pointer alignment fixed (small-D
    # cases use contiguous allocations) to isolate scalar specialization.
    if geometry == "qwen":
        hq, hk, d = 12, 2, 128
        cases = [(1, 1, "contiguous", 1), (2, 3, "transposed", 1),
                 (1, 16, "gapped", 1), (2, 17, "contiguous", 2),
                 (3, 31, "transposed", 1), (1, 32, "gapped", 1),
                 (2, 65, "contiguous", 2), (2, 3, "gapped", 2)]
    elif geometry == "unit_counts":
        # NQ/NK=1, nonmultiples/multiples of 16, and Q_BLOCKS=1 -> 2.
        hq, hk, d = 1, 1, 2
        cases = [(1, t, "contiguous", 1) for t in (1, 15, 16, 17, 255, 256, 257)]
    else:
        assert geometry == "block_tails"
        # Q crosses a block boundary at 9 -> 10 tokens; K at 28 -> 29.
        hq, hk, d = 3, 1, 18
        cases = [(1, t, "contiguous", 1) for t in (9, 10, 28, 29)]
    for i, (b, t, layout, trig_batch) in enumerate(cases):
        q, k, c, s = inputs((b, t, hq, hk, d), *dtypes, device, layout, trig_batch)
        q.add_(i / 8)
        k.sub_(i / 16)
        if i % 3 == 1:
            c = c[..., :1].expand_as(c)
            s = s[:, :1].expand_as(s)
        elif i % 3 == 2:
            c = c.contiguous()
            s = torch.randn(trig_batch, t, d * 3, dtype=s.dtype, device=device)[..., ::3]
            q = q[:, :1].expand_as(q)
        yield q, k, c, s


def varying_shape_gradients(outputs, index):
    gradients = []
    for x in outputs:
        b, h, t, d = x.shape
        if index % 3 == 0:
            g = torch.randn(*x.shape, 3, dtype=x.dtype, device=x.device)[..., 0]
        elif index % 3 == 1:
            g = torch.randn((), dtype=x.dtype, device=x.device).expand_as(x)
        else:
            g = torch.randn(b, t, h, d, dtype=x.dtype, device=x.device).transpose(1, 2)
        gradients.append(g)
    return tuple(gradients)


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="inspect ordinary JIT keys")  # lint: ok -- test-only interpreter/full-shape opt-in
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("geometry", ["qwen", "unit_counts", "block_tails"])
def test_varying_shapes_have_bounded_jit_keys(monkeypatch, dtypes, geometry):
    """Use the real NVIDIA argument binder/cache key without a CUDA driver."""
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    from triton.runtime.jit import compute_cache_key, create_function_from_signature

    kernel = rope._triton_kernel()
    binder = create_function_from_signature(kernel.signature, kernel.params,
                                            make_backend(GPUTarget("cuda", 75, 32)))
    keys = set()

    class CaptureLaunch:
        def __getitem__(self, grid):
            assert grid[0] > 0

            def capture(*args, **kwargs):
                _, specialization, options = binder(*args, **kwargs)
                keys.add(compute_cache_key({}, specialization, options))

            return capture

    monkeypatch.setattr(rope, "_triton_kernel", lambda: CaptureLaunch())
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    for i, (q, k, c, s) in enumerate(varying_shape_inputs(dtypes, "cpu", geometry)):
        outputs = rope._launch(q, k, c, s, q.shape, k.shape)
        gq, gk = varying_shape_gradients(outputs, i)
        for dq, dk in ((gq, gk), (gq, None), (None, gk)):
            rope._launch(dq, dk, c, s, q.shape, k.shape, backward=True)
    # Missing gradients use trig placeholders. Mixed trig/activation dtypes
    # therefore need distinct pointer signatures for Q-only and K-only bwd.
    assert len(keys) == (2 if dtypes[0] == dtypes[1] else 4)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("geometry", ["qwen", "unit_counts", "block_tails"])
def test_varying_shapes_reuse_kernels_with_exact_parity(monkeypatch, backend, dtypes, geometry):
    """CUDA additionally checks actual compiled-cache cardinality, not timing."""
    device = "cuda" if backend == "cuda" else "cpu"
    if backend == "cuda":
        if os.environ.get("TRITON_INTERPRET") == "1" or torch.version.hip is not None:  # lint: ok -- test-only interpreter/full-shape opt-in
            pytest.skip("compiled-cache validation requires NVIDIA CUDA in ordinary JIT mode")
        import triton

        original = rope._triton_kernel()
        # Fresh in-memory cache, independent of test order. Normal on-demand
        # compilation/launch only; no warmup or production cache manipulation.
        kernel = triton.jit(original.fn, do_not_specialize=original.do_not_specialize,
                            do_not_specialize_on_alignment=original.do_not_specialize_on_alignment)
        monkeypatch.setattr(rope, "_triton_kernel", lambda: kernel)

    for i, (q, k, c, s) in enumerate(varying_shape_inputs(dtypes, device, geometry)):
        q.requires_grad_()
        k.requires_grad_()
        qr, kr = (x.detach().clone().requires_grad_() for x in (q, k))
        actual = rope.apply_rotary_pos_emb(q, k, c, s)
        expected = reference(qr, kr, c, s)
        if backend != "cpu":
            assert type(actual[0].grad_fn).__name__ == "_FrozenRoPEBackward"
        upstream = varying_shape_gradients(actual, i)
        for indices in ((0, 1), (0,), (1,)):
            def gradients(outputs, activations):
                return torch.autograd.grad(tuple(outputs[j] for j in indices), activations,
                                           tuple(upstream[j] for j in indices),
                                           retain_graph=True, allow_unused=True)

            actual_grads = gradients(actual, (q, k))
            expected_grads = gradients(expected, (qr, kr))
            for value, oracle in zip(actual_grads, expected_grads):
                if oracle is None:
                    assert value is None
                else:
                    exact(value, oracle)
        for value, oracle in zip(actual, expected):
            exact(value, oracle)
    if backend == "cuda":
        torch.cuda.synchronize()
        cache = kernel.device_caches[torch.cuda.current_device()][0]
        assert len(cache) == (2 if dtypes[0] == dtypes[1] else 4)
        assert all(compiled.asm["cubin"] for compiled in cache.values())


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("requires", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("used", ["q", "k", "both"])
def test_independent_gradients_and_missing_output_grad(backend, requires, used):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 3, 3, 1, 18), device=device)
    for value, need in zip((q, k), requires):
        value.requires_grad_(need)
    result = rope.apply_rotary_pos_emb(q, k, c, s)
    assert tuple(x.requires_grad for x in result) == requires
    indices = [i for i, name in enumerate(("q", "k")) if used in (name, "both")]
    losses = [result[i].sum() for i in indices if requires[i]]
    if not losses:
        return
    sum(losses).backward()
    for i, value in enumerate((q, k)):
        if requires[i] and i in indices:
            exact(value.grad, transpose_reference(torch.ones_like(value), c, s))
        else:
            assert value.grad is None


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [2.0**-16, 4096.0])
def test_upstream_scale_and_multiply_rounding(backend, dtype, scale):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 7, 3, 2, 128), dtype=dtype, device=device)
    q.requires_grad_()
    k.requires_grad_()
    upstream = [torch.randn_like(x) * scale for x in (q, k)]
    outputs = rope.apply_rotary_pos_emb(q, k, c, s)
    gradients = torch.autograd.grad(outputs, (q, k), upstream)
    expected = reference(q, k, c, s)
    expected_gradients = torch.autograd.grad(expected, (q, k), upstream)
    for actual, oracle in zip(outputs + gradients, expected + expected_gradients):
        exact(actual, oracle)
    if dtype == torch.float16:
        # Ensure this fixture actually catches removal of intermediate casts.
        wrong_forward = reference(q.float(), k.float(), c.float(), s.float())[0].half()
        wrong_backward = transpose_reference(upstream[0].float(), c.float(), s.float()).half()
        assert not torch.equal(wrong_forward, expected[0])
        assert not torch.equal(wrong_backward, expected_gradients[0])


def test_patch_signature_idempotence_and_original_fallback(monkeypatch):
    calls = []

    def original(q, k, cos, sin, unsqueeze_dim=1):
        calls.append((q, k, cos, sin, unsqueeze_dim))
        return reference(q, k, cos, sin, unsqueeze_dim)

    name = "transformers.models.qwen2.modeling_qwen2"
    fake = types.ModuleType(name)
    fake.apply_rotary_pos_emb = original
    monkeypatch.setitem(sys.modules, name, fake)
    assert rope.patch_rope()
    patched = fake.apply_rotary_pos_emb
    assert inspect.signature(patched) == inspect.signature(original)
    assert patched._soup_rope_original is original
    assert rope.patch_rope()
    assert fake.apply_rotary_pos_emb is patched
    q, k, c, s = inputs()
    for actual, expected in zip(patched(q, k, c, s), original(q, k, c, s)):
        exact(actual, expected)
    assert len(calls) == 2
    assert all(actual is expected for actual, expected in zip(calls[0][:4], (q, k, c, s)))


def test_patch_missing_optional_function(monkeypatch):
    name = "transformers.models.qwen2.modeling_qwen2"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    assert rope.patch_rope() is False


def test_missing_triton_uses_saved_original(monkeypatch):
    name = "transformers.models.qwen2.modeling_qwen2"
    fake = types.ModuleType(name)
    marker = object()
    fake.apply_rotary_pos_emb = lambda q, k, cos, sin, unsqueeze_dim=1: marker
    monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(rope, "_supports_triton", lambda *args: True)
    monkeypatch.setattr(rope, "_triton_kernel", lambda: None)
    assert rope.patch_rope()
    tensors = inputs()
    assert fake.apply_rotary_pos_emb(*tensors) is marker
    for actual, expected in zip(rope.apply_rotary_pos_emb(*tensors), reference(*tensors)):
        exact(actual, expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("case", [
    "trainable_cos", "trainable_sin", "trainable_both", "last_stride", "bf16", "double",
    "half_with_float_trig", "mixed_qk", "mixed_trig", "odd", "wide", "empty",
    "broadcast_tokens", "broadcast_width", "broadcast_k", "other_axis", "negative_view",
])
def test_unsupported_calls_retain_eager_outputs_and_gradients(monkeypatch, device, case):
    shape = (2, 5, 3, 1, 17 if case == "odd" else 258 if case == "wide" else 18)
    q, k, c, s = inputs(shape, device=device)
    dim = 1
    if case == "last_stride":
        q = torch.randn(2, 3, 5, 36, device=device)[..., ::2]
    elif case in ("bf16", "double"):
        dtype = torch.bfloat16 if case == "bf16" else torch.float64
        q, k, c, s = (x.to(dtype) for x in (q, k, c, s))
    elif case == "half_with_float_trig":
        q, k = q.half(), k.half()
    elif case == "mixed_qk":
        k = k.half()
    elif case == "mixed_trig":
        s = s.half()
    elif case == "empty":
        q, k, c, s = q[:, :, :0], k[:, :, :0], c[:, :0], s[:, :0]
    elif case == "broadcast_tokens":
        c, s = c[:, :1], s[:, :1]
    elif case == "broadcast_width":
        c, s = c[..., :1], s[..., :1]
    elif case == "broadcast_k":
        k = k[:, :, :1]
    elif case == "other_axis":
        q, k, dim = q.transpose(1, 2), k.transpose(1, 2), 2
    elif case == "negative_view":
        q = torch._neg_view(q)
        assert q.is_neg()
    q.requires_grad_()
    k.requires_grad_()
    c.requires_grad_(case in ("trainable_cos", "trainable_both"))
    s.requires_grad_(case in ("trainable_sin", "trainable_both"))
    originals = (q, k, c, s)
    copies = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in originals)

    def reject_kernel():
        pytest.fail("unsupported call attempted to load Triton")

    monkeypatch.setattr(rope, "_triton_kernel", reject_kernel)
    actual = rope.apply_rotary_pos_emb(*originals, unsqueeze_dim=dim)
    expected = reference(*copies, unsqueeze_dim=dim)
    upstream = [torch.randn_like(x) for x in actual]
    gradients = torch.autograd.grad(actual, [x for x in originals if x.requires_grad], upstream)
    expected_gradients = torch.autograd.grad(expected, [x for x in copies if x.requires_grad], upstream)
    for value, oracle in zip(actual + gradients, expected + expected_gradients):
        exact(value, oracle)


@pytest.mark.parametrize("axis", [True, 1.0, None, "1"])
def test_invalid_axis_is_not_coerced_by_patch(monkeypatch, axis):
    name = "transformers.models.qwen2.modeling_qwen2"
    fake = types.ModuleType(name)
    received = []

    def original(q, k, cos, sin, unsqueeze_dim=1):
        received.append(unsqueeze_dim)
        return reference(q, k, cos, sin, unsqueeze_dim)

    fake.apply_rotary_pos_emb = original
    monkeypatch.setitem(sys.modules, name, fake)
    assert rope.patch_rope()
    with pytest.raises(TypeError):
        fake.apply_rotary_pos_emb(*inputs(), unsqueeze_dim=axis)
    assert received == [axis]


def test_cpu_import_and_execution_do_not_import_triton(monkeypatch):
    original_import = builtins.__import__

    def reject_triton(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            pytest.fail("CPU RoPE attempted to import Triton")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_triton)
    spec = importlib.util.spec_from_file_location("isolated_rope", rope.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q, k, c, s = inputs()
    q.requires_grad_()
    result = module.apply_rotary_pos_emb(q, k, c, s)
    result[0].sum().backward()
    assert q.grad is not None
    assert "triton" not in module.__dict__


def test_patch_import_has_no_side_effect_and_matches_installed_signature():
    pytest.importorskip("transformers")
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    spec = importlib.util.spec_from_file_location("unpatched_rope", rope.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert modeling_qwen2.apply_rotary_pos_emb is original
    assert inspect.signature(module.apply_rotary_pos_emb) == inspect.signature(original)
    q, k, c, s = inputs()
    for actual, expected in zip(module.apply_rotary_pos_emb(q, k, c, s), original(q, k, c, s)):
        exact(actual, expected)


@pytest.mark.parametrize("backend", BACKENDS[1:], indirect=True)
def test_only_frozen_trig_is_saved(backend):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 5, 3, 1, 18), device=device)
    q.requires_grad_()
    k.requires_grad_()
    saved = []

    def pack(x):
        saved.append(x)
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        outputs = rope.apply_rotary_pos_emb(q, k, c, s)
    assert len(saved) == 2
    assert saved[0] is c and saved[1] is s
    assert outputs[0].grad_fn is outputs[1].grad_fn
    sum(x.sum() for x in outputs).backward()


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_no_grad_and_inference(backend, mode):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 3, 3, 1, 18), device=device)
    q.requires_grad_()
    k.requires_grad_()
    with mode():
        actual = rope.apply_rotary_pos_emb(q, k, c, s)
        expected = reference(q, k, c, s)
    for value, oracle in zip(actual, expected):
        assert not value.requires_grad
        exact(value, oracle)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_higher_derivatives_and_gradient_dependency(backend, dtype):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 3, 3, 1, 18), dtype=dtype, device=device)
    q.requires_grad_()
    k.requires_grad_()
    qr, kr = (x.detach().clone().requires_grad_() for x in (q, k))
    actual = rope.apply_rotary_pos_emb(q, k, c, s)
    expected = reference(qr, kr, c, s)
    loss = sum(x.square().sum() for x in actual)
    ref_loss = sum(x.square().sum() for x in expected)
    grads = torch.autograd.grad(loss, (q, k), create_graph=True)
    ref_grads = torch.autograd.grad(ref_loss, (qr, kr), create_graph=True)
    curvature = torch.autograd.grad(sum(x.sum() for x in grads), (q, k))
    ref_curvature = torch.autograd.grad(sum(x.sum() for x in ref_grads), (qr, kr))
    for value, oracle in zip(grads + curvature, ref_grads + ref_curvature):
        exact(value, oracle)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_large_small_and_nonfinite_activations(backend, dtype):
    device = "cuda" if backend == "cuda" else "cpu"
    values = [0.0, -0.0, 2.0**-24, -2.0**-24, 1.001, -1.001, 60000, -60000,
              float("inf"), -float("inf"), float("nan"), 0.5]
    q = torch.tensor(values, device=device, dtype=dtype).reshape(1, 1, 1, -1).requires_grad_()
    k = q.detach().clone().requires_grad_()
    c = torch.tensor([2, -2, .25, .25, 1, -1, 2, -2, .5, .5, 1, 1],
                     dtype=dtype, device=device).reshape(1, 1, -1)
    s = c.flip(-1)
    outputs = rope.apply_rotary_pos_emb(q, k, c, s)
    refs = reference(q, k, c, s)
    grad = q.detach()
    gradients = torch.autograd.grad(outputs, (q, k), (grad, grad))
    ref_gradients = torch.autograd.grad(refs, (q, k), (grad, grad))
    for value, oracle in zip(outputs + gradients, refs + ref_gradients):
        exact(value, oracle)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_forward_ad_uses_eager_fallback(monkeypatch, device):
    q, k, c, s = inputs((1, 3, 3, 1, 18), device=device)

    def reject_kernel():
        pytest.fail("forward AD call attempted to load Triton")

    monkeypatch.setattr(rope, "_triton_kernel", reject_kernel)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(q, torch.ones_like(q))
        actual = rope.apply_rotary_pos_emb(dual, k, c, s)
        expected = reference(dual, k, c, s)
        a = torch.autograd.forward_ad.unpack_dual(actual[0])
        e = torch.autograd.forward_ad.unpack_dual(expected[0])
        exact(a.primal, e.primal)
        exact(a.tangent, e.tangent)


@pytest.mark.parametrize("backend", BACKENDS[1:], indirect=True)
def test_forward_ad_through_incoming_gradient_uses_eager_backward(backend):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 3, 3, 1, 18), device=device)
    q.requires_grad_()
    qr = q.detach().clone().requires_grad_()
    output = rope.apply_rotary_pos_emb(q, k, c, s)[0]
    expected = reference(qr, k, c, s)[0]
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.randn_like(q), torch.ones_like(q))
        dq, = torch.autograd.grad(output, q, dual, create_graph=False)
        ref_dq, = torch.autograd.grad(expected, qr, dual, create_graph=False)
        actual_dual = torch.autograd.forward_ad.unpack_dual(dq)
        expected_dual = torch.autograd.forward_ad.unpack_dual(ref_dq)
        assert actual_dual.tangent is not None
        exact(actual_dual.primal, expected_dual.primal)
        exact(actual_dual.tangent, expected_dual.tangent)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("used", ["q", "k", "both"])
@pytest.mark.parametrize("batching", ["legacy", "functorch"])
def test_batched_incoming_gradients_use_eager_backward(monkeypatch, backend, dtypes, used, batching):
    device = "cuda" if backend == "cuda" else "cpu"
    q, k, c, s = inputs((1, 3, 3, 1, 18), *dtypes, device)
    q.requires_grad_()
    k.requires_grad_()
    qr, kr = (x.detach().clone().requires_grad_() for x in (q, k))
    if backend == "cpu":
        # Exercise the real custom backward without requiring a CUDA forward.
        monkeypatch.setattr(rope, "_launch", lambda q, k, c, s, *args: reference(q, k, c, s))
        actual = rope._FrozenRoPE.apply(q, k, c, s)
    else:
        actual = rope.apply_rotary_pos_emb(q, k, c, s)
    assert type(actual[0].grad_fn).__name__ == "_FrozenRoPEBackward"
    expected = reference(qr, kr, c, s)
    indices = [i for i, name in enumerate(("q", "k")) if used in (name, "both")]
    actual = tuple(actual[i] for i in indices)
    expected = tuple(expected[i] for i in indices)
    # A noncontiguous batch of distinct vectors catches lost/merged batch axes.
    upstream = tuple(torch.randn(4, *x.shape, 2, dtype=x.dtype, device=device)[..., 0]
                     for x in actual)

    def gradients(outputs, activations):
        if batching == "legacy":
            return torch.autograd.grad(outputs, activations, upstream,
                                       is_grads_batched=True, create_graph=False,
                                       allow_unused=True)
        out_dims = tuple(0 if i in indices else None for i in range(2))
        return torch.vmap(lambda *g: torch.autograd.grad(
            outputs, activations, g, retain_graph=True, create_graph=False,
            allow_unused=True), out_dims=out_dims)(*upstream)

    def reject_launch(*args, **kwargs):
        pytest.fail("batched incoming gradient reached Triton launch")

    expected_gradients = gradients(expected, (qr, kr))
    monkeypatch.setattr(rope, "_launch", reject_launch)
    actual_gradients = gradients(actual, (q, k))
    for value, oracle in zip(actual_gradients, expected_gradients):
        if oracle is None:
            assert value is None
        else:
            exact(value, oracle)


def test_real_fx_trace_uses_eager_graph(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    tensors = inputs((1, 3, 3, 1, 18))

    def reject_candidate_dispatch(x):
        pytest.fail("real-mode FX tracing reached candidate tensor dispatch")

    monkeypatch.setattr(rope, "_plain_strided", reject_candidate_dispatch)
    traced = make_fx(lambda q, k, c, s: rope.apply_rotary_pos_emb(q, k, c, s),
                     tracing_mode="real")(*tensors)
    for actual, expected in zip(traced(*tensors), reference(*tensors)):
        exact(actual, expected)


def test_jit_trace_uses_serializable_eager_graph(monkeypatch):
    def reject_layout_check(x):
        pytest.fail("JIT tracing reached candidate tensor dispatch")

    # Prove tracing exits before device dispatch even on a CPU-only host.
    monkeypatch.setattr(rope, "_plain_strided", reject_layout_check)
    tensors = inputs((1, 3, 3, 1, 18))
    traced = torch.jit.trace(rope.apply_rotary_pos_emb, tensors, check_trace=False)
    artifact = io.BytesIO()
    torch.jit.save(traced, artifact)
    artifact.seek(0)
    reloaded = torch.jit.load(artifact)
    for actual, expected in zip(reloaded(*tensors), reference(*tensors)):
        exact(actual, expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_named_inputs_keep_names_on_eager_fallback(monkeypatch, device):
    q, k, c, s = inputs((1, 3, 3, 1, 18), device=device)
    names = ("batch", "heads", "tokens", "features")
    q, k = q.refine_names(*names), k.refine_names(*names)
    assert not rope._plain_strided(q)

    def reject_kernel():
        pytest.fail("named tensor attempted to load Triton")

    monkeypatch.setattr(rope, "_triton_kernel", reject_kernel)
    for actual, expected in zip(rope.apply_rotary_pos_emb(q, k, c, s), reference(q, k, c, s)):
        assert actual.names == expected.names == names
        exact(actual.rename(None), expected.rename(None))


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_real_qwen_attention_output_and_parameter_gradients(monkeypatch, backend, dtype):
    pytest.importorskip("transformers")
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    device = "cuda" if backend == "cuda" else "cpu"
    torch.manual_seed(21)
    config = Qwen2Config(hidden_size=192, intermediate_size=256, num_hidden_layers=1,
                        num_attention_heads=12, num_key_value_heads=2,
                        max_position_embeddings=32, attention_dropout=0.2)
    config._attn_implementation = "eager"
    attention = modeling_qwen2.Qwen2Attention(config, layer_idx=0).to(device=device, dtype=dtype).train()
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(config=config).to(device=device)
    x = torch.randn(2, 7, 192, dtype=dtype, device=device).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    # Real generation gives singleton-batch trig for a two-row batch.
    trig = rotary(x, torch.arange(7, device=device).unsqueeze(0))
    assert trig[0].shape == (1, 7, 16)
    mask = torch.full((7, 7), -float("inf"), device=device, dtype=dtype).triu(1)
    params = tuple(attention.parameters())
    current = modeling_qwen2.apply_rotary_pos_emb
    original = getattr(current, "_soup_rope_original", current)
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb", original)
    torch.manual_seed(39)
    expected = attention(xr, trig, mask)[0]
    upstream = torch.randn_like(expected)
    ref_grads = torch.autograd.grad(expected, (xr,) + params, upstream)
    launches = []
    original_launch = rope._launch

    def record_launch(*args, **kwargs):
        launches.append(kwargs.get("backward", False))
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(rope, "_launch", record_launch)
    assert rope.patch_rope()
    torch.manual_seed(39)
    actual = attention(x, trig, mask)[0]
    grads = torch.autograd.grad(actual, (x,) + params, upstream)
    assert tuple(attention.parameters()) == params
    assert launches == ([] if backend == "cpu" else [False, True])
    tolerance = ({"rtol": 0, "atol": 0} if backend != "cuda" else
                 {"rtol": 2e-3, "atol": 2e-3} if dtype == torch.float16 else
                 {"rtol": 2e-5, "atol": 2e-6})
    for value, oracle in zip((actual,) + grads, (expected,) + ref_grads):
        torch.testing.assert_close(value, oracle, **tolerance)


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="compile in ordinary JIT mode")  # lint: ok -- test-only interpreter/full-shape opt-in
@pytest.mark.parametrize("dtype,trig_dtype", [("fp32", "fp32"), ("fp16", "fp16"), ("fp32", "fp16")])
@pytest.mark.parametrize("backward,active,d", [(False, "both", 128), (True, "both", 66),
                                              (True, "q", 128), (True, "k", 18)])
def test_offline_sm75_compile(dtype, trig_dtype, backward, active, d):
    """Compile runtime shapes/strides; never initialize CUDA or claim execution."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    constants = dict(HQ=12, HK=2, D=d, BACKWARD=backward, BLOCK=256)
    signature = {key: f"*{dtype}" for key in ("Q", "K", "OQ", "OK")}
    signature.update(C=f"*{trig_dtype}", S=f"*{trig_dtype}")
    signature.update({key: "i32" for key in ("NQ", "NK", "Q_BLOCKS", "T", "QP", "KP")})
    signature.update({f"{name}{i}": "i32" for name, count in (("QS", 4), ("KS", 4),
                                                            ("CS", 3), ("SS", 3))
                      for i in range(count)})
    if active == "q":
        signature.update(K=f"*{trig_dtype}", OK=f"*{trig_dtype}")
    elif active == "k":
        signature.update(Q=f"*{trig_dtype}", OQ=f"*{trig_dtype}")
    compiled = triton.compile(
        ASTSource(rope._triton_kernel(), signature, constexprs=constants),
        target=GPUTarget("cuda", 75, 32),
        options={"num_warps": 4, "enable_fp_fusion": False},
    )
    assert compiled.asm["cubin"]
    ptx = compiled.asm["ptx"]
    assert ".target sm_75" in ptx
    assert "fma." not in ptx
