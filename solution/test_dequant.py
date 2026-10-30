"""Exact format and opt-in patch checks; CUDA tests require real hardware.

Run CPU kernel semantics separately with TRITON_INTERPRET=1. Neither those
checks nor offline compilation establish CUDA correctness or performance.
"""

import functools
import os
import struct
import sys
from types import SimpleNamespace

import pytest
import torch

from solution import dequant


# Independently specified binary32 NF4 format values, not the candidate table.
_NF4_BITS = (
    0xBF800000, 0xBF3239B1, 0xBF066B30, 0xBECA32A0,
    0xBE91A24D, 0xBE3D353F, 0xBDBA7871, 0x00000000,
    0x3DA2FAFF, 0x3E24CAE3, 0x3E7C04DD, 0x3EAD033A,
    0x3EE1A4B8, 0x3F1007AB, 0x3F3913B3, 0x3F800000,
)


def _codebook():
    return torch.tensor([struct.unpack("!f", struct.pack("!I", v))[0] for v in _NF4_BITS])


def _case(shape=(17, 65), nested=False):
    count = shape[0] * shape[1]
    groups = (count + 63) // 64
    codes = ((torch.arange(count) * 11 + 3) % 16).to(torch.uint8)
    padded = torch.cat((codes, torch.tensor([15], dtype=torch.uint8))) if count % 2 else codes
    packed = ((padded[::2] << 4) | padded[1::2]).view(-1, 1)
    state = SimpleNamespace(shape=torch.Size(shape), dtype=torch.float16,
                            quant_type="nf4", blocksize=64, nested=nested,
                            state2=None, offset=None)
    if nested:
        state.absmax = ((torch.arange(groups) * 73 + 9) % 256).to(torch.uint8)
        state.state2 = SimpleNamespace(
            code=torch.linspace(-0.9375, 0.9375, 256),
            absmax=torch.arange(1, (groups + 255) // 256 + 1).float() / 7,
            dtype=torch.float32, blocksize=256, nested=False, state2=None,
        )
        state.offset = torch.tensor(1.234567, dtype=torch.float32)
        # Explicitly round the product before the addition; no fused arithmetic.
        centered = (state.state2.code[state.absmax.long()].double()
                    * state.state2.absmax[torch.arange(groups) // 256].double()).float()
        scales = (centered.double() + state.offset.double()).float()
    else:
        scales = (torch.arange(groups).float() % 29) / 13
        state.absmax = scales.clone()
    dense = (_codebook()[codes.long()].double() * scales.repeat_interleave(64)[:count].double()).float().half()
    return packed, state, dense.reshape(shape)


def _to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{k: _to_cuda(v) for k, v in vars(value).items()})
    return value


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float16
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (3, 17), (17, 65), (129, 129)])
def test_reference_flat_groups_and_tails(nested, shape):
    packed, state, expected = _case(shape, nested)
    result = dequant.dequantize_nf4_reference(packed, state)
    _assert_bits(result, expected.t() if packed.shape[0] == 1 else expected)


@pytest.mark.parametrize("layout", ["flat", "column", "row"])
@pytest.mark.parametrize("nested", [False, True])
def test_reference_all_bytes_orientation_and_out(layout, nested):
    packed, state, _ = _case((16, 32), nested)
    packed = torch.arange(256, dtype=torch.uint8)
    if layout == "column":
        packed = packed[:, None]
    elif layout == "row":
        packed = packed[None, :]
    if nested:
        state.absmax.zero_()
        state.state2.code.zero_()
        state.offset.fill_(1)
    else:
        state.absmax.fill_(1)
    expected = _codebook()[torch.tensor([(b >> 4, b & 15) for b in range(256)])].half().reshape(16, 32)
    out = torch.full((16, 32), float("nan"), dtype=torch.float16)
    result = dequant.dequantize_nf4_reference(packed, state, out=out)
    _assert_bits(out, expected)
    _assert_bits(result, expected.t() if layout == "row" else expected)
    assert result.data_ptr() == out.data_ptr()


def _rounding_case():
    packed, state, _ = _case((1, 64), nested=True)
    packed.fill_(0xFF)
    state.absmax.zero_()
    state.state2.code[0] = 1 + 2**-23
    state.state2.absmax[0] = 2**26 + 8
    state.offset.fill_(-(2**26 + 16))
    # Rounded multiplication then addition is zero; FMA would be 2**-20.
    return packed, state


def test_reference_separate_scale_rounding_and_mutable_state():
    packed, state = _rounding_case()
    assert torch.count_nonzero(dequant.dequantize_nf4_reference(packed, state)) == 0
    state.offset.add_(16)
    assert torch.all(dequant.dequantize_nf4_reference(packed, state) == 16)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_cpu_reference_matches_installed_bitsandbytes(nested, transposed):
    bnb = pytest.importorskip("bitsandbytes")
    assert bnb.__version__ == "0.50.2", "reference contract targets the pinned version"
    original = getattr(bnb.functional.dequantize_4bit, dequant._PATCH_MARKER,
                       bnb.functional.dequantize_4bit)
    packed, state, _ = _case((129, 129), nested)
    if transposed:
        packed = packed.t()
    candidate_out = torch.empty(state.shape, dtype=torch.float16)
    # The installed CPU backend does not register the .out overload. Its
    # allocating operation remains an independent format/orientation oracle.
    expected = original(packed, state)
    actual = dequant.dequantize_nf4_reference(packed, state, out=candidate_out)
    _assert_bits(actual, expected)
    _assert_bits(candidate_out, expected.t() if transposed else expected)


@pytest.mark.parametrize("bad_out", ["dtype", "shape", "stride", "grad", "alias"])
def test_reference_rejects_unsupported_out(bad_out):
    packed, state, dense = _case()
    out = torch.empty_like(dense)
    if bad_out == "dtype":
        out = out.float()
    elif bad_out == "shape":
        out = out.t().contiguous()
    elif bad_out == "stride":
        out = torch.empty((65, 17), dtype=torch.float16).t()
    elif bad_out == "grad":
        out.requires_grad_()
    else:
        # Frozen quantization metadata must never be overwritten through out.
        base = torch.empty(dense.numel() * 2, dtype=torch.uint8)
        packed = base[:packed.numel()].view_as(packed)
        out = base.view(torch.float16).view_as(dense)
    with pytest.raises((NotImplementedError, ValueError)):
        dequant.dequantize_nf4_reference(packed, state, out=out)


def test_runtime_rejects_cpu_and_unsupported_states():
    packed, state, _ = _case()
    with pytest.raises(NotImplementedError, match="CUDA"):
        dequant.dequantize_nf4(packed, state)
    assert not dequant.supports_dequantize_nf4(packed, state)
    state.blocksize = 128
    with pytest.raises(NotImplementedError, match="blocksize"):
        dequant.dequantize_nf4_reference(packed, state)


@pytest.fixture
def patch_surfaces(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    calls = []
    sentinel = object()

    def original_public(*args, **kwargs):
        calls.append(("public", args, kwargs))
        return sentinel

    def original_fallback(*args, **kwargs):
        calls.append(("fallback", args, kwargs))
        return sentinel

    backend = SimpleNamespace(_dequant_linear_fallback=original_fallback,
                              _gemm_4bit_kernel_impl=object(),
                              _gemm_4bit_use_custom_fn=object(),
                              _dequantize_4bit_impl=object())
    monkeypatch.setattr(bnb.functional, "dequantize_4bit", original_public)
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda.ops", backend)
    monkeypatch.setattr(bnb, "__version__", "0.50.2")
    return bnb, backend, calls, sentinel


def test_patch_idempotence_and_cpu_fallback(monkeypatch, patch_surfaces):
    bnb, backend, calls, sentinel = patch_surfaces
    untouched = (backend._gemm_4bit_kernel_impl, backend._gemm_4bit_use_custom_fn,
                 backend._dequantize_4bit_impl)
    originals = (bnb.functional.dequantize_4bit, backend._dequant_linear_fallback)
    surfaces = dequant.patch_dequant()
    assert surfaces == ("bitsandbytes.functional.dequantize_4bit",
                        "bitsandbytes.backends.cuda.ops._dequant_linear_fallback")
    installed = (bnb.functional.dequantize_4bit, backend._dequant_linear_fallback)
    assert dequant.patch_dequant() == surfaces
    assert installed == (bnb.functional.dequantize_4bit, backend._dequant_linear_fallback)
    assert tuple(fn.__wrapped__ for fn in installed) == originals
    assert untouched == (backend._gemm_4bit_kernel_impl, backend._gemm_4bit_use_custom_fn,
                         backend._dequantize_4bit_impl)
    packed, state, _ = _case(nested=True)
    out = torch.empty(state.shape, dtype=torch.float16)
    assert bnb.functional.dequantize_4bit(packed, state, out=out) is sentinel
    assert calls[-1][0] == "public" and calls[-1][1][3] is out
    x = torch.zeros((3, state.shape[1]), dtype=torch.float16)
    assert backend._dequant_linear_fallback(
        x, packed, state.shape, state.state2.absmax, 64, "nf4",
        absmax_8bit=state.absmax, absmax_code=state.state2.code,
        absmax_offset=state.offset) is sentinel
    assert calls[-1][0] == "fallback"
    assert calls[-1][2]["absmax_8bit"] is state.absmax


def _use_reference_instead_of_cuda(monkeypatch):
    # Exercise the real validation/routing boundary on CPU, substituting only
    # device eligibility and execution. This is explicitly NOT a GPU test.
    monkeypatch.setattr(dequant, "_require_cuda", lambda packed: None)

    def decode(packed, state, out=None):
        dense = dequant._dense_reference(packed, state)
        return dense if out is None else out.copy_(dense)

    monkeypatch.setattr(dequant, "_dense_cuda", decode)


@pytest.mark.parametrize("nested", [False, True])
def test_public_patch_precedence_out_and_transpose(monkeypatch, patch_surfaces, nested):
    bnb, _, calls, _ = patch_surfaces
    _use_reference_instead_of_cuda(monkeypatch)
    dequant.patch_dequant()
    packed, state, expected = _case(nested=nested)
    out = torch.empty_like(expected)
    # The supplied NF4 state overrides default/contradictory legacy arguments.
    result = bnb.functional.dequantize_4bit(packed.t(), state, absmax=torch.zeros(1),
                                           out=out, blocksize=128, quant_type="fp4")
    _assert_bits(out, expected)
    _assert_bits(result, expected.t())
    assert result.data_ptr() == out.data_ptr()
    assert calls == []


@pytest.mark.parametrize("defect", ["dtype", "type", "block", "nested_block", "trainable", "out", "no_state"])
def test_public_patch_unsupported_preserves_original(monkeypatch, patch_surfaces, defect):
    bnb, _, calls, sentinel = patch_surfaces
    _use_reference_instead_of_cuda(monkeypatch)
    dequant.patch_dequant()
    packed, state, expected = _case(nested=True)
    out = None
    if defect == "dtype":
        state.dtype = torch.bfloat16
    elif defect == "type":
        state.quant_type = "fp4"
    elif defect == "block":
        state.blocksize = 128
    elif defect == "nested_block":
        state.state2.blocksize = 128
    elif defect == "trainable":
        state.offset.requires_grad_()
    elif defect == "out":
        out = expected.t()
    else:
        state = None
    assert bnb.functional.dequantize_4bit(packed, state, out=out) is sentinel
    assert len(calls) == 1 and calls[0][0] == "public"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("layout", ["column", "row"])
def test_forward_patch_dense_orientation_bias_and_gradients(monkeypatch, patch_surfaces, nested, layout):
    _, backend, calls, _ = patch_surfaces
    _use_reference_instead_of_cuda(monkeypatch)
    dequant.patch_dequant()
    packed, state, expected = _case(nested=nested)
    if layout == "row":
        packed = packed.t()
    x = torch.linspace(-0.5, 0.5, 3 * 65).half().reshape(3, 65).requires_grad_()
    bias = torch.linspace(-0.1, 0.1, 17).half().requires_grad_()
    result = backend._dequant_linear_fallback(
        x, packed, state.shape, state.state2.absmax if nested else state.absmax,
        64, "nf4", bias,
        absmax_8bit=state.absmax if nested else None,
        absmax_code=state.state2.code if nested else None,
        absmax_offset=state.offset)
    reference = torch.nn.functional.linear(x, expected, bias)
    _assert_bits(result, reference)
    upstream = torch.arange(result.numel()).reshape_as(result).half() / 64
    actual_grad = torch.autograd.grad(result, (x, bias), upstream)
    expected_grad = torch.autograd.grad(reference, (x, bias), upstream)
    for actual, expected in zip(actual_grad, expected_grad):
        _assert_bits(actual, expected)
    assert calls == []


def test_forward_patch_unsupported_and_kernel_errors(monkeypatch, patch_surfaces):
    bnb, backend, calls, sentinel = patch_surfaces
    _use_reference_instead_of_cuda(monkeypatch)
    dequant.patch_dequant()
    packed, state, _ = _case()
    x = torch.zeros((3, 65), dtype=torch.float32)
    assert backend._dequant_linear_fallback(x, packed, state.shape, state.absmax, 64, "nf4") is sentinel
    assert calls[-1][0] == "fallback"

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failure must remain visible")

    monkeypatch.setattr(dequant, "_dense_cuda", fail)
    with pytest.raises(RuntimeError, match="kernel failure"):
        bnb.functional.dequantize_4bit(packed, state)


def test_patch_pinned_version_and_no_cuda_backend(monkeypatch, patch_surfaces):
    bnb, _, _, _ = patch_surfaces
    original = bnb.functional.dequantize_4bit
    monkeypatch.setattr(bnb, "__version__", "0.50.1")
    assert dequant.patch_dequant() == ()
    assert bnb.functional.dequantize_4bit is original
    monkeypatch.setattr(bnb, "__version__", "0.50.2")
    monkeypatch.delitem(sys.modules, "bitsandbytes.backends.cuda.ops")
    assert dequant.patch_dequant() == ("bitsandbytes.functional.dequantize_4bit",)


def test_fake_tensors_are_never_eligible(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        packed, state, _ = _case()
    _use_reference_instead_of_cuda(monkeypatch)
    assert not dequant.supports_dequantize_nf4(packed, state)


requires_interpreter = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1",
                                        reason="opt-in CPU Triton interpreter only")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA hardware")
requires_variant_kernel = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="requires CUDA hardware or explicit CPU interpreter",
)


def _variant_tensors(packed, state):
    # An explicit interpreter run is the only CPU execution of the kernel.
    # The production entry still rejects CPU, as tested separately.
    if os.environ.get("TRITON_INTERPRET") == "1":
        return packed, state
    return packed.cuda(), _to_cuda(state)


@pytest.fixture
def unpatched_bnb(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    from bitsandbytes.backends.cuda import ops

    assert bnb.__version__ == "0.50.2", "acceptance targets the pinned reference version"
    for module, name in ((bnb.functional, "dequantize_4bit"), (ops, "_dequant_linear_fallback")):
        current = getattr(module, name)
        monkeypatch.setattr(module, name, getattr(current, dequant._PATCH_MARKER, current))
    return bnb


def _kernel_output(packed, state):
    out = torch.empty(state.shape, device=packed.device, dtype=torch.float16)
    dequant._launch(packed, state, out)
    return out


_EXPERIMENTAL_VARIANTS = ("tree", "paired", "grouped")


@pytest.mark.parametrize("variant", _EXPERIMENTAL_VARIANTS)
def test_experimental_entry_keeps_runtime_guards(variant):
    packed, state, _ = _case(nested=True)
    with pytest.raises(NotImplementedError, match="CUDA"):
        dequant._dequantize_nf4_variant(packed, state, variant=variant)
    state.blocksize = 128
    with pytest.raises(NotImplementedError, match="blocksize"):
        dequant._dequantize_nf4_variant(packed, state, variant=variant)


@pytest.mark.parametrize("options", [
    {"variant": "unknown"}, {"variant": "tree", "block_bytes": 300},
    {"variant": "paired", "num_warps": 3}, {"variant": "grouped", "block_bytes": True},
])
def test_experimental_invalid_config_fails_before_launch(options):
    packed, state, expected = _case()
    with pytest.raises(ValueError):
        dequant._launch_variant(packed, state, torch.empty_like(expected), **options)


def test_default_launch_still_uses_validated_baseline(monkeypatch):
    calls = []

    class Baseline:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))
            return launch

    def forbidden(*args, **kwargs):
        pytest.fail("the default must not select an experimental variant")

    monkeypatch.setattr(dequant, "_dequantize_nf4_kernel", Baseline())
    monkeypatch.setattr(dequant, "_launch_variant", forbidden)
    packed, state, expected = _case(nested=True)
    dequant._launch(packed, state, torch.empty_like(expected))
    assert len(calls) == 1
    grid, args, kwargs = calls[0]
    assert grid == (2,)
    assert args[6:] == (expected.numel(), True, 512, dequant.NF4_CODEBOOK)
    assert kwargs == {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


@requires_variant_kernel
@pytest.mark.parametrize("variant", _EXPERIMENTAL_VARIANTS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("out_offset", [15, 16])
@pytest.mark.parametrize("shape", [
    (1, 1), (3, 17), (17, 65), (129, 129),
    *[(1, n) for n in (63, 64, 65, 1023, 1024, 1025, 16383, 16384, 16385)],
])
def test_variants_exact_tails_and_canaries(variant, nested, shape, out_offset):
    packed, state, expected = _case(shape, nested)
    packed, state = _variant_tensors(packed, state)
    # Even contiguous views need not begin at the storage's base address.
    packed_storage = torch.full((packed.numel() + 5,), 255, dtype=torch.uint8, device=packed.device)
    packed_storage[3:-2].copy_(packed.reshape(-1))
    packed = packed_storage[3:-2].view_as(packed)
    backing = torch.full((expected.numel() + 32,), float("nan"), dtype=torch.float16, device=packed.device)
    out = backing[out_offset:out_offset + expected.numel()].reshape(shape)
    dequant._launch_variant(packed, state, out, variant=variant)
    _assert_bits(out, expected.to(out.device))
    assert torch.isnan(backing[:out_offset]).all() and torch.isnan(backing[out_offset + expected.numel():]).all()


@requires_variant_kernel
@pytest.mark.parametrize("variant", _EXPERIMENTAL_VARIANTS)
@pytest.mark.parametrize("block_bytes,num_warps", [
    (256, 4), (512, 4), (1024, 4), (256, 8), (512, 8), (1024, 8),
])
def test_variants_all_bytes_and_scale_mutations(variant, block_bytes, num_warps):
    # Every byte occurs within each 256-scale span; cross that boundary and
    # leave a partial final scale group and a partial final tile.
    packed, state, _ = _case((257, 65), nested=True)
    packed.copy_((torch.arange(packed.numel()) % 256).to(torch.uint8).reshape_as(packed))
    packed, state = _variant_tensors(packed, state)
    out = torch.empty(state.shape, dtype=torch.float16, device=packed.device)
    for mutate in (False, True):
        if mutate:
            state.absmax.add_(13)
            state.state2.code.neg_()
            state.state2.absmax.mul_(3)
            state.offset.add_(0.25)
        dequant._launch_variant(packed, state, out, variant=variant,
                                block_bytes=block_bytes, num_warps=num_warps)
        _assert_bits(out, dequant._dense_reference(packed, state))


@requires_variant_kernel
@pytest.mark.parametrize("variant", _EXPERIMENTAL_VARIANTS)
def test_variants_rounding_and_signed_zero(variant):
    packed, state = _rounding_case()
    packed, state = _variant_tensors(packed, state)
    out = torch.empty(state.shape, dtype=torch.float16, device=packed.device)
    dequant._launch_variant(packed, state, out, variant=variant)
    assert torch.count_nonzero(out) == 0
    packed, state, _ = _case((8, 64))
    state.absmax = torch.tensor([0.0, -0.0, 2**-24, 2**-14, 65504.0, 2**10, 1.00048828125, 1.234567])
    packed, state = _variant_tensors(packed, state)
    out = torch.empty(state.shape, dtype=torch.float16, device=packed.device)
    dequant._launch_variant(packed, state, out, variant=variant)
    _assert_bits(out, dequant._dense_reference(packed, state))


@requires_interpreter
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (3, 17), (17, 65), (129, 129)])
def test_interpreter_exact_decode(nested, shape):
    packed, state, expected = _case(shape, nested)
    _assert_bits(_kernel_output(packed, state), expected)


@requires_interpreter
def test_interpreter_scale_rounding():
    packed, state = _rounding_case()
    assert torch.count_nonzero(_kernel_output(packed, state)) == 0


@requires_interpreter
def test_interpreter_scale_extremes_and_signed_zero():
    packed, state, _ = _case((8, 64))
    state.absmax = torch.tensor([0.0, -0.0, 2**-24, 2**-14, 65504.0, 2**10, 1.00048828125, 1.234567])
    _assert_bits(_kernel_output(packed, state), dequant._dense_reference(packed, state))


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="offline compilation uses the JIT")
@pytest.mark.parametrize("nested", [False, True])
def test_offline_sm75_compile_preserves_rounding(nested):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(
        dequant._dequantize_nf4_kernel,
        {"Packed": "*u8", "Absmax": "*u8" if nested else "*fp32", "ScaleCode": "*fp32",
         "ScaleAbsmax": "*fp32", "Offset": "*fp32", "Out": "*fp16"},
        constexprs={"NUMEL": 16641, "NESTED": nested, "BLOCK_BYTES": dequant._BLOCK_BYTES,
                    "CODEBOOK": dequant.NF4_CODEBOOK},
    )
    compiled = triton.compile(source, target=GPUTarget("cuda", 75, 32),
                              options={"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False})
    ptx = compiled.asm["ptx"]
    assert "sm_75" in ptx and "mul.rn.f32" in ptx and "cvt.rn.f16.f32" in ptx
    assert "fma." not in ptx
    if nested:
        assert "add.rn.f32" in ptx


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="offline compilation uses the JIT")
@pytest.mark.parametrize("variant", _EXPERIMENTAL_VARIANTS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("pair_aligned", [False, True])
@pytest.mark.parametrize("block_bytes,num_warps", [
    (256, 4), (512, 4), (1024, 4), (256, 8), (512, 8), (1024, 8),
])
def test_offline_variants_sm75_preserve_rounding(variant, nested, block_bytes, num_warps, pair_aligned):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(
        dequant._dequantize_nf4_variant_kernel,
        {"Packed": "*u8", "Absmax": "*u8" if nested else "*fp32", "ScaleCode": "*fp32",
         "ScaleAbsmax": "*fp32", "Offset": "*fp32", "Out": "*fp16"},
        constexprs={"NUMEL": 16641, "NESTED": nested, "BLOCK_BYTES": block_bytes,
                    "CODEBOOK": dequant.NF4_CODEBOOK, "VARIANT": variant,
                    "PAIR_ALIGNED": pair_aligned},
    )
    compiled = triton.compile(source, target=GPUTarget("cuda", 75, 32),
                              options={"num_warps": num_warps, "num_stages": 1,
                                       "enable_fp_fusion": False})
    ptx = compiled.asm["ptx"]
    assert "sm_75" in ptx and "mul.rn.f32" in ptx and "cvt.rn.f16.f32" in ptx
    assert "fma." not in ptx
    if nested:
        assert "add.rn.f32" in ptx


@pytest.fixture(params=("baseline", *_EXPERIMENTAL_VARIANTS))
def decode_variant(request, monkeypatch):
    # Test-only selection: exercise the identical public entry and both patch
    # surfaces. This fixture never changes integration defaults outside pytest.
    if request.param != "baseline":
        monkeypatch.setattr(dequant, "_launch", functools.partial(
            dequant._launch_variant, variant=request.param))
    return request.param


@requires_cuda
@pytest.mark.parametrize("variant", _EXPERIMENTAL_VARIANTS)
def test_cuda_private_variant_entry_orientation_and_out(unpatched_bnb, variant):
    packed, state, _ = _case((17, 65), nested=True)
    packed, state = packed.cuda(), _to_cuda(state)
    expected = unpatched_bnb.functional.dequantize_4bit(packed, state)
    actual = dequant._dequantize_nf4_variant(packed, state, variant=variant)
    _assert_bits(actual, expected)
    backing = torch.full((expected.numel() + 2,), float("nan"), device="cuda", dtype=torch.float16)
    out = backing[1:-1].view_as(expected)
    actual = dequant._dequantize_nf4_variant(packed.t(), state, out=out, variant=variant)
    _assert_bits(out, expected)
    _assert_bits(actual, expected.t())
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isnan(backing[0]) and torch.isnan(backing[-1])


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [
    (1, 64), (3, 17), (17, 65), (129, 129), (256, 1536), (1536, 256),
    *[pytest.param(shape, marks=pytest.mark.skipif(
        os.environ.get("DEQUANT_FULL_GPU_TESTS") != "1", reason="opt-in full Qwen weight shapes"))
      for shape in [(1536, 1536), (8960, 1536), (1536, 8960)]],
])
def test_cuda_real_bitsandbytes_exact_decode(unpatched_bnb, decode_variant, nested, shape):
    functional = unpatched_bnb.functional
    gen = torch.Generator(device="cuda").manual_seed(271828)
    weight = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float16)
    packed, state = functional.quantize_4bit(weight, blocksize=64, quant_type="nf4", compress_statistics=nested)
    reference = functional.dequantize_4bit(packed, state)
    result = dequant.dequantize_nf4(packed, state)
    _assert_bits(result, reference)
    out = torch.empty_like(reference)
    returned = dequant.dequantize_nf4(packed.t(), state, out=out)
    _assert_bits(out, reference)
    _assert_bits(returned, reference.t())
    assert returned.data_ptr() == out.data_ptr()


@requires_cuda
def test_cuda_adversarial_rounding_and_metadata_mutation(decode_variant):
    packed, state = _rounding_case()
    packed, state = packed.cuda(), _to_cuda(state)
    assert torch.count_nonzero(dequant.dequantize_nf4(packed, state)) == 0
    state.offset.add_(16)
    assert torch.all(dequant.dequantize_nf4(packed, state) == 16)


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
def test_cuda_synthetic_all_nibbles_and_signed_zero(unpatched_bnb, decode_variant, nested):
    packed, state, _ = _case((8, 64), nested=nested)
    if not nested:
        state.absmax = torch.tensor([0.0, -0.0, 2**-24, 2**-14, 65504.0, 2**10, 1.00048828125, 1.234567])
    packed, state = packed.cuda(), _to_cuda(state)
    expected = unpatched_bnb.functional.dequantize_4bit(packed, state)
    _assert_bits(dequant.dequantize_nf4(packed, state), expected)


@requires_cuda
def test_cuda_out_on_nondefault_stream(unpatched_bnb, decode_variant):
    functional = unpatched_bnb.functional
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        weight = torch.arange(17 * 65, dtype=torch.float32, device="cuda").reshape(17, 65).sin().half()
        packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
        expected = functional.dequantize_4bit(packed, state)
        backing = torch.full((17 * 65 + 32,), float("nan"), dtype=torch.float16, device="cuda")
        out = backing[16:-16].reshape(17, 65)
        actual = dequant.dequantize_nf4(packed, state, out=out)
    stream.synchronize()
    _assert_bits(actual, expected)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isnan(backing[:16]).all() and torch.isnan(backing[-16:]).all()


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
def test_cuda_independent_downstream_adapter_gradients(unpatched_bnb, decode_variant, nested):
    functional = unpatched_bnb.functional
    gen = torch.Generator(device="cuda").manual_seed(12345)
    weight = torch.randn((33, 65), generator=gen, device="cuda", dtype=torch.float16) / 4
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=nested)
    baseline = functional.dequantize_4bit(packed, state)
    candidate = dequant.dequantize_nf4(packed, state)
    inputs = [torch.randn(shape, generator=gen, device="cuda", dtype=torch.float16) / 8
              for shape in [(7, 65), (4, 65), (33, 4), (33,)]]
    upstream = torch.randn((7, 33), generator=gen, device="cuda", dtype=torch.float16)

    def run(dense):
        x, down, up, bias = [v.detach().clone().requires_grad_(True) for v in inputs]
        # Ordinary cuBLAS and a separate low-rank branch, independent of bnb autograd.
        y = torch.nn.functional.linear(x, dense, bias) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up)
        y.backward(upstream)
        return y.detach(), [v.grad for v in (x, down, up, bias)]

    actual_y, actual_grads = run(candidate)
    expected_y, expected_grads = run(baseline)
    _assert_bits(actual_y, expected_y)
    for actual, expected in zip(actual_grads, expected_grads):
        _assert_bits(actual, expected)


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("rows", [4, 1537])
def test_cuda_patched_bnb_autograd_preserves_dispatch_and_gradients(
    monkeypatch, unpatched_bnb, decode_variant, nested, rows,
):
    bnb = unpatched_bnb
    from bitsandbytes.backends.cuda import ops

    assert bnb.__version__ == "0.50.2", "patch acceptance targets the pinned version"
    gen = torch.Generator(device="cuda").manual_seed(2048)
    weight = torch.randn((33, 64), device="cuda", dtype=torch.float16, generator=gen) / 4
    packed, state = bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=nested)
    data = torch.randn((rows, 64), device="cuda", dtype=torch.float16, generator=gen) / 8
    bias = torch.randn((33,), device="cuda", dtype=torch.float16, generator=gen) / 8
    upstream = torch.randn((rows, 33), device="cuda", dtype=torch.float16, generator=gen) / 8

    def run():
        x, b = data.clone().requires_grad_(), bias.clone().requires_grad_()
        y = bnb.matmul_4bit(x, packed, quant_state=state, bias=b)
        y.backward(upstream)
        return y.detach(), x.grad, b.grad

    expected = run()
    custom, heuristic = ops._gemm_4bit_kernel_impl, ops._gemm_4bit_use_custom_fn
    decode_calls = []
    launch = dequant._launch

    def counted(*args, **kwargs):
        decode_calls.append(1)
        return launch(*args, **kwargs)

    monkeypatch.setattr(dequant, "_launch", counted)
    assert len(dequant.patch_dequant()) == 2
    assert ops._gemm_4bit_kernel_impl is custom and ops._gemm_4bit_use_custom_fn is heuristic
    actual = run()
    for a, b in zip(actual, expected):
        _assert_bits(a, b)
    # M=4 keeps bnb's existing fused forward; M>1536 forces its dense fallback.
    assert len(decode_calls) == (1 if rows == 4 else 2)
