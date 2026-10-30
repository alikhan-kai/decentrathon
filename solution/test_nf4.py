"""NF4 format checks on CPU and optional numerical checks on real CUDA hardware.

Run: python -m pytest solution/test_nf4.py -q
CPU kernel interpreter: TRITON_INTERPRET=1 python -m pytest solution/test_nf4.py -q -k interpreter
Full Qwen shapes: NF4_FULL_GPU_TESTS=1 python -m pytest solution/test_nf4.py -q
The environment switch belongs only to this test suite, never the implementation.
"""

import os
import struct
from types import SimpleNamespace

import pytest
import torch

from solution.nf4 import (
    NF4_CODEBOOK,
    dequantize_nf4_reference,
    nf4_input_grad,
    nf4_input_grad_reference,
)


# Independent format values: IEEE-754 binary32 words in nibble-index order.
_NF4_BITS = (
    0xBF800000, 0xBF3239B1, 0xBF066B30, 0xBECA32A0,
    0xBE91A24D, 0xBE3D353F, 0xBDBA7871, 0x00000000,
    0x3DA2FAFF, 0x3E24CAE3, 0x3E7C04DD, 0x3EAD033A,
    0x3EE1A4B8, 0x3F1007AB, 0x3F3913B3, 0x3F800000,
)


def _format_values():
    return torch.tensor(
        [struct.unpack("!f", struct.pack("!I", word))[0] for word in _NF4_BITS],
        dtype=torch.float32,
    )


def _pack(indices):
    flat = indices.to(torch.uint8).flatten()
    if flat.numel() % 2:
        # Deliberately nonzero padding nibble: the decoder must ignore it.
        flat = torch.cat((flat, torch.tensor([15], dtype=torch.uint8)))
    return ((flat[0::2] << 4) | flat[1::2]).reshape(-1, 1)


def _synthetic(shape=(17, 65), nested=False):
    size = shape[0] * shape[1]
    groups = (size + 63) // 64
    indices = (torch.arange(size) * 13 + 3) % 16
    state = SimpleNamespace(
        shape=torch.Size(shape), dtype=torch.float16, quant_type="nf4",
        blocksize=64, code=_format_values(), nested=nested,
        state2=None, offset=None,
    )
    if nested:
        # Exact binary fractions make the hand-built scale oracle unambiguous.
        state.absmax = ((torch.arange(groups) * 37 + 17) % 256).to(torch.uint8)
        state.state2 = SimpleNamespace(
            blocksize=256, dtype=torch.float32, nested=False, state2=None,
            code=torch.arange(256, dtype=torch.float32) / 128 - 1,
            absmax=(torch.arange((groups + 255) // 256, dtype=torch.float32) + 1) / 8,
        )
        state.offset = torch.tensor(1.25, dtype=torch.float32)
        scales = torch.tensor([
            (int(state.absmax[i]) / 128 - 1) * ((i // 256 + 1) / 8) + 1.25
            for i in range(groups)
        ], dtype=torch.float32)
    else:
        scales = (torch.arange(groups, dtype=torch.float32) % 19) / 8
        state.absmax = scales.clone()
    expected = (_format_values()[indices] * scales.repeat_interleave(64)[:size])
    return _pack(indices), state, expected.half().reshape(shape)


def _on_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{
            key: _on_device(item, device) for key, item in vars(value).items()
        })
    return value


def test_codebook_binary32_values_are_exact():
    actual = tuple(struct.unpack("!I", struct.pack("!f", x))[0] for x in NF4_CODEBOOK)
    assert actual == _NF4_BITS


@pytest.mark.parametrize("layout", ["flat", "column", "row"])
def test_decode_all_bytes_high_nibble_first(layout):
    packed, state, _ = _synthetic((16, 32))
    packed = torch.arange(256, dtype=torch.uint8)
    if layout == "column":
        packed = packed[:, None]
    elif layout == "row":
        packed = packed[None, :]
    state.absmax.fill_(1)
    pairs = [(byte // 16, byte % 16) for byte in range(256)]
    expected = _format_values()[torch.tensor(pairs)].reshape(16, 32).half()
    torch.testing.assert_close(dequantize_nf4_reference(packed, state), expected, rtol=0, atol=0)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [
    (1, 1), (3, 17), (5, 33), (17, 65), (129, 129), (256, 1536), (1536, 256),
])
def test_decode_flat_groups_row_boundaries_and_tails(shape, nested):
    packed, state, expected = _synthetic(shape, nested)
    result = dequantize_nf4_reference(packed, state)
    assert result.shape == shape and result.dtype == torch.float16
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if expected.numel() % 2:
        packed[-1] &= 0xF0
        torch.testing.assert_close(dequantize_nf4_reference(packed, state), expected, rtol=0, atol=0)


def _rounding_case():
    packed, state, _ = _synthetic((1, 1), nested=True)
    packed.fill_(0xF0)
    state.absmax.zero_()
    state.state2.code[0] = 1 + 2**-23
    state.state2.absmax[0] = 2**26 + 8
    state.offset.fill_(-(2**26 + 16))
    # Separate fp32 multiply/add yields 0; fused arithmetic yields 2**-20.
    return packed, state


def test_nested_scale_rounding_matches_separate_fp32_operations():
    packed, state = _rounding_case()
    assert dequantize_nf4_reference(packed, state).item() == 0


def test_reference_does_not_cache_mutable_scales():
    packed, state, expected = _synthetic(nested=True)
    torch.testing.assert_close(dequantize_nf4_reference(packed, state), expected, rtol=0, atol=0)
    state.offset.add_(1)
    result = dequantize_nf4_reference(packed, state)
    assert not torch.equal(result, expected)
    state.absmax.zero_()
    assert not torch.equal(result, dequantize_nf4_reference(packed, state))


@pytest.mark.parametrize("leading", [(), (2,), (2, 3), (0, 3)])
@pytest.mark.parametrize("nested", [False, True])
def test_input_grad_reference_shape_dtype_and_orientation(leading, nested):
    packed, state, weight = _synthetic((19, 35), nested)
    dy = torch.linspace(-0.5, 0.5, torch.Size((*leading, 19)).numel()).half().reshape(*leading, 19)
    expected = (dy.float() @ weight.float()).half()
    result = nf4_input_grad_reference(dy, packed, state)
    assert result.shape == (*leading, 35) and result.dtype == dy.dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("attribute,value,message", [
    ("quant_type", "fp4", "nf4"),
    ("blocksize", 128, "blocksize=64"),
    ("dtype", torch.bfloat16, "float16"),
    ("shape", (17, 65, 1), "shape"),
    ("shape", (0, 65), "positive"),
    ("shape", (17, 64), "packed_weight"),
    ("packing_format_for_cpu", True, "packing_format_for_cpu"),
    ("absmax", torch.ones(1), "absmax"),
    ("absmax", torch.ones(18, dtype=torch.float16), "absmax"),
    ("nested", True, "state2"),
])
def test_reject_unsupported_or_inconsistent_states(attribute, value, message):
    packed, state, _ = _synthetic()
    setattr(state, attribute, value)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        dequantize_nf4_reference(packed, state)


@pytest.mark.parametrize("attribute,value,message", [
    ("blocksize", 128, "state2.blocksize=256"),
    ("dtype", torch.float16, "state2.dtype"),
    ("nested", True, "single level"),
    ("code", torch.zeros(255), "state2.code"),
    ("absmax", torch.zeros(2), "state2.absmax"),
])
def test_reject_unsupported_nested_states(attribute, value, message):
    packed, state, _ = _synthetic(nested=True)
    setattr(state.state2, attribute, value)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        dequantize_nf4_reference(packed, state)


def test_reject_noncanonical_packing_and_trainable_scales():
    packed, state, _ = _synthetic((16, 32))
    with pytest.raises(NotImplementedError, match="uint8"):
        dequantize_nf4_reference(packed.float(), state)
    with pytest.raises(NotImplementedError, match="packed_weight"):
        dequantize_nf4_reference(packed.reshape(16, 16), state)
    strided = torch.empty(packed.numel() * 2, dtype=torch.uint8)[::2]
    strided.copy_(packed.flatten())
    with pytest.raises(NotImplementedError, match="contiguous"):
        dequantize_nf4_reference(strided, state)
    state.absmax.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="frozen"):
        dequantize_nf4_reference(packed, state)


def test_runtime_is_explicitly_cuda_only():
    packed, state, _ = _synthetic()
    with pytest.raises(NotImplementedError, match="CUDA"):
        nf4_input_grad(torch.zeros((2, 17), dtype=torch.float16), packed, state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runtime_rejects_other_gradient_dtypes(dtype):
    packed, state, _ = _synthetic()
    with pytest.raises(NotImplementedError, match="float16"):
        nf4_input_grad(torch.zeros((2, 17), dtype=dtype), packed, state)


def test_reference_rejects_wrong_gradient_orientation():
    packed, state, _ = _synthetic()
    with pytest.raises(ValueError, match="last dimension"):
        nf4_input_grad_reference(torch.zeros((2, 65), dtype=torch.float16), packed, state)


requires_interpreter = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="set TRITON_INTERPRET=1 to execute Triton semantics on CPU (not GPU validation)",
)


@requires_interpreter
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("m,n,k", [
    (1, 1, 1), (7, 19, 35), (33, 65, 129), (2, 129, 129), (3, 256, 1536), (3, 1536, 256),
])
def test_cpu_interpreter_original_kernel(nested, m, n, k):
    pytest.importorskip("triton")
    from solution.nf4 import _decode_nested_scales, _nf4_input_grad_kernel, _scales_reference

    packed, state, weight = _synthetic((n, k), nested)
    scales = state.absmax
    if nested:
        groups = scales.numel()
        scales = torch.empty(groups, dtype=torch.float32)
        _decode_nested_scales[((groups + 255) // 256,)](
            state.absmax, state.state2.code, state.state2.absmax, state.offset,
            scales, groups, 256, num_warps=4, enable_fp_fusion=False,
        )
        torch.testing.assert_close(scales, _scales_reference(state), rtol=0, atol=0)
    gen = torch.Generator().manual_seed(45)
    dy = (torch.randn((m, n), generator=gen) / 8).half()
    out = torch.empty((m, k), dtype=torch.float16)
    _nf4_input_grad_kernel[(((m + 31) // 32) * ((k + 63) // 64),)](
        dy, packed, scales, out, m, n, k, 32, 64, 32, NF4_CODEBOOK,
        num_warps=4, num_stages=2,
    )
    torch.testing.assert_close(out, (dy.float() @ weight.float()).half(), rtol=2e-3, atol=2e-3)


@requires_interpreter
def test_cpu_interpreter_nested_rounding():
    pytest.importorskip("triton")
    from solution.nf4 import _decode_nested_scales

    _, state = _rounding_case()
    scales = torch.empty(1, dtype=torch.float32)
    _decode_nested_scales[(1,)](
        state.absmax, state.state2.code, state.state2.absmax, state.offset,
        scales, 1, 256, num_warps=4, enable_fp_fusion=False,
    )
    assert scales.item() == 0


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an actual CUDA GPU")


@pytest.fixture
def cuda_runtime():
    pytest.importorskip("triton")
    # The numerical oracle must not silently use reduced precision matmul.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_fp16


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (7, 19, 35), (33, 65, 129), (2, 129, 129), (3, 256, 1536), (3, 1536, 256)])
def test_cuda_synthetic_gemm_and_tails(cuda_runtime, nested, m, n, k):
    packed, state, weight = _synthetic((n, k), nested)
    gen = torch.Generator().manual_seed(913)
    dy = (torch.randn((m, n), generator=gen) / 8).half().cuda()
    result = nf4_input_grad(dy, packed.cuda(), _on_device(state, "cuda"))
    expected = (dy.float() @ weight.cuda().float()).half()
    assert result.dtype == dy.dtype and result.shape == (m, k)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
def test_cuda_identity_recovers_exact_dequantized_values(cuda_runtime, nested):
    packed, state, weight = _synthetic((33, 65), nested)
    result = nf4_input_grad(torch.eye(33, device="cuda", dtype=torch.float16), packed.cuda(), _on_device(state, "cuda"))
    torch.testing.assert_close(result, weight.cuda(), rtol=0, atol=0)


@requires_cuda
def test_cuda_nested_scale_rounding(cuda_runtime):
    packed, state = _rounding_case()
    result = nf4_input_grad(torch.ones((1, 1), device="cuda", dtype=torch.float16), packed.cuda(), _on_device(state, "cuda"))
    assert result.item() == 0


@requires_cuda
def test_cuda_shape_noncontiguous_empty_and_mutated_scales(cuda_runtime):
    packed, state, _ = _synthetic((19, 35), nested=True)
    packed, state = packed.cuda(), _on_device(state, "cuda")
    dy = torch.arange(2 * 19 * 3, device="cuda", dtype=torch.float16).reshape(2, 19, 3).transpose(1, 2) / 128
    for _ in range(2):
        result = nf4_input_grad(dy, packed, state)
        torch.testing.assert_close(result, nf4_input_grad_reference(dy, packed, state), rtol=2e-3, atol=2e-3)
        assert result.shape == (2, 3, 35)
        state.offset.add_(0.5)
    vector = dy[0, 0]
    assert nf4_input_grad(vector, packed, state).shape == (35,)
    empty = torch.empty((0, 3, 19), device="cuda", dtype=torch.float16)
    assert nf4_input_grad(empty, packed, state).shape == (0, 3, 35)
    with pytest.raises(NotImplementedError, match="higher-order"):
        nf4_input_grad(dy.requires_grad_(True), packed, state)


def _check_bnb_case(shape, leading, nested):
    functional = pytest.importorskip("bitsandbytes.functional")
    gen = torch.Generator(device="cuda").manual_seed(31415)
    weight = (torch.randn(shape, device="cuda", dtype=torch.float16, generator=gen) / 8)
    packed, state = functional.quantize_4bit(
        weight, blocksize=64, quant_type="nf4", compress_statistics=nested,
    )
    dense = functional.dequantize_4bit(packed, state)
    torch.testing.assert_close(dequantize_nf4_reference(packed, state), dense, rtol=0, atol=0)
    dy = torch.randn((*leading, shape[0]), device="cuda", dtype=torch.float16, generator=gen) / 8
    actual = nf4_input_grad(dy, packed, state)
    baseline = torch.matmul(dy, dense)
    oracle = torch.matmul(dy.float(), dense.float()).half()
    assert actual.shape == (*leading, shape[1]) and actual.dtype == dy.dtype
    torch.testing.assert_close(actual, baseline, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [(17, 65), (129, 129), (256, 1536), (1536, 256)])
def test_cuda_real_bitsandbytes_quantization(cuda_runtime, shape, nested):
    _check_bnb_case(shape, (2, 3), nested)


@requires_cuda
@pytest.mark.skipif(os.environ.get("NF4_FULL_GPU_TESTS") != "1", reason="set NF4_FULL_GPU_TESTS=1 for full Qwen shapes")
@pytest.mark.parametrize("shape", [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)])
def test_cuda_full_qwen_shapes(cuda_runtime, shape):
    _check_bnb_case(shape, (4, 512), nested=True)
