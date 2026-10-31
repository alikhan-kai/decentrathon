"""Unit tests for solution/nf4.py against bitsandbytes 0.50.2. No model download: random weights of the
Qwen2.5-1.5B layer shapes are quantized with the settings of task/task.yaml (nf4, double quantization,
blocksize 64, fp16 compute). Skipped without CUDA.

Tolerance: none. NF4 dequantization is a table lookup times a scale, rounded once to fp16, so the Triton
result must equal bitsandbytes' result bit for bit (torch.equal).
The speed test prints a table (run pytest with -s) and only fails if the kernel is 1.5x slower than bnb's.
"""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Qwen2.5-1.5B: hidden 1536, intermediate 8960, 2 kv heads x head_dim 128 = 256
QWEN_SHAPES = {"q_proj": (1536, 1536), "k_proj": (256, 1536), "o_proj": (1536, 1536),
               "gate_proj": (8960, 1536), "down_proj": (1536, 8960)}
ODD_SHAPES = {"odd_len": (37, 129), "tiny": (3, 5), "not_multiple_of_block": (1001, 1000)}


@pytest.fixture(autouse=True)
def unpatched():
    from solution import nf4

    nf4.unpatch()
    yield nf4
    nf4.unpatch()


def quantize(shape, seed=0):
    import bitsandbytes.functional as F

    torch.manual_seed(seed)
    weight = torch.randn(*shape, device="cuda", dtype=torch.float16)
    return F.quantize_4bit(weight, blocksize=64, compress_statistics=True, quant_type="nf4")


def decoded_absmax(state):
    import bitsandbytes.functional as F

    return F.dequantize_blockwise(state.absmax, state.state2) + state.offset


@pytest.mark.parametrize("name", list(QWEN_SHAPES) + list(ODD_SHAPES))
def test_kernel_equals_bitsandbytes(name, unpatched):
    import bitsandbytes.functional as F

    shape = {**QWEN_SHAPES, **ODD_SHAPES}[name]
    packed, state = quantize(shape)
    assert state.nested and state.blocksize == 64 and state.quant_type == "nf4"
    reference = F.dequantize_4bit(packed, state)
    out = torch.empty(shape, device="cuda", dtype=torch.float16)
    unpatched.dequantize_nf4(packed.contiguous(), decoded_absmax(state).float(), out)
    assert torch.equal(out, reference)


def test_nf4_code_table_equals_bitsandbytes():
    import bitsandbytes.functional as F

    from solution.nf4 import NF4_CODE

    assert torch.equal(torch.tensor(NF4_CODE, dtype=torch.float32), F.get_4bit_type("nf4", device="cpu"))


def test_patched_dequantize_and_matmul_forward_backward(unpatched):
    """After patch(), bitsandbytes' own entry points give the same numbers in forward and backward
    (M = 2048 rows is the training case: dequantize + F.linear in forward, dequantize in backward)."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as F

    packed, state = quantize((1536, 8960))
    torch.manual_seed(1)
    x = torch.randn(2048, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2048, 1536, device="cuda", dtype=torch.float16)

    ref_w = F.dequantize_4bit(packed, state)
    ref_out = bnb.matmul_4bit(x, packed.t(), quant_state=state)
    ref_out.backward(grad)
    ref_grad, x.grad = x.grad.clone(), None

    unpatched.patch()
    assert unpatched._patched
    assert torch.equal(F.dequantize_4bit(packed, state), ref_w)
    out = bnb.matmul_4bit(x, packed.t(), quant_state=state)
    out.backward(grad)
    assert torch.equal(out, ref_out)
    assert torch.equal(x.grad, ref_grad)


def test_patch_is_idempotent_and_falls_back(unpatched):
    import bitsandbytes.functional as F

    unpatched.patch()
    first = unpatched._original
    unpatched.patch()
    assert unpatched._original is first
    # fp4 and a different blocksize are not ours: bitsandbytes' kernel must still handle them
    torch.manual_seed(0)
    weight = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    for kwargs in ({"quant_type": "fp4", "blocksize": 64}, {"quant_type": "nf4", "blocksize": 128}):
        packed, state = F.quantize_4bit(weight, **kwargs)
        out = F.dequantize_4bit(packed, state)
        assert out.shape == weight.shape and torch.isfinite(out).all()


def _time(fn, iters=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return 1000 * start.elapsed_time(end) / iters  # microseconds


def test_speed_against_bitsandbytes(unpatched):
    """Prints microseconds per call; the kernel must not be much slower than bnb's CUDA kernel."""
    packed0, state0 = quantize((8, 64))
    out0 = torch.empty((8, 64), device="cuda", dtype=torch.float16)
    began = time.perf_counter()
    unpatched.dequantize_nf4(packed0, decoded_absmax(state0).float(), out0)  # first launch = compile
    torch.cuda.synchronize()
    print(f"\nfirst launch (Triton compile + load): {time.perf_counter() - began:.2f} s")
    print(f"{'shape':>14} {'bnb us':>9} {'triton us':>10} {'ratio':>7}")
    total_bnb = total_ours = 0.0
    for shape in [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)]:
        packed, state = quantize(shape)
        absmax = decoded_absmax(state).float()
        out = torch.empty(shape, device="cuda", dtype=torch.float16)
        ref = torch.empty_like(out)

        def bnb_call():
            torch.ops.bitsandbytes.dequantize_4bit.out(packed, absmax, 64, "nf4", shape, torch.float16, out=ref)

        def our_call():
            unpatched.dequantize_nf4(packed, absmax, out)

        bnb_us, our_us = _time(bnb_call), _time(our_call)
        assert torch.equal(out, ref)
        total_bnb, total_ours = total_bnb + bnb_us, total_ours + our_us
        print(f"{str(shape):>14} {bnb_us:9.1f} {our_us:10.1f} {our_us / bnb_us:7.2f}")
    assert total_ours <= 1.5 * total_bnb, (total_ours, total_bnb)
