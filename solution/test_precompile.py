"""solution/precompile.py: the precompiled specializations are exactly the ones the training launches use.

After compile_all(), each kernel is driven through its real wrapper with the training dtypes and the
Qwen2.5-1.5B geometry, forward and backward; a launch that does not find its variant compiles a new one,
so every kernel's variant count must stay the same. Skipped without CUDA.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

N = 300  # packed tokens, not a multiple of 16


def _kernels():
    from solution import head_loss, nf4, rmsnorm, rotary, swiglu

    return [nf4.nf4_dequant_kernel, rmsnorm.rmsnorm_fwd_kernel, rmsnorm.rmsnorm_bwd_kernel,
            head_loss.head_loss_kernel, rotary._triton_kernel(),
            swiglu.example_swiglu_fwd_kernel, swiglu.example_swiglu_bwd_kernel]


def _run_training_launches():
    from solution import head_loss, native_half, nf4, rmsnorm, rotary, swiglu
    from solution.precompile import HEAD_DIM, HEADS, HIDDEN, KV_HEADS, VOCAB

    dev = "cuda"
    n = HIDDEN * HIDDEN
    nf4.dequantize_nf4_nested(
        torch.randint(0, 256, (n // 2, 1), dtype=torch.uint8, device=dev),
        torch.randint(0, 256, (n // 64,), dtype=torch.uint8, device=dev),
        torch.rand(256, device=dev),
        torch.rand((n // 64 + 255) // 256, device=dev),
        torch.tensor(0.5, dtype=torch.float32, device=dev),
        torch.empty(HIDDEN, HIDDEN, dtype=torch.float16, device=dev),
    )

    x = torch.randn(1, N, HIDDEN, device=dev, requires_grad=True)
    rmsnorm.rmsnorm(x, torch.rand(HIDDEN, device=dev), 1e-6).sum().backward()

    hidden = torch.randn(N, HIDDEN, device=dev, requires_grad=True)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float16, device=dev) * 0.02
    target = torch.randint(0, VOCAB, (N,), device=dev)
    target[:7] = head_loss.IGNORE
    head_loss.fused_head_loss(hidden, weight, target, N - 7)[0].backward()
    del weight

    q = torch.randn(1, N, HEADS, HEAD_DIM, dtype=torch.float16, device=dev).transpose(1, 2).requires_grad_()
    k = torch.randn(1, N, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev).transpose(1, 2).requires_grad_()
    cos = torch.randn(1, N, HEAD_DIM, dtype=torch.float16, device=dev)
    sin = torch.randn(1, N, HEAD_DIM, dtype=torch.float16, device=dev)
    oq, ok = rotary.apply_rotary_pos_emb(q, k, cos, sin)
    (oq.sum() + ok.sum()).backward()

    gate = torch.randn(1, N, 8960, dtype=torch.float16, device=dev, requires_grad=True)
    up = torch.randn(1, N, 8960, dtype=torch.float16, device=dev, requires_grad=True)
    out = native_half._NativeHalfSwiGLU.apply(gate, up, swiglu)
    out.sum().backward()
    torch.cuda.synchronize()


def test_training_launches_find_precompiled_variants():
    from solution import precompile

    precompile.compile_all()
    assert not precompile._STATS["failed"], precompile._STATS["failed"]
    before = {precompile._kernel_name(k): precompile._variants(k) for k in _kernels()}
    _run_training_launches()
    after = {precompile._kernel_name(k): precompile._variants(k) for k in _kernels()}
    assert after == before, f"launches compiled new variants: {before} -> {after}"


def test_patch_idempotent_single_thread():
    from solution import precompile

    precompile.patch()
    first = precompile._thread
    precompile.patch()
    assert precompile._thread is first
    first.join()
