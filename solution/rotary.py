"""Experimental joint Q/K RoPE, disabled until explicitly patched and measured.

RoPE and kernel fusion are established techniques; this is an original Triton
implementation of Qwen2's split-half rotation, not a new rotation algorithm.
Only the rotation is replaced: trig generation, projections, attention and
dropout stay upstream. One launch covers Q and K without concatenating them.
Backward retains only frozen cos/sin, never Q/K or their rotated activations.
The reference graph is transformers 5.16.1's Qwen2 apply_rotary_pos_emb.

The candidate preserves eager product rounding (also in fp16 backward), reads
both trig halves, and disables FMA. CPU/interpreter checks and offline sm75
compilation do not establish GPU correctness or a performance improvement.
Importing this module does not patch anything. Add rotary to the team registry
only after target validation. The implementation has prior isolated T4 evidence;
integration with the feat/check4 baseline is measured separately.
"""

import functools
import importlib

import torch


def _eager_rope(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _plain_strided(x):
    return (
        type(x) in (torch.Tensor, torch.nn.Parameter)
        # Both vmap implementations expose logical metadata without directly
        # addressable storage. is_grads_batched uses the legacy wrapper, which
        # is_batchedtensor and the functorch-active flag do not detect.
        and not torch._C._functorch.is_batchedtensor(x)
        and not torch._C._functorch.is_legacy_batchedtensor(x)
        and x.layout == torch.strided
        and not x.has_names()
        and not x.is_conj()
        and not x.is_neg()
        and all(s >= 0 for s in x.stride())
        and 0 < x.numel() < 2**31
        # Offset arithmetic in the kernel is int32. The pointer already
        # includes storage_offset; gaps in any dimension still count here.
        and sum((n - 1) * s for n, s in zip(x.shape, x.stride())) < 2**31
    )


def _supports_triton(q, k, cos, sin, unsqueeze_dim=1):
    """A metadata-only guard; unsupported calls stay on the original graph."""
    tensors = (q, k, cos, sin)
    if (
        type(unsqueeze_dim) is not int or unsqueeze_dim != 1
        or torch.compiler.is_compiling() or torch.jit.is_tracing()
        or torch._C._are_functorch_transforms_active()
        or not all(_plain_strided(x) for x in tensors)
        or q.device.type != "cuda" or torch.version.hip is not None
        or any(x.device != q.device for x in tensors[1:])
        or q.dtype not in (torch.float32, torch.float16) or k.dtype != q.dtype
        or cos.dtype not in (torch.float32, torch.float16) or sin.dtype != cos.dtype
        # A half input with fp32 trig promotes the forward but rounds each
        # MulBackward result to half. Leave that mixed-output path upstream.
        or (q.dtype == torch.float16 and cos.dtype != torch.float16)
        or cos.requires_grad or sin.requires_grad
        or q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3
    ):
        return False
    b, _, t, d = q.shape
    if (
        k.shape[0] != b or k.shape[2:] != (t, d)
        or not 2 <= d <= 256 or d % 2
        or q.stride(-1) != 1 or k.stride(-1) != 1
        or cos.shape != sin.shape or cos.shape[0] not in (1, b)
        or cos.shape[1:] != (t, d)
    ):
        return False
    # Autograd.Function without a jvp must not consume forward-AD tangents.
    return all(torch.autograd.forward_ad.unpack_dual(x).tangent is None for x in tensors)


@functools.lru_cache(maxsize=1)
def _triton_kernel():
    # Import lazily: CPU use and unsupported calls need neither Triton nor a
    # CUDA runtime. Triton resolves language symbols through these globals.
    global triton, tl, _rope_rotate_block
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as exc:
        if exc.name == "triton":
            return None
        raise

    @triton.jit
    def _rope_rotate_block(  # lint: ok -- device helper called by ember_rope_joint
        X, C, S, Y, pid,
        N, T, H: tl.constexpr, D: tl.constexpr,
        XS, CS, SS,
        BACKWARD: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pair = pid * BLOCK + tl.arange(0, BLOCK)
        valid = pair < N
        row, column = pair // (D // 2), pair % (D // 2)
        head = row % H
        token = row // H % T
        batch = row // (H * T)
        x0 = batch * XS[0] + head * XS[1] + token * XS[2] + column * XS[3]
        c0 = batch * CS[0] + token * CS[1] + column * CS[2]
        s0 = batch * SS[0] + token * SS[1] + column * SS[2]
        a = tl.load(X + x0, valid, other=0).to(tl.float32)
        b = tl.load(X + x0 + (D // 2) * XS[3], valid, other=0).to(tl.float32)
        ca = tl.load(C + c0, valid, other=0).to(tl.float32)
        cb = tl.load(C + c0 + (D // 2) * CS[2], valid, other=0).to(tl.float32)
        sa = tl.load(S + s0, valid, other=0).to(tl.float32)
        sb = tl.load(S + s0 + (D // 2) * SS[2], valid, other=0).to(tl.float32)
        # PyTorch's independent mul operations round BEFORE their sum. This
        # cast also preserves half product overflow/subnormal behavior. FMA
        # is disabled at launch, including for fp32 input/output.
        direct_a = (a * ca).to(Y.dtype.element_ty).to(tl.float32)
        direct_b = (b * cb).to(Y.dtype.element_ty).to(tl.float32)
        if BACKWARD:
            cross_a = (b * sb).to(Y.dtype.element_ty).to(tl.float32)
            cross_b = (a * sa).to(Y.dtype.element_ty).to(tl.float32)
            out_a, out_b = direct_a + cross_a, direct_b + (-cross_b)
        else:
            cross_a = ((-b) * sa).to(Y.dtype.element_ty).to(tl.float32)
            cross_b = (a * sb).to(Y.dtype.element_ty).to(tl.float32)
            out_a, out_b = direct_a + cross_a, direct_b + cross_b
        # Outputs use dense B,T,H,D storage viewed as B,H,T,D. Input and
        # incoming-gradient strides are independent of this allocation.
        y0 = row * D + column
        tl.store(Y + y0, out_a, valid)
        tl.store(Y + y0 + D // 2, out_b, valid)

    # Padding lengths, batch sizes, missing gradients and tensor layouts vary
    # between calls. Removing constexpr alone still lets Triton specialize
    # integers equal to 1 and their divisibility. Triton 3.6 does not propagate
    # do_not_specialize into tuple elements, so pass strides as separate scalar
    # arguments. Model geometry and arithmetic stay specialized.
    @triton.jit(do_not_specialize=[
        "NQ", "NK", "Q_BLOCKS", "T",
        "QS0", "QS1", "QS2", "QS3", "KS0", "KS1", "KS2", "KS3",
        "CS0", "CS1", "CS2", "SS0", "SS1", "SS2",
    ])
    def ember_rope_joint(  # lint: ok -- returned by _triton_kernel and launched via kernel
        Q, K, C, S, OQ, OK,
        NQ, NK, Q_BLOCKS,
        T, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
        QS0, QS1, QS2, QS3, KS0, KS1, KS2, KS3,
        CS0, CS1, CS2, SS0, SS1, SS2,
        BACKWARD: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        QS, KS = (QS0, QS1, QS2, QS3), (KS0, KS1, KS2, KS3)
        CS, SS = (CS0, CS1, CS2), (SS0, SS1, SS2)
        # Uniform branch; K receives only the blocks its smaller head count
        # needs. Missing Q/K gradients have zero blocks and no allocation.
        if pid < Q_BLOCKS:
            _rope_rotate_block(Q, C, S, OQ, pid, NQ, T, HQ, D, QS, CS, SS, BACKWARD, BLOCK)
        else:
            _rope_rotate_block(K, C, S, OK, pid - Q_BLOCKS, NK, T, HK, D, KS, CS, SS, BACKWARD, BLOCK)

    return ember_rope_joint


def _launch(q, k, cos, sin, q_shape, k_shape, *, backward=False):
    def allocate(x, shape):
        if x is None:
            return None
        b, h, t, d = shape
        return torch.empty((b, t, h, d), dtype=x.dtype, device=x.device).transpose(1, 2)

    oq, ok = allocate(q, q_shape), allocate(k, k_shape)
    nq = 0 if q is None else q.numel() // 2
    nk = 0 if k is None else k.numel() // 2
    if nq + nk == 0:
        return oq, ok
    block = 256
    q_blocks = (nq + block - 1) // block
    k_blocks = (nk + block - 1) // block
    cs, ss = list(cos.stride()), list(sin.stride())
    if cos.shape[0] == 1:
        cs[0] = ss[0] = 0
    # Absent branches receive harmless placeholders; their stores are masked
    # out and no blocks are scheduled for them. No Q/K tensor is retained.
    with torch.cuda.device(cos.device):
        _triton_kernel()[(q_blocks + k_blocks,)](
            q if q is not None else cos, k if k is not None else cos, cos, sin,
            oq if oq is not None else cos, ok if ok is not None else cos,
            nq, nk, q_blocks, q_shape[2], q_shape[1], k_shape[1], q_shape[3],
            *(q.stride() if q is not None else (0, 0, 0, 0)),
            *(k.stride() if k is not None else (0, 0, 0, 0)),
            *cs, *ss, backward, block,
            num_warps=4, enable_fp_fusion=False,
        )
    return oq, ok


def _eager_backward(g, cos, sin):
    if g is None:
        return None
    direct = g * cos.unsqueeze(1)
    rotated = g * sin.unsqueeze(1)
    half = g.shape[-1] // 2
    return direct + torch.cat((rotated[..., half:], -rotated[..., :half]), dim=-1)


class _FrozenRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.set_materialize_grads(False)
        ctx.q_shape, ctx.k_shape = tuple(q.shape), tuple(k.shape)
        ctx.save_for_backward(cos, sin)
        oq, ok = _launch(q, k, cos, sin, ctx.q_shape, ctx.k_shape)
        frozen = tuple(output for output, needed in zip((oq, ok), ctx.needs_input_grad[:2]) if not needed)
        ctx.mark_non_differentiable(*frozen)
        return oq, ok

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        grad_q = grad_q if ctx.needs_input_grad[0] else None
        grad_k = grad_k if ctx.needs_input_grad[1] else None
        gradients = (grad_q, grad_k)
        if torch.is_grad_enabled() or any(
            x is not None and (
                not _plain_strided(x)
                or torch.autograd.forward_ad.unpack_dual(x).tangent is not None
            ) for x in gradients
        ):
            # The transpose is linear and depends only on trig and incoming
            # gradients. This preserves higher derivatives without saving Q/K.
            return _eager_backward(grad_q, cos, sin), _eager_backward(grad_k, cos, sin), None, None
        dq, dk = _launch(grad_q, grad_k, cos, sin, ctx.q_shape, ctx.k_shape, backward=True)
        return dq, dk, None, None


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Qwen2-compatible rotation with a conservative, lazy CUDA fast path.

    Supported: B,H,T,D Q/K with even 2<=D<=256, matching B/T/D and dtype,
    contiguous last dimensions (outer strides may differ), frozen B,T,D trig
    with batch B or 1 and arbitrary nonnegative strides, unsqueeze_dim=1.
    Dtypes: fp32 Q/K with fp32/fp16 trig, or fp16 throughout. Other calls use
    eager PyTorch. The patch helper instead calls the saved upstream function
    on fallback, preserving upstream behavior and optional wrappers.
    """
    if _supports_triton(q, k, cos, sin, unsqueeze_dim) and _triton_kernel() is not None:
        return _FrozenRoPE.apply(q, k, cos, sin)
    return _eager_rope(q, k, cos, sin, unsqueeze_dim)


ENABLED = False


def patch():
    """Registry adapter for the existing runtime-length RoPE candidate."""
    return patch_rope()


def patch_rope():
    """Idempotently patch only Qwen2.apply_rotary_pos_emb; False if absent.

    Does not generate trig, replace projections, or enable any Soup switches.
    The saved original is available as ``_soup_rope_original`` for restoration.
    """
    try:
        module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    except ImportError:
        return False
    original = getattr(module, "apply_rotary_pos_emb", None)
    if not callable(original):
        return False
    if getattr(original, "_soup_rope_original", None) is not None:
        return True

    @functools.wraps(original)
    def forward(q, k, cos, sin, unsqueeze_dim=1):
        if _supports_triton(q, k, cos, sin, unsqueeze_dim) and _triton_kernel() is not None:
            return _FrozenRoPE.apply(q, k, cos, sin)
        return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    forward._soup_rope_original = original
    module.apply_rotary_pos_emb = forward
    return True
