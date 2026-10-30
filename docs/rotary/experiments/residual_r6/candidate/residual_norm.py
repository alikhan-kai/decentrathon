"""Optional first-residual + post-attention RMSNorm fusion for Qwen2 5.16.1.

Original fusion code built on the team's ``solution/rmsnorm.py`` foundation;
RMSNorm and residual fusion are established techniques, not a new algorithm.
Both outputs are live: s = residual.float() + attention.float(), y = w *
(s * rsqrt(mean(s*s) + eps)). Save s, frozen w and inv, never the two inputs.
Backward sums the residual and normalization branches in fp32 BEFORE its sole
cast to attention dtype. In particular, scaled gradients must not be rounded
to half, clamped or unscaled separately.

Candidate remains disabled and outside MODULES. Only the first residual add
and post_attention_layernorm call in each decoder can be fused: expected 28
forward + 28 backward launches in the task, replacing 28 post-norm launches
in each direction, not all 57 norms. Input/final norms and final adds stay put.
CPU interpretation/offline compilation do not establish T4 correctness or speed.
"""

import functools
import hashlib
import importlib
import inspect
import math
import sys
import types

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
import triton
import triton.language as tl

ENABLED = False
MAX_HIDDEN = 8192
NUM_WARPS = 4
_patched = False
_original = None
_replacement = None
_decoder_class = None
_norm_class = None
_norm_original = None

# Exact upstream source fingerprints, installed transformers==5.16.1. Unknown
# wrappers/source versions stay upstream rather than losing their semantics.
_DECODER_SHA256 = "75d1d62e173e8c574b8a61338d1fa5052774d13b848eecd282cde88cd6690b8e"
_NORM_SHA256 = "790fa612e4fa57acfcbea31da45f92ba91f9016f7260ef539f61c5530ba06a5f"


@triton.jit(do_not_specialize=["EPS"])
def residual_norm_fwd_kernel(R, A, W, S, Y, INV, H: tl.constexpr, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < H
    offset = row * H + col
    r = tl.load(R + offset, mask, other=0).to(tl.float32)
    a = tl.load(A + offset, mask, other=0).to(tl.float32)
    s = r + a
    inv = tl.rsqrt(tl.sum(s * s, 0) / H + EPS)
    w = tl.load(W + col, mask, other=0).to(tl.float32)
    # Separate fp32 add, square, normalize and weight multiply, with FMA off.
    y = w * (s * inv)
    tl.store(S + offset, s, mask)
    tl.store(Y + offset, y, mask)
    tl.store(INV + row, inv)


@triton.jit(do_not_specialize=["GS0", "GS1", "GY0", "GY1", "HAS_GS", "HAS_GY"])
def residual_norm_bwd_kernel(S, W, INV, GS, GY, DR, DA,
                             H: tl.constexpr, GS0, GS1, GY0, GY1, HAS_GS, HAS_GY,
                             NEED_R: tl.constexpr, NEED_A: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < H
    offset = row * H + col
    total = tl.full((BLOCK,), 0, tl.float32)
    if HAS_GY:
        s = tl.load(S + offset, mask, other=0).to(tl.float32)
        w = tl.load(W + col, mask, other=0).to(tl.float32)
        inv = tl.load(INV + row)
        gy = tl.load(GY + row * GY0 + col * GY1, mask, other=0).to(tl.float32)
        xhat = s * inv
        wdy = gy * w
        c = tl.sum(wdy * xhat, 0) / H
        total = (wdy - xhat * c) * inv
    if HAS_GS:
        gs = tl.load(GS + row * GS0 + col * GS1, mask, other=0).to(tl.float32)
        total = gs + total
    if NEED_R:
        tl.store(DR + offset, total, mask)
    if NEED_A:
        # This store is the ONLY half cast, after the branch sum (GradScaler).
        tl.store(DA + offset, total.to(DA.dtype.element_ty), mask)


def _eager(residual, attention, weight, eps):
    # Preserve normal PyTorch promotion and Qwen's cast-before-weight boundary
    # for every unsupported dtype/layout, broadcasting and trainable weight.
    summed = residual + attention
    value = summed.to(torch.float32)
    inv = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    normalized = weight * (value * inv).to(summed.dtype)
    return summed, normalized


def _ordinary_mode():
    return not (torch.compiler.is_compiling() or torch.jit.is_tracing()
                or torch.jit.is_scripting() or torch._C._are_functorch_transforms_active()
                or get_proxy_mode() is not None
                or torch.is_inference_mode_enabled())


def _plain(x):
    return (type(x) in (torch.Tensor, torch.nn.Parameter) and x.layout == torch.strided
            and not torch._C._functorch.is_batchedtensor(x)
            and not torch._C._functorch.is_legacy_batchedtensor(x)
            and not torch._is_functional_tensor(x) and not x.has_names()
            and not x.is_meta and not x.is_conj() and not x.is_neg()
            and torch.autograd.forward_ad.unpack_dual(x).tangent is None)


def _cuda_supported(x):
    return x.device.type == "cuda" and torch.version.hip is None


def _supported_residual_weight(residual, weight, eps):
    return (_ordinary_mode() and _plain(residual) and _plain(weight)
            and _cuda_supported(residual) and weight.device == residual.device
            and residual.dtype == weight.dtype == torch.float32
            and residual.ndim >= 1 and weight.ndim == 1
            and 0 < residual.numel() < 2**31
            and residual.is_contiguous() and weight.is_contiguous()
            and residual.shape[-1] == weight.numel() <= MAX_HIDDEN
            and not weight.requires_grad
            and type(eps) in (int, float) and math.isfinite(eps) and eps > 0)


def _supports_triton(residual, attention, weight, eps):
    return (_supported_residual_weight(residual, weight, eps) and _plain(attention)
            and attention.device == residual.device and attention.shape == residual.shape
            and attention.dtype in (torch.float16, torch.float32) and attention.is_contiguous())


def _launch_forward(residual, attention, weight, eps):
    hidden = residual.shape[-1]
    rows = residual.numel() // hidden
    summed, normalized = torch.empty_like(residual), torch.empty_like(residual)
    inv = torch.empty(rows, device=residual.device, dtype=torch.float32)
    # The existing guards establish contiguous rows and 0 < hidden <= MAX_HIDDEN.
    # Specialize only hidden so row alignment is derived, never asserted: odd
    # widths and offset pointers stay safe. Rows are grid-only; epsilon, incoming
    # strides and missing-gradient flags remain dynamic (including heuristics).
    with torch.cuda.device(residual.device):
        residual_norm_fwd_kernel[(rows,)](
            residual, attention, weight, summed, normalized, inv, hidden, eps,
            BLOCK=triton.next_power_of_2(hidden), num_warps=NUM_WARPS, enable_fp_fusion=False)
    return summed, normalized, inv


def _launch_backward(summed, weight, inv, grad_s, grad_y, need_r, need_a, attention_dtype):
    hidden = summed.shape[-1]
    # reshape may materialize irregular outer dimensions. Inner and broadcast
    # strides need no copy and are runtime scalars, never constexpr tuples.
    gs = grad_s.reshape(-1, hidden) if grad_s is not None else summed.reshape(-1, hidden)
    gy = grad_y.reshape(-1, hidden) if grad_y is not None else summed.reshape(-1, hidden)
    dr = torch.empty_like(summed) if need_r else None
    da = torch.empty_like(summed, dtype=attention_dtype) if need_a else None
    with torch.cuda.device(summed.device):
        residual_norm_bwd_kernel[(summed.numel() // hidden,)](
            summed, weight, inv, gs, gy, dr if need_r else summed, da if need_a else summed,
            hidden, *gs.stride(), *gy.stride(), int(grad_s is not None), int(grad_y is not None),
            NEED_R=need_r, NEED_A=need_a, BLOCK=triton.next_power_of_2(hidden),
            num_warps=NUM_WARPS, enable_fp_fusion=False)
    return dr, da


class _ResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, attention, weight, eps):
        ctx.set_materialize_grads(False)
        summed, normalized, inv = _launch_forward(residual, attention, weight, eps)
        ctx.save_for_backward(summed, weight, inv)
        ctx.attention_dtype, ctx.eps = attention.dtype, eps
        return summed, normalized

    @staticmethod
    def backward(ctx, grad_s, grad_y):
        summed, weight, inv = ctx.saved_tensors
        need_r, need_a = ctx.needs_input_grad[:2]
        if (grad_s is None and grad_y is None) or not (need_r or need_a):
            return None, None, None, None
        if torch.is_grad_enabled() or not _ordinary_mode() or any(
                x is not None and not _plain(x) for x in (grad_s, grad_y)):
            # Higher derivatives/batched incoming gradients use a differentiable
            # formula. Recompute inv from saved *output* s, since cached inv has
            # no autograd history. This does not retain residual/attention.
            total = grad_s
            if grad_y is not None:
                inverse = torch.rsqrt(summed.square().mean(-1, keepdim=True) + ctx.eps)
                xhat = summed * inverse
                wdy = grad_y * weight
                norm_grad = (wdy - xhat * (wdy * xhat).mean(-1, keepdim=True)) * inverse
                total = norm_grad if total is None else total + norm_grad
            return (total if need_r else None,
                    total.to(ctx.attention_dtype) if need_a else None, None, None)
        dr, da = _launch_backward(summed, weight, inv, grad_s, grad_y,
                                  need_r, need_a, ctx.attention_dtype)
        return dr, da, None, None


def fused_residual_norm(residual, attention, weight, eps):
    """Return ``(residual + attention, QwenRMSNorm(residual + attention))``.

    Fast path: ordinary contiguous nonempty CUDA tensors, identical activation
    shapes, fp32 residual/frozen 1-D weight, fp16 or fp32 attention, hidden <=8192,
    finite positive scalar eps. Unsupported calls retain the eager autograd graph,
    including weight gradients. Outputs in the supported path are both fp32.
    """
    if _supports_triton(residual, attention, weight, eps):
        return _fused_residual_norm_validated(residual, attention, weight, eps)
    return _eager(residual, attention, weight, eps)


def _fused_residual_norm_validated(residual, attention, weight, eps):
    """Private entry: caller must have just passed _supports_triton."""
    return _ResidualNorm.apply(residual, attention, weight, eps)


def _source_matches(function, expected):
    # inspect.getsource unwraps functools.wraps: reject that indirection before
    # hashing, or an unknown wrapper could look exactly like upstream source.
    if not isinstance(function, types.FunctionType) or hasattr(function, "__wrapped__"):
        return False
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == expected
    except (OSError, TypeError):
        return False


def _team_forward_known(function, original):
    # The team patch can run before OR after this optional decoder patch.
    team = sys.modules.get("solution.rmsnorm")
    return (team is not None and getattr(team, "_patched", False)
            and getattr(team, "_original", None) is original
            and isinstance(function, types.FunctionType)
            and function.__globals__ is vars(team)
            and function.__code__ in team.patch.__code__.co_consts)


def _safe_norm(norm):
    if (type(norm) is not _norm_class or "forward" in vars(norm)
            or "_call_impl" in vars(norm) or getattr(norm, "_compiled_call_impl", None) is not None
            or hasattr(norm, "_hf_hook") or hasattr(norm, "_old_forward")):
        return False
    if any(getattr(norm, name, None) for name in (
            "_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")):
        return False
    global_hooks = torch.nn.modules.module
    if any(getattr(global_hooks, name, None) for name in (
            "_global_forward_pre_hooks", "_global_forward_hooks",
            "_global_backward_pre_hooks", "_global_backward_hooks")):
        return False
    forward = type(norm).forward
    return forward is _norm_original or _team_forward_known(forward, _norm_original)


def _can_patch_call(layer, hidden_states, past_key_values, use_cache, kwargs):
    return (type(layer) is _decoder_class and layer.training and torch.is_grad_enabled()
            and not getattr(layer, "gradient_checkpointing", False)
            and not hasattr(layer, "_hf_hook") and not hasattr(layer, "_old_forward")
            and past_key_values is None and not use_cache
            and kwargs.get("past_key_value") is None and kwargs.get("layer_past") is None
            and _safe_norm(layer.post_attention_layernorm)
            and _supported_residual_weight(hidden_states, layer.post_attention_layernorm.weight,
                                            layer.post_attention_layernorm.variance_epsilon))


def patch():
    """Opt-in, idempotent patch of ONLY Qwen2DecoderLayer.forward; false if unknown."""
    global _patched, _original, _replacement, _decoder_class, _norm_class, _norm_original
    if _patched:
        return _decoder_class.forward is _replacement
    try:
        transformers = importlib.import_module("transformers")
        module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    except ImportError:
        return False
    decoder, norm = getattr(module, "Qwen2DecoderLayer", None), getattr(module, "Qwen2RMSNorm", None)
    original = getattr(decoder, "forward", None)
    norm_forward = getattr(norm, "forward", None)
    team = sys.modules.get("solution.rmsnorm")
    norm_original = getattr(team, "_original", None)
    if _source_matches(norm_forward, _NORM_SHA256):
        norm_original = norm_forward
    elif not _team_forward_known(norm_forward, norm_original):
        return False
    if (getattr(transformers, "__version__", None) != "5.16.1"
            or not _source_matches(original, _DECODER_SHA256)
            or not _source_matches(norm_original, _NORM_SHA256)):
        return False

    @functools.wraps(original)
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if not _can_patch_call(self, hidden_states, past_key_values, use_cache, kwargs):
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs)
        norm = self.post_attention_layernorm
        # Attention/hooks may change dtype or install a norm hook dynamically.
        # Fall back on this already-computed result: NEVER replay attention.
        if (_safe_norm(norm) and _supports_triton(residual, hidden_states, norm.weight,
                                                 norm.variance_epsilon)):
            residual, hidden_states = _fused_residual_norm_validated(
                residual, hidden_states, norm.weight, norm.variance_epsilon)
        else:
            residual = residual + hidden_states
            hidden_states = norm(residual)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    _original, _replacement = original, forward
    _decoder_class, _norm_class, _norm_original = decoder, norm, norm_original
    decoder.forward = forward
    _patched = True
    return True


def unpatch():
    """Restore only our own decoder replacement, without overwriting later patches."""
    global _patched
    if _patched and _decoder_class.forward is _replacement:
        _decoder_class.forward = _original
    _patched = False
