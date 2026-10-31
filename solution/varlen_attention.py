"""One native PyTorch attention call for independently packed causal rows.

Private validated entry from packed.py on the pinned T4 path. This is PyTorch's
existing memory-efficient attention, not an original kernel and not credited
toward the challenge's own-kernel requirement. No extra dependency is loaded.
The caller constructs cumulative lengths from its admitted right-padding plan.
"""
import torch


class _PackedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cumulative, longest, scale):
        # The native operator consumes B,N,H,D; the existing caller uses B,H,N,D.
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        out, lse, seed, offset, max_q, max_k = torch.ops.aten._efficient_attention_forward(
            q, k, v, None, cumulative, cumulative, longest, longest,
            0.0, 1, True, scale=scale)
        ctx.save_for_backward(q, k, v, out, lse, seed, offset, cumulative)
        ctx.max_q, ctx.max_k, ctx.scale = max_q, max_k, scale
        return out.transpose(1, 2)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError('Native packed attention supports first-order backward only')
        q, k, v, out, lse, seed, offset, cumulative = ctx.saved_tensors
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            grad.transpose(1, 2), q, k, v, None, out, cumulative, cumulative,
            ctx.max_q, ctx.max_k, lse, 0.0, seed, offset, 1, False,
            scale=ctx.scale)
        return (dq.transpose(1, 2) if ctx.needs_input_grad[0] else None,
                dk.transpose(1, 2) if ctx.needs_input_grad[1] else None,
                dv.transpose(1, 2) if ctx.needs_input_grad[2] else None,
                None, None, None)


def try_packed_attention(query, key, value, cumulative, longest, *, dropout, scale):
    """Return None before dispatch for unsupported metadata; never replay a call."""
    if (dropout != 0.0 or torch.__version__.split('+')[0] != '2.11.0'
            or torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
            or query.ndim != 4 or query.shape[0] != 1 or query.shape[1] != 12
            or query.shape[-1] != 128 or query.shape[2] <= 0
            or key.shape != query.shape or value.shape != query.shape
            or any(type(t) is not torch.Tensor or t.dtype != torch.float16
                   or t.device != query.device or t.stride(-1) != 1
                   for t in (query, key, value))
            or query.device.type != 'cuda' or cumulative.dtype != torch.int32
            or cumulative.device != query.device or cumulative.ndim != 1
            or not cumulative.is_contiguous() or cumulative.numel() < 2
            or type(longest) is not int or longest < 1):
        return None
    return _PackedAttention.apply(query, key, value, cumulative, longest, scale)


def try_grouped_attention(query, key, value, cumulative, longest, *, dropout, scale):
    """Avoid materializing six copies of each K/V head on the admitted GQA path.

    Treat the two KV groups as independent packed sequences with six Q heads
    each. K/V expand with a zero head stride. PyTorch's ExpandBackward sums each
    group's six cotangents in the same FP16 boundary as the original repeat_kv.
    cumulative describes the original row lengths repeated twice, group-major.
    """
    if (dropout != 0.0 or torch.__version__.split('+')[0] != '2.11.0'
            or torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
            or query.ndim != 4 or query.shape[0] != 1 or query.shape[1] != 12
            or query.shape[-1] != 128 or query.shape[2] <= 0
            or key.shape != (1, 2, query.shape[2], 128) or value.shape != key.shape
            or any(type(t) is not torch.Tensor or t.dtype != torch.float16
                   or t.device != query.device or t.stride(-1) != 1
                   for t in (query, key, value))
            or query.device.type != 'cuda' or cumulative.dtype != torch.int32
            or cumulative.device != query.device or cumulative.ndim != 1
            or not cumulative.is_contiguous() or cumulative.numel() < 3
            or type(longest) is not int or longest < 1):
        return None
    n = query.shape[2]
    q = query.reshape(1, 2, 6, n, 128).permute(0, 2, 1, 3, 4).reshape(1, 6, 2*n, 128)
    k = key.reshape(1, 1, 2*n, 128).expand(1, 6, 2*n, 128)
    v = value.reshape(1, 1, 2*n, 128).expand(1, 6, 2*n, 128)
    out = _PackedAttention.apply(q, k, v, cumulative, longest, scale)
    return out.reshape(1, 6, 2, n, 128).permute(0, 2, 1, 3, 4).reshape(1, 12, n, 128)
