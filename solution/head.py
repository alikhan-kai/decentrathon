"""Frozen output projection with bounded logits and backward-time scaling.

The chunking idea is established by CCE/Liger and Soup's own chunked loss.
This implementation uses our original vocabulary kernels and independently
recomputes chunks in backward, when the actual AMP loss scale is available.
No full-batch vocabulary tensor or pre-scaled fp16 gradient is saved.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .vocab import vocab_backward, vocab_stats


@dataclass(frozen=True)
class HeadResult:
    loss: torch.Tensor
    token_loss: torch.Tensor
    token_entropy: torch.Tensor
    predictions: torch.Tensor


def projection_dtype(hidden, weight):
    if torch.is_autocast_enabled(hidden.device.type):
        return torch.get_autocast_dtype(hidden.device.type)
    if hidden.dtype != weight.dtype:
        raise TypeError("without autocast, hidden and weight dtypes must match")
    return hidden.dtype


class _FrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, denominator, chunk_size, compute_dtype):
        flat = hidden.reshape(-1, hidden.shape[-1])
        target = targets.reshape(-1)
        rows = flat.shape[0]
        losses = torch.empty(rows, device=hidden.device, dtype=torch.float32)
        entropy = torch.empty_like(losses)
        lse = torch.empty_like(losses)
        predictions = torch.empty(rows, device=hidden.device, dtype=torch.int64)
        # Frozen weights are not reliably retained by autocast's weight cache.
        # Cast once per invocation, release after forward, and recreate in backward.
        with torch.autocast(hidden.device.type, enabled=False):
            projected_weight = weight.to(compute_dtype)
            for start in range(0, rows, chunk_size):
                end = min(start + chunk_size, rows)
                logits = F.linear(flat[start:end].to(compute_dtype), projected_weight)
                loss_i, entropy_i, prediction_i, lse_i = vocab_stats(logits, target[start:end])
                losses[start:end] = loss_i
                entropy[start:end] = entropy_i
                predictions[start:end] = prediction_i
                lse[start:end] = lse_i
                del logits
            loss = losses.sum() / denominator
        ctx.save_for_backward(hidden, weight, targets, lse, denominator)
        ctx.chunk_size = chunk_size
        ctx.compute_dtype = compute_dtype
        losses = losses.reshape(targets.shape)
        entropy = entropy.reshape(targets.shape)
        predictions = predictions.reshape(targets.shape)
        ctx.mark_non_differentiable(losses, entropy, predictions)
        return loss, losses, entropy, predictions

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_tokens, _grad_entropy, _grad_predictions):
        hidden, weight, targets, lse, denominator = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        target = targets.reshape(-1)
        result = torch.empty_like(flat)
        with torch.autocast(hidden.device.type, enabled=False):
            projected_weight = weight.to(ctx.compute_dtype)
            # This includes GradScaler's upstream multiplier before fp16 narrowing.
            scale = grad_loss.float() / denominator
            for start in range(0, flat.shape[0], ctx.chunk_size):
                end = min(start + ctx.chunk_size, flat.shape[0])
                logits = F.linear(flat[start:end].to(ctx.compute_dtype), projected_weight)
                dz = vocab_backward(logits, target[start:end], lse[start:end], scale)
                # Half GEMM returns half, then the original input cast promotes dH.
                result[start:end] = torch.mm(dz, projected_weight).to(hidden.dtype)
                del logits, dz
        return result.reshape(hidden.shape), None, None, None, None, None


def frozen_head_loss(hidden, weight, labels, num_items_in_batch=None, chunk_size=128):
    """Compute shifted causal CE and *all-position* detached diagnostics.

    ``weight`` must be a frozen bias-free [vocabulary, hidden] matrix. ``labels``
    are unshifted [batch, time], with -100 ignored. A supplied item count is the
    normalization used by HF across accumulation batches. Otherwise use the
    number of valid shifted targets (including HF's NaN for an empty mean).
    """
    if hidden.ndim != 3 or weight.ndim != 2 or labels.shape != hidden.shape[:2]:
        raise ValueError("expected hidden [B,T,H], weight [V,H], labels [B,T]")
    if hidden.shape[-1] != weight.shape[-1] or weight.shape[0] < 1:
        raise ValueError("incompatible output-head dimensions")
    if hidden.shape[0] < 1 or hidden.shape[1] < 1:
        raise ValueError("empty batch or sequence is unsupported")
    if weight.requires_grad:
        raise ValueError("the output head must be frozen")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must be integer token indices")
    if hidden.device != weight.device or labels.device != hidden.device:
        raise ValueError("hidden, weight and labels must share a device")
    dtype = projection_dtype(hidden, weight)
    if dtype not in (torch.float16, torch.float32):
        raise TypeError("only fp16 and fp32 projection paths are implemented")
    targets = F.pad(labels[..., 1:], (0, 1), value=-100).contiguous()
    denominator = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(denominator, dtype=torch.float32, device=hidden.device)
    if denominator.numel() != 1:
        raise ValueError("num_items_in_batch must be a scalar")
    outputs = _FrozenHead.apply(hidden, weight, targets, denominator.reshape(()), chunk_size, dtype)
    return HeadResult(*outputs)
