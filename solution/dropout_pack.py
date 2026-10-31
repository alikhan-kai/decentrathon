"""Keep native dropout RNG and values, store its boolean mask in one bit.

Only the canonical floating-point LoRA dropout path calls this helper. The native forward
still samples the mask. Our backward decodes that exact mask and applies the
same scale; dropout probability and generator consumption do not change.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def pack_dropout_mask(mask, packed, N, BLOCK: tl.constexpr):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 8)
    index = group[:, None] * 8 + bit[None, :]
    keep = tl.load(mask + index, index < N, 0).to(tl.int32)
    byte = tl.sum(keep << bit[None, :], 1)
    tl.store(packed + group, byte, group * 8 < N)


@triton.jit(do_not_specialize=['N'])
def packed_dropout_backward(grad, packed, output, N, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(packed + index // 8, index < N, 0).to(tl.int32)
    keep = ((byte >> (index % 8)) & 1).to(tl.float32)
    value = tl.load(grad + index, index < N, 0).to(tl.float32)
    tl.store(output + index, (value * SCALE) * keep, index < N)


class _Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probability):
        out, mask = torch.ops.aten.native_dropout(x, probability, True)
        n = x.numel()
        packed = torch.empty((triton.cdiv(n, 8),), device=x.device, dtype=torch.uint8)
        pack_dropout_mask[(triton.cdiv(n, 1024),)](mask, packed, n, 128)
        ctx.save_for_backward(packed)
        ctx.scale = 1.0 / (1.0 - probability)
        return out

    @staticmethod
    def backward(ctx, grad):
        (packed,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            index = torch.arange(grad.numel(), device=grad.device)
            keep = ((packed[index // 8] >> (index % 8)) & 1).reshape(grad.shape)
            return (grad * ctx.scale) * keep, None
        grad = grad.contiguous()
        output = torch.empty_like(grad)
        packed_dropout_backward[(triton.cdiv(grad.numel(), 512),)](
            grad, packed, output, grad.numel(), ctx.scale, 512, enable_fp_fusion=False)
        return output, None


def dropout_or_pack(module, x):
    hooks = torch.nn.modules.module
    if (type(module) is not torch.nn.Dropout or module.inplace or not module.training
            or module.p != 0.05 or type(x) is not torch.Tensor
            or x.device.type != 'cuda' or x.dtype not in (torch.float16, torch.float32)
            or not x.requires_grad or not torch.is_grad_enabled() or not x.is_contiguous()
            or x.numel() == 0 or x.is_conj() or x.is_neg()
            or torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()
            or torch.__version__.split('+')[0] != '2.11.0'
            or any(getattr(module, name, None) for name in (
                '_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'))
            or any(getattr(hooks, name, None) for name in (
                '_global_forward_hooks', '_global_forward_pre_hooks',
                '_global_backward_hooks', '_global_backward_pre_hooks'))
            or 'forward' in module.__dict__):
        return module(x)
    if torch.autograd.forward_ad.unpack_dual(x).tangent is not None:
        return module(x)
    return _Dropout.apply(x, module.p)
