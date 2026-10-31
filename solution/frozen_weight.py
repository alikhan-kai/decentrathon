"""Lossless FP16 storage for an eligible frozen, tied embedding/head weight.

The loaded FP32 values must round-trip bit for bit. Embedding results remain
FP32; ordinary linear calls expand storage when needed to retain their original
compute dtype. The fused head already computes with FP16. No trainer flag changes.
"""
import torch

_DTYPE = "_ember_original_storage_dtype"
_CHECKED = "_ember_half_storage_checked"
_patched = False
_original_embedding = None
_original_linear = None


def _half_if_exact(weight):
    """Bound the verification temporary; reject nonfinite or lossy values."""
    half = weight.detach().to(torch.float16)
    exact = torch.ones((), device=weight.device, dtype=torch.bool)
    source = weight.detach().reshape(-1)
    packed = half.reshape(-1)
    for start in range(0, source.numel(), 1 << 20):
        original = source[start:start + (1 << 20)]
        restored = packed[start:start + (1 << 20)].float()
        exact &= torch.isfinite(original).all()
        exact &= (original.view(torch.int32) == restored.view(torch.int32)).all()
    return half if bool(exact) else None


def restore_weight(weight):
    dtype = getattr(weight, _DTYPE, None)
    if dtype is not None:
        with torch.no_grad():
            weight.data = weight.data.to(dtype)
        delattr(weight, _DTYPE)


def _install_forwards():
    global _patched, _original_embedding, _original_linear
    if _patched:
        return
    _original_embedding = torch.nn.Embedding.forward
    _original_linear = torch.nn.Linear.forward

    def embedding(self, *args, **kwargs):
        dtype = getattr(self.weight, _DTYPE, None)
        if dtype is not None and (self.weight.requires_grad or self.max_norm is not None):
            restore_weight(self.weight)
            dtype = None
        result = _original_embedding(self, *args, **kwargs)
        return result.to(dtype) if dtype is not None else result

    def linear(self, input):
        dtype = getattr(self.weight, _DTYPE, None)
        if dtype is None:
            return _original_linear(self, input)
        if self.weight.requires_grad:
            restore_weight(self.weight)
            return _original_linear(self, input)
        # FP16 autocast already casts the original FP32 weight to these exact
        # values. In other modes preserve the original linear compute contract.
        device = self.weight.device.type
        if torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device) == torch.float16:
            return _original_linear(self, input)
        return torch.nn.functional.linear(input, self.weight.to(dtype), self.bias)

    torch.nn.Embedding.forward = embedding
    torch.nn.Linear.forward = linear
    _patched = True


def compact_tied_weight(model):
    """Called only after fused-head dispatch admits a supported training call."""
    head = getattr(model, "lm_head", None)
    embedding = getattr(getattr(model, "model", None), "embed_tokens", None)
    if (type(head) is not torch.nn.Linear or type(embedding) is not torch.nn.Embedding
            or embedding.weight is not head.weight or embedding.max_norm is not None
            or "forward" in embedding.__dict__):
        return False
    weight = head.weight
    if getattr(weight, _DTYPE, None) is not None:
        return True
    if (not weight.is_cuda or weight.requires_grad or weight.dtype != torch.float32
            or not weight.is_contiguous() or getattr(weight, _CHECKED, False)):
        return False
    setattr(weight, _CHECKED, True)
    half = _half_if_exact(weight)
    if half is None:
        print("FROZEN_WEIGHT_STORAGE skipped: FP16 round-trip is not exact", flush=True)
        return False
    _install_forwards()
    with torch.no_grad():
        setattr(weight, _DTYPE, weight.dtype)
        weight.data = half
    print(f"FROZEN_WEIGHT_STORAGE exact FP16: {weight.numel()} values; tied Parameter retained", flush=True)
    return True
