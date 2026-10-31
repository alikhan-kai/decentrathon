"""Fused LoRA forward on q_proj and v_proj, eliminating intermediate tensor allocations.

Eliminates:
  1. `result = result.clone()` in peft.tuners.lora.bnb.Linear4bit.forward (56 redundant 6.3MB copies/step).
  2. Temporary tensor creation for `output * scaling`.
  3. Fuses scale + add into a single PyTorch C++ kernel `torch.add(result, lora_out, alpha=scaling)`.
"""

import torch

ENABLED = True
_original_forward = None
_patched = False


def lora_epilogue(result, lora_out, scaling):
    """Fused epilogue: result + lora_out * scaling."""
    if lora_out.dtype != result.dtype:
        lora_out = lora_out.to(result.dtype)
    return torch.add(result, lora_out, alpha=scaling)


def patch():
    global _original_forward, _patched
    if _patched:
        return
    try:
        import peft.tuners.lora.bnb as bnb_lora
    except ImportError:
        return

    if not hasattr(bnb_lora, "Linear4bit"):
        return

    _original_forward = bnb_lora.Linear4bit.forward

    def fast_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is not None:
            return _original_forward(self, x, *args, **kwargs)
        if self.disable_adapters or self.merged:
            return self.base_layer(x, *args, **kwargs)

        lora_variant = getattr(self, "lora_variant", {})
        use_dora = getattr(self, "use_dora", {})
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            if active_adapter in lora_variant or use_dora.get(active_adapter, False):
                return _original_forward(self, x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x_ = self._cast_input_dtype(x, lora_A.weight.dtype)
            from .dropout_pack import dropout_or_pack
            lora_out = lora_B(lora_A(dropout_or_pack(dropout, x_)))
            if lora_out.dtype != result_dtype:
                lora_out = lora_out.to(result_dtype)
            result = torch.add(result, lora_out, alpha=scaling)
        return result

    bnb_lora.Linear4bit.forward = fast_forward
    _patched = True


def unpatch():
    global _patched
    if not _patched:
        return
    try:
        import peft.tuners.lora.bnb as bnb_lora

        if _original_forward is not None:
            bnb_lora.Linear4bit.forward = _original_forward
    except ImportError:
        pass
    _patched = False
