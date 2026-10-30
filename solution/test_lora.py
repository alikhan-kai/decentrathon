"""Template unit test for solution/lora.py (owner: lora). Skipped without CUDA and until the kernel
exists. Contract: define `lora_epilogue(result, lora_out, scaling)` = result + (lora_out * scaling)
cast to result.dtype, with autograd. Dropout is NOT part of it and must not change.

Tolerances (explicit, from solution/kcheck.py): fp16 atol = rtol = 2e-3, fp32 atol = rtol = 1e-4.
"""

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_lora_epilogue_matches_pytorch(dtype):
    from solution.kcheck import check_fwd_bwd

    fn = getattr(importlib.import_module("solution.lora"), "lora_epilogue", None)
    if fn is None:
        pytest.skip("kernel not written yet: define lora_epilogue(result, lora_out, scaling) in solution/lora.py")
    scaling = 2.0  # alpha 32 / r 16
    check_fwd_bwd(lambda r, o: fn(r, o, scaling), lambda r, o: r + o * scaling,
                  shapes=[(4, 512, 1536), (4, 512, 1536)], dtype=dtype)
