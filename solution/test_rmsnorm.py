"""Template unit test for solution/rmsnorm.py (owner: rmsnorm). Skipped without CUDA and until the
kernel exists. Contract: define `rmsnorm(x, weight, eps)` in solution/rmsnorm.py with autograd support.

Tolerances (explicit, from solution/kcheck.py): fp16 atol = rtol = 2e-3, fp32 atol = rtol = 1e-4.
The reference is the base Qwen2RMSNorm math in fp32. CHECK the dtype the base returns on the T4
(fp32 weight * fp16 input promotes to fp32) and make the kernel return the same dtype.
"""

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(x, weight, eps=1e-6):
    x = x.float()
    return weight.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("hidden", [1536, 1001])  # 1536 = Qwen2.5-1.5B hidden size (CHECK), 1001 = odd
def test_rmsnorm_matches_pytorch(dtype, hidden):
    from solution.kcheck import check_fwd_bwd

    fn = getattr(importlib.import_module("solution.rmsnorm"), "rmsnorm", None)
    if fn is None:
        pytest.skip("kernel not written yet: define rmsnorm(x, weight, eps) in solution/rmsnorm.py")
    check_fwd_bwd(lambda x, w: fn(x, w, 1e-6), reference, shapes=[(4, 512, hidden), (hidden,)],
                  dtype=dtype)
