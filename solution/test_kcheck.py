"""Example of using solution/kcheck.py (it checks the shipped SwiGLU against PyTorch)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kcheck_on_swiglu(dtype):
    from solution.kcheck import check_fwd_bwd
    from solution.swiglu import swiglu

    errors = check_fwd_bwd(swiglu, lambda a, b: torch.nn.functional.silu(a) * b,
                           shapes=[(3, 77, 1001), (3, 77, 1001)], dtype=dtype)
    assert set(errors) == {"out", "grad0", "grad1"}
