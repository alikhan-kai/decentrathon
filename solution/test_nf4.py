"""Template unit test for solution/nf4.py (owner: nf4). Skipped without CUDA and until the kernel
exists. Contract: define `dequantize_nf4(packed, quant_state)` returning the same tensor as
bitsandbytes.functional.dequantize_4bit(packed, quant_state) (fp16, shape [out, in]).

Tolerance: fp16 atol = rtol = 2e-3 (kcheck default); bit-exact output is expected, so tighten it to 0
if the kernel can match. Test the double-quantized state (compress_statistics=True), blocksize 64.
"""

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (1000, 1000)])
def test_dequantize_nf4_matches_bitsandbytes(shape):
    fn = getattr(importlib.import_module("solution.nf4"), "dequantize_nf4", None)
    if fn is None:
        pytest.skip("kernel not written yet: define dequantize_nf4(packed, quant_state) in solution/nf4.py")
    import bitsandbytes.functional as F  # CHECK the signature in bitsandbytes 0.50.2

    torch.manual_seed(0)
    weight = torch.randn(*shape, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(weight, blocksize=64, quant_type="nf4", compress_statistics=True)
    ref = F.dequantize_4bit(packed, state)
    out = fn(packed, state)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)
