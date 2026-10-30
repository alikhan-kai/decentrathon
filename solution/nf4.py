"""STUB (owner: nf4, branch feat/nf4). Triton NF4 dequantization instead of bitsandbytes' CUDA kernel.

Task: dequantize the packed 4-bit weights (NF4, double quantization, blocksize 64) with our own
Triton kernel, no slower than bnb's, ideally fused with the following steps.
Goal: our kernel time as large a share of the step as it can honestly be, outputs bit-compatible
with bnb within fp16 rounding. The base spends about 5% of GPU time on NF4 dequant (README).

Where to patch (read in bitsandbytes 0.50.2):
  - bitsandbytes.functional.dequantize_4bit(A, quant_state, absmax, out, blocksize, quant_type)
      public wrapper; on CUDA it ends in torch.ops.bitsandbytes.dequantize_4bit, registered for
      "cuda" in bitsandbytes/backends/cuda/ops.py (`_dequantize_4bit_impl`, calls lib.cdequantize_blockwise_*_nf4)
  - bitsandbytes.autograd._functions.MatMul4Bit
      forward -> torch.ops.bitsandbytes.gemm_4bit.default(A, B, shape, absmax, blocksize, quant_type, bias, ...)
      backward -> F.dequantize_4bit(B, ctx.state), then torch.matmul(grad_output, W)
      (this is where the dequantized weight is needed in training)
  - torch.ops.bitsandbytes.gemm_4bit "cuda" kernel is registered at backends/cuda/ops.py:934; the
      "default" implementation (backends/default/ops.py) is dequantize_4bit + F.linear.
      CHECK on T4 which of the two the training forward takes for 2048-token batches.
  - double quantization: quant_state.nested is True, quant_state.state2.blocksize == 256: the per-block
      absmax is itself stored in 8 bit (absmax_8bit, absmax_code, absmax_offset); the kernel must
      reproduce  absmax = code[absmax_8bit] * state2.absmax[i // 256] + offset  (CHECK against
      _gemm_4bit_default_impl in backends/default/ops.py).
  - Linear4bit.forward (bitsandbytes/nn/modules.py) casts x to compute_dtype and calls bnb.matmul_4bit.

TODO:
  [ ] kernel: unpack 2 nibbles per byte, NF4 code table, scale by (nested) absmax, write fp16
  [ ] compare with bnb.functional.dequantize_4bit on real quant_states (kcheck + a real Linear4bit)
  [ ] patch() (idempotent) around dequantize_4bit or the cuda op, fall back to the original for
      non-NF4 / non-fp16 / non-CUDA / empty input
  [ ] test_nf4.py (skipped without CUDA); set ENABLED = True after `quick` is green on a T4
  [ ] no @triton.autotune, no CUDA graphs
"""

ENABLED = False


def patch():
    return
