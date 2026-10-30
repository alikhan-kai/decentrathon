"""STUB (owner: rmsnorm, branch feat/rmsnorm). Fused RMSNorm, forward + backward, in Triton.

Task: replace the ~5 PyTorch ops of Qwen2RMSNorm (cast to fp32, pow, mean, rsqrt, mul, cast back,
mul by weight) by one forward and one backward kernel; later fuse with the residual add.
Goal: our own kernel time, summed over all norms of a step, as large a cut of the elementwise/cast
share as possible, without changing the outputs (check 5: entropy/accuracy/row losses within 3%).
Do NOT add trainable parameters; the norm weight stays frozen as in base.

Where to patch (read in transformers 5.16.1, solution/ notes in docs/TEAM.md):
  - transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward
      (class is decorated with @use_kernel_forward_from_hub("RMSNorm"); patching the class
       attribute `forward` after import should work, but CHECK that no kernelize() call replaces it)
      base forward: x.float(); var = x.pow(2).mean(-1, keepdim=True);
      x * rsqrt(var + self.variance_epsilon); return self.weight * x.to(input_dtype)
      NOTE: if self.weight is fp32 (README: prepare_model_for_kbit_training keeps norms in fp32)
      the product is fp32, not fp16. CHECK on T4 which dtype comes out and reproduce exactly.
  - step 2, fusion with the residual: Qwen2DecoderLayer.forward
      (residual = h; h = input_layernorm(h); attn; h = residual + h; residual = h;
       h = post_attention_layernorm(h); mlp; h = residual + h)
      and Qwen2Model.forward for the final self.norm.
  - there are 2 norms per layer + 1 final: 28 layers x 2 + 1 for Qwen2.5-1.5B (CHECK num_hidden_layers).

TODO:
  [ ] kernel fwd: one program per row, fp32 accumulation, save rstd for backward
  [ ] kernel bwd: dx (and no dweight: the weight is frozen; CHECK requires_grad on T4)
  [ ] torch.autograd.Function + patch() (idempotent, guard on x.is_cuda)
  [ ] test_rmsnorm.py with solution/kcheck.py (fp16 and fp32, odd hidden sizes)
  [ ] set ENABLED = True only after the unit test and `quick` are green on a T4
"""

ENABLED = False


def patch():
    return
