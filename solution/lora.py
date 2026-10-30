"""STUB (owner: lora, branch feat/lora). Fused LoRA epilogue on q_proj and v_proj.

Task: fuse the elementwise work around the LoRA branch: dtype casts, `* scaling` and the addition
to the base output, in forward and backward. DO NOT change dropout (0.05), rank, alpha, targets
or add parameters (check 5: same adapter).
Goal: remove the casts/clone/add round trips of the PEFT LoRA path on 2 x 28 linears per step.

Where to patch (read in peft 0.20.0):
  - peft.tuners.lora.bnb.Linear4bit.forward   (peft/tuners/lora/bnb.py, class defined inside
      `if is_bnb_4bit_available():`; patch after `import peft.tuners.lora.bnb`)
      base path: result = self.base_layer(x); result = result.clone();
      x = self._cast_input_dtype(x, lora_A.weight.dtype)   (when autocast is off)
      output = lora_B(lora_A(dropout(x))) * scaling; output = output.to(expected_dtype);
      result = result + output
  - dispatch: peft.tuners.lora.bnb.dispatch_bnb_4bit builds Linear4bit(target, adapter_name, ...)
      for bnb.nn.Linear4bit base layers; with `target_modules: auto` only q_proj and v_proj get it
      (README; CHECK in the trained model).
  - the dropout must run exactly as in base (same RNG consumption, same mask), CHECK before fusing it.

TODO:
  [ ] decide the fusion boundary (scale + cast + add after the two small matmuls)
  [ ] kernel fwd and bwd, autograd.Function, patch() (idempotent, fall back for other adapters)
  [ ] test_lora.py with solution/kcheck.py; set ENABLED = True after `quick` is green on a T4
"""

ENABLED = False


def patch():
    return
