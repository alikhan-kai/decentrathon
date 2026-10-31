# Exact-commit review: 25855208695fd113fa1897385abccc73f71a19cd

Reviewed 2026-10-31, bounded CPU-only review. **Proceed with the requested unchanged-commit Kaggle benchmark, but do not sign off the new LoRA forward as numerically equivalent.** There is a reproduced task-relevant gradient mismatch from the added BF16 cast before dropout. This is a blocker to an equivalence claim, not proof that the official harness will fail and not a reason to replace the user's requested commit with a patch.

No repository edits, installs, network, browser, GPU use, or child agents. Only this report and `/tmp/decentrathon-time-new-lora-repro-r1.py` / `.log` were written. Repository status was clean before and after. The worktree HEAD is `50e84ed7b9fa9fbe7ddb8bb83de2e26c3cbcd73b`, so reviewed project sources were read explicitly through `git show 258552...:<path>`, never inferred from checkout files.

## Scope and actual task

`git log 662557..258552` contains exactly one commit: `2585520 perf: eliminate LoRA clone and intermediate tensors, optimize NF4 contiguous check`. Its only changed files are `solution/lora.py` (71 additions / 25 deletions) and `solution/nf4.py` (5 / 1). No task, harness, memory-policy, or other kernel changes occur in that range.

Installed versions: PyTorch `2.11.0+cpu`, PEFT `0.20.0`, bitsandbytes `0.50.2`, Transformers `5.16.1`.

Exact `task/task.yaml:18-22` specifies **rank 16, alpha 32, scaling 2.0**, not rank 8 / alpha 16. Batch size 4, maximum sequence length 512, accumulation 1, 4-bit quantization, and automatic targets remain unchanged. Soup's installed `config/schema.py` defaults dropout to 0.05. Installed `trl/trainer/sft_trainer.py:777-785` explicitly converts trainable QLoRA parameters to BF16 independently of FP16 AMP. Installed Soup SFT precision selection uses FP16 on CUDA cards without BF16 support. PEFT k-bit preparation and the enabled RMSNorm path preserve FP32 residual/norm activations; BNB `nn/modules.py:610-637` returns the base linear result in its incoming dtype. Thus the relevant comparison is FP32 input/base result, BF16 LoRA weights, FP16 matmul AMP. The new comment at `solution/lora.py:48` claiming FP16 input matching LoRA weights is incorrect for this task.

## Task-relevant findings

**1. Added cast changes dropout values and training gradients (high confidence, reproduced).**

At `solution/lora.py:50-53`, input is converted FP32 -> BF16 before dropout even when autocast is enabled. PEFT `tuners/lora/bnb.py:forward` deliberately skips its input cast when CUDA autocast is enabled: dropout runs on FP32 input and only the following linear autocasts to FP16. The new sequence is FP32 -> BF16 -> BF16 dropout -> FP16 GEMM input. This adds rounding before dropout, changes dropout output rounding, and introduces a BF16 gradient boundary on the input branch.

The repro executes AST-extracted, otherwise unchanged installed PEFT `Linear4bit.forward` and exact-commit `fast_forward` on real CPU tensors, BF16 `nn.Linear` weights, and actual CPU FP16 and BF16 autocast. Because PEFT's no-argument autocast query addresses CUDA, the reference function's global torch facade redirects only that query to `torch.is_autocast_enabled('cpu')`. This allows the actual CPU AMP execution to follow PEFT's CUDA-autocast branch. The base quantized linear is replaced by the same FP32 `x * 0.25` in both runs; this isolates LoRA behavior and does **not** test BNB CUDA GEMM, CUDA dropout, full-model loss, performance, or memory. Shapes: input/output 8 x 64, rank 16, scaling 2, dropout 0.05, identical fixed RNG seeds and parameter values.

Actual CPU FP16 AMP results:

- Dropout received FP32 in PEFT and BF16 in the replacement. Masks and post-forward RNG states were equal. Dropout values differed at 489/512 elements, maximum absolute difference 0.018199.
- With B initialized to zero, outputs, input gradients, and A gradients were identical, **but B gradients differed at 505/1024 elements**, maximum absolute difference 0.015625 and relative L2 error 0.003362. Of these, 159 exceeded `atol=rtol=2e-3`. This demonstrates why matching the first forward loss is insufficient: the first adapter update already differs.
- With nonzero B, forward outputs differed at 470/512 elements, maximum absolute difference 0.00878906, relative L2 error 0.002451; 81 elements exceeded `atol=rtol=2e-3`. A gradients had max absolute difference 0.0078125; input gradients 0.00118215; B gradients had the same discrepancy as above.
- Actual CPU BF16 autocast also reproduced forward and gradient discrepancies (log contains full metrics).

These are controlled CPU numerical findings, not measured Kaggle error bounds. For the ordinary single-adapter task there is still one dropout call of the same shape; CPU evidence does not establish any changed RNG consumption. CUDA dtype-specific dropout mask/state equality remains a bounded check for the parent, not an asserted failure.

**2. `add(alpha=2)` moves the scaling/cast boundary (reproduced, narrower significance).**

PEFT under FP16 AMP computes `FP16_lora_out * 2` in FP16, then adds to FP32 base result. New lines 54-56 first promote the LoRA result to FP32, then use `torch.add(..., alpha=2)` in FP32. For normal finite FP16 values whose doubling does not overflow, scaling by the power-of-two 2 is exact, so there is no basis for claiming a generic forward FMA rounding defect here. However, backward and extreme-value boundaries differ:

- Real FP16 tensor scalar repro, `lora_out=0.25`, base result `0.125`, incoming FP32 gradient `2^-25`: both outputs are `0.625`; PEFT-style LoRA-output gradient is `0`, replacement gradient is `2^-24`. Old backward casts the incoming gradient to FP16 before scaling; new backward scales in FP32 before casting to FP16.
- Synthetic overflow example `lora_out=40000`, FP32 base `-80000`: old result `inf`, new result `0`. This is boundary evidence only; no claim that actual model activations reach it. GradScaler can also change whether the tiny-gradient boundary is encountered.

The existing `solution/test_lora.py` checks only same-dtype epilogue inputs, uses scaling 2, and does not exercise the mixed FP32/BF16/FP16 forward, dropout, or first B update. It cannot certify this replacement.

**3. Clone removal: no demonstrated task-path blocker.**

The new addition is out-of-place. CPU forward/backward completed without an alias/view error. The pinned PEFT defensive-clone comment alone is not evidence of a real error with Torch 2.11; do not block the benchmark on that speculation. Actual BNB CUDA backward is outside this CPU check and belongs in the unchanged official training run.

## Compatibility boundaries outside the ordinary task

Source inspection plus small control-flow stand-ins established:

- Merged, adapters enabled: both implementations call base once; no mismatch identified for that ordinary merged case.
- Disabled, unmerged: both call base once.
- **Disabled and merged** (`lora.py:36-37`): reference calls `unmerge()` before base; replacement returns base without unmerging. Repro confirms reference unmerges and replacement leaves the merged state intact, hence disabled adapters can remain baked into output.
- **`adapter_names`**: reference validates and pops the keyword, then dispatches mixed-batch forwarding. Replacement forwards it to the base. BNB's actual base signature accepts only `x`; the stand-in reproduces `TypeError: Base.forward() got an unexpected keyword argument 'adapter_names'`. The omitted `_check_forward_args` also removes upstream validation.
- **DoRA**: pinned PEFT still assigns `use_dora[adapter_name] = config.use_dora` (`layer.py:281`); the new test recognizes normal DoRA. It falls back only after already evaluating the base. Single-DoRA control-flow repro: base is called twice instead of once. If a vanilla adapter precedes DoRA, the earlier adapter/dropout is evaluated before fallback and then evaluated again inside PEFT; seeded repro shows different post-forward RNG states. DoRA arithmetic itself was not emulated or claimed tested.
- Other `lora_variant` entries that do not set legacy `use_dora` are treated as vanilla by the replacement; a sentinel variant confirms skipped variant dispatch. `alora_offsets` is also no longer removed before the base call. These compatibility cases are not the fixed vanilla single-adapter training task and do not independently block its requested benchmark.
- Outside autocast, the replacement directly casts instead of honoring PEFT's `_cast_input_dtype` / `cast_input_dtype_enabled` control. The normal default-enabled path is not affected by this particular API difference.

## NF4 and memory policy

The NF4 delta is behavior-preserving for ordinary tensors: `.contiguous()` already returns the same tensor without allocating when it is contiguous. CPU checks confirm both old and new paths retain identity for contiguous tensors and copy to equal contiguous values for transposed tensors. Noncontiguous behavior and kernel arguments are otherwise unchanged. This change does not eliminate a previous tensor copy for already-contiguous input; no CUDA speedup or memory saving was measured.

**No memory-policy constant changed between 662557 and 258552.** Both retain `ENABLE_SAVE=True`, `LAYER_BUDGET=True`, and `BUDGET_ROWS=1800`. Commit **662557 itself**, immediately before the reviewed range, changed the budget from 1300 to 1800. The policy keeps all 28 layers on native saves through 1800 packed rows, and selects only 20 native layers at the maximum 2048 rows (remaining eligible layers rematerialize). Thus “eliminate recompute on all batches” is not a universal statement for every legal task shape. The surrounding comment saying the budget is below 1024 is stale.

The newest LoRA change removes the base clone and FP16 scaling temporary, but also introduces BF16 pre-dropout conversion and FP32 LoRA-output promotion for the actual task. Their net peak effect needs measurement. The “just over 4 GiB” observation cannot be confirmed or extrapolated from this CPU review. Use the official harness's externally sampled all-GPU peak MiB, with allocated/reserved figures as distinct supporting metrics, on exact 258552.

## Bounded gate recommendation

1. Preserve the exact commit and official harness bytes. A numerical comparison can live in a separate diagnostic process; do not substitute a corrected LoRA implementation into the measured submission.
2. A focused T4 diagnostic should use FP32 activations/base output, BF16 A/B, FP16 autocast, rank 16/alpha 32/dropout 0.05, both zero and nonzero B, and identical CUDA RNG snapshots. Record forward, input/A/B gradients, dropout mask and post-call RNG, plus one optimizer update. Use task-sized q/v shapes when practical. This is an explicit check of the identified boundaries, not a broad new test campaign.
3. Proceed with the parent's unchanged official Kaggle run and official checks. Report finite successful forward/backward, actual trainer dtypes/config, first and full loss-curve gaps, evaluation loss, and measured peak memory. Existing rules use first loss gap <=1%, mean curve gap <=3%, eval loss increase <=0.5%, and their other required checks. The CPU discrepancy does not establish that these tolerances will fail. A zero-B first-loss match alone cannot clear the gradient finding.
4. If official checks fail, publish the exact-commit failure and evidence. Do not silently patch, loosen tolerances, or relabel an alternate source as 258552. If checks pass, report that measured result with this precision caveat; do not call the LoRA transform bitwise or gradient equivalent.

## Provenance and reproduction

Run (CPU only, no cache writes):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/homelab/.t3/projects/decentrathon/.venv/bin/python /tmp/decentrathon-time-new-lora-repro-r1.py
```

Full captured output: `/tmp/decentrathon-time-new-lora-repro-r1.log`. Repro verifies reviewed Git blobs and installed PEFT/BNB source bytes are unchanged after its numerical checks. SHA-256 values:

```text
solution/lora.py          292b0a7ec98c75b0d5118a930bd0a4137b39c1ff1c6bd0d228154275b1573b0e
solution/nf4.py           d52fce01a6d04d755341ff34ffa97ba1f6f2c944a6c22fcc683607ec0f716d7a
solution/swiglu_pack.py   64ecd83e1b02e21a467f5e0a0c9dc0dbd95bc028a8726b4b062d2748cbf9093d
task/task.yaml           dd70c739fcd143e06aef3bb1efcc9386f682bf1de26f30e08409d86a97ff47ce
peft/tuners/lora/bnb.py   96ed6fda63d7c67a4646faea8ba38483e8be8670059a3ac3dd740a99bdab6950
peft/tuners/lora/layer.py 1ef4e7033ebd01c9c6fbf0fc3837695b75b795c54f8a837b7578e8dfba2251a4
bitsandbytes/nn/modules.py f7ab2160681e2f089caae80d6fd283ac6a9b9f1b82269e6f395c3b046faf606e
```
