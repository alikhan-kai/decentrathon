# Rotary and residual-normalization contribution

Person 3 / `feat/rotary`, based on the team's passing `feat/check4` commit
`462a9f7`. Owns `solution/rotary.py`, `solution/residual_norm.py`, their tests,
and this evidence directory. The separately reviewed head correction from
[PR #3](https://github.com/alikhan-kai/decentrathon/pull/3) is now merged into
`solution/head_loss.py`. NF4, RMSNorm, LoRA, registry, task and harness are unchanged.

The [follow-up research](RESEARCH.md) records the Reddit leads, primary technical
references and rejected T4 experiments. Real-mode FX tracing now takes the eager
RoPE path in both forward and backward. The merged
[head-gradient correction](head_correction/README.md) fixes a demonstrated
scale-before-cast defect in the team implementation and is used by the existing
head registry entry.
That correction now passes repeated full-training validation: **78.07 s median,
8367 MiB, 7.0106% minimum eligible share**, all ten checks passed, evaluation
loss **2.04966**, equal to the measured baseline. The archived standalone patch
remains available for older checkouts; do not apply it again to this branch.

**Rotary is ready for integrator selection.** Its conservative observed
combined kernel share is **7.05%**, versus about **6.08–6.10%** in the later
team reference repeats. The requested 8–9% margin is not yet established.
The combined candidate passed all ten official checks across two repeats at
**7.60% minimum share**, but residual fusion has not demonstrated an
end-to-end speedup over rotary alone. It remains disabled as an optional
experiment; both measured configurations are supplied.

## What the kernels do

- **Joint Q/K RoPE:** one forward and one backward launch replace the eager
  rotate/negate/concatenate/multiply/add chain. Trig generation, attention,
  projections and dropout are unchanged. The implementation reads both trig
  halves independently, preserves separate product rounding, handles strided
  inputs and missing gradients, and saves only frozen cosine/sine tensors.
  Runtime token counts and flattened strides avoid recompilation for each
  padded length. Unsupported types, layouts and AD modes use eager fallback;
  JIT and real-mode FX tracing fallbacks are tested.
- **Attention residual plus RMSNorm:** one forward produces both the residual
  sum and normalized tensor; backward combines the two gradient branches in
  fp32 before casting the attention gradient to fp16. Only the first residual
  addition and post-attention norm are fused. The final addition, input norm
  and final model norm remain unchanged. Trainable norm weights, hooks,
  checkpointing, inference, unsupported layouts and tracing fall back.

These are original implementations of established operations, not new
algorithms. Residual fusion builds on the team's RMSNorm work. Reference
semantics are pinned Transformers 5.16.1 Qwen2. No extra work is added to
inflate the kernel-time numerator.

## Integration and reproduction

`rotary.ENABLED=True`; `residual_norm.ENABLED=False`. The integrator selects
rotary by adding `"rotary"` to `solution.MODULES`. The registry is deliberately
unchanged on this branch. [INTEGRATION.patch](INTEGRATION.patch) contains the
rotary-only change. [COMBINED_EXPERIMENT.patch](COMBINED_EXPERIMENT.patch) is
an alternative that selects both modules and enables residual fusion; apply
one selection, not both. `rotary.patch()` is idempotent and replaces only
`modeling_qwen2.apply_rotary_pos_emb`.

The optional residual patch owns `Qwen2DecoderLayer.forward` and must not be
stacked with another decoder-forward replacement. It recognizes only the
pinned upstream decoder and upstream or current team norm forward, preserves
module calls/hooks through conservative guards, and exposes `unpatch()`.

Use the existing pinned Kaggle T4 environment. From this checkout:

```bash
python docs/rotary/stage.py --variant rotary --out /kaggle/working/rotary-validation
```

The script creates a new directory with an explicit registry selection and a
SHA-256 manifest, then prints the test/data/harness commands. It never edits
the source checkout. `--variant team` stages the team registry selection with
the current corrected head;
`--variant combined` explicitly enables the residual experiment too. Always
use one visible T4 (`CUDA_VISIBLE_DEVICES=0`), sequential runs, and the
unchanged official harness. Compilation counts; each harness run clears its
own compilation caches. Final shared integration still needs its own repeated
baseline/solution verification.

Omit `--head-correction` on this branch: the correction is already installed.
That flag is retained only for older checkouts containing the original head; its
hash guard deliberately rejects applying the correction twice.

## Validation before the head correction was merged

The FX fallback fix passed **270 T4 tests, 126 skipped**, and all **79 official
regressions**. Its unchanged training arithmetic passed [all ten official
checks](evidence/r5-final-checks.json) in two repeats. The latest
[raw result](evidence/r5-final-result.json) records:

| Configuration | Command times, s | Peak MiB | Minimum eligible share | Eval loss |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 87.29 / 87.09 | 14345 | — | 2.04966 |
| Team + rotary + FX fallback fix | 74.82 / 77.92 | 8367 | 7.0456% | 2.04969 |

Median command time is 76.37 s versus 87.19 s. This is the entire team
configuration's result, not an isolated speed gain from the tracing fix.
[Source manifest](evidence/r5-final-manifest.json), logs and telemetry accompany
the result. This historical configuration still used the defective head; automatic admission
is not proof that every gradient matches. Current-source validation is the
corrected R8 configuration linked above.

## Earlier rotary training evidence

The [initial full pair](evidence/r1-candidate-result.json) recorded original
baseline **84.55 s / 14,345 MiB** versus candidate **75.63 s / 8,367 MiB**,
with held-out losses **2.04964 / 2.04966**.
[All ten automatic checks passed](evidence/r1-candidate-checks.json).
Those totals include all existing team optimizations; they cannot be
attributed to rotary alone. Fresh T4 tests: **223 passed, 67 skipped**, plus
**79 official regressions passed**.

The separate [team reference](evidence/r1-team-result.json) initially ran
in **68.01 s / 8,383 MiB**. That unfavorable rotary timing is retained.
The following alternating repeats used fresh directories and the same sources:

| Run | Command, s | Peak MiB | Eligible share | Eval loss |
| --- | ---: | ---: | ---: | ---: |
| Rotary 1 | 69.27 | 8367 | 8.0118% | 2.04965 |
| Team 1 | 76.80 | 8383 | 6.1042% | 2.04966 |
| Team 2 | 77.54 | 8383 | 6.0784% | 2.04965 |
| Rotary 2 | 76.44 | 8367 | 7.1029% | 2.04969 |

[Raw summaries and telemetry](evidence/abba-summary.json) accompany each
`abba-*-result.json` and run log. The first run began at 53°C; later runs
began at 75–77°C. The median is therefore not presented as a causal speedup.
The later rotary runtime is close to the team times, with **16 MiB lower
peak**. Rotary contributes about **6.3 ms/step** and **3,584 launches over
64 steps**. Minimum observed combined share: **7.10%**. An isolated 8.01%
result does not establish a robust 8–9% margin.

Measured rotary source SHA-256:
`676bd9bc736c24f545c85c64a287c3d98d2dfbfe7e876c5fd334ca783b4f33d2`.

## Residual experiment

[CUDA unit output](evidence/residual-gpu-r2-output.txt): **44 passed,
59 skipped**, including all 29 native CUDA cases. These cover both output
branches, scaled gradients, missing gradients, irregular shapes, compilation
reuse and active Qwen2/LoRA AMP integration. Independent review found and
verified the real-mode FX tracing fallback fix; the final interpreter suite
passed **65 tests**. No remaining correctness blockers were found in that
review.

The complete allocating forward/backward operation was compared with the
current team add-plus-RMSNorm path. At `[4,512,1536]`, candidate/reference
median time was **0.784** in [r1](evidence/residual-micro-r1.json) and **0.876**
in [r2](evidence/residual-micro-r2.json). Smaller shapes were near parity.
These are component measurements, not training speedup or share claims.
The first microbenchmark preceded the FX guard fix; r2 uses the fixed source.

Fixed residual source SHA-256, disabled flag:
`6b453981e8570a9e448accde1422d4fbfc766d330d432ec6082a6f72d6de473c`.
The development snapshot changes only the enable flag and explicit registry
selection; its manifest records those exact bytes.

## Combined repeated official run

The [two-repeat result](evidence/combined-result.json) passed
[all ten official checks](evidence/combined-checks.json), with minimum
eligible kernel share **7.6017%** (second repeat **7.6325%**).

| Configuration | Command times, s | Peak MiB | Held-out loss |
| --- | --- | ---: | ---: |
| Original baseline | 83.41, 86.06 | 14345 | 2.04962 |
| Team + rotary + residual fusion | 76.95, 76.39 | 8367 | 2.04965 |

Median command time is **76.67 s** versus baseline **84.73 s**. These full
solution totals include the team's pre-existing optimizations. The quality
delta is **+0.001%**; task settings, metrics, token counts and rewind passed
the official comparisons. Fresh full solution suite: **267 passed,
126 skipped**; official regressions: **79 passed**.

Each measured solution launched residual forward and backward **1,792 times
each**, or **28 of each per step**. Ordinary RMSNorm launches fell to **29
forward / 28 backward per step**. Residual kernels total about **8.9 ms per
timed step**, but replace already-counted norm work: their whole time is not
an incremental contribution to eligible share. Rotary remains about **6.3
ms/step**. [Detailed summaries](evidence/combined-summary.json) retain the
individual kernel times and counts.

The enabled residual source hash is
`a118b95b25dbef5f519481af4f8d40f149168f051e74402b87bb5fb9557b9162`;
the [snapshot manifest](evidence/combined-source-manifest.json) records all
files, including the explicit registry selection. All 15 protected starter
files match the original bytes. The preceding rotary-only reference was
67.56 s; the [following reference](evidence/rotary-after-result.json) was
71.23 s, at the same 8,367 MiB peak, versus combined 76.39–76.95 s. The
following reference started at 59°C and ended at 78°C; earlier warm rotary
runs were about 76 s. These runs do not isolate a causal speedup or slowdown,
and the two bracketing references favor rotary-only wall time.

**Decision:** recommend rotary-only integration and leave residual fusion
disabled. Its useful computation and admission are demonstrated, but a
training-time benefit over rotary alone is not. The optional combined
configuration is available to the integrator as a coverage/runtime tradeoff.
Neither configuration establishes the requested robust 8–9% share.

## Limits

Measurements are public-task T4 results, not hidden-test or winner predictions.
The branch does not change or certify the team's existing head-loss arithmetic.
Team integration may select different modules, so its final timing, memory,
quality and minimum kernel share must be measured again. Local
`bash tools/check_local.sh` passes, including protected-file comparisons.
