# Frozen MLP experiment on the teammate baseline

Status: the exact selected source passed repeated official T4 qualification and all ten checks. It remains a draft contribution; combination with later main changes is unvalidated.

## Source and ownership

The experiment starts from Decentrathon `b0fa8af4c6493c43781b978055d71d103437cc21`. Its entire solution, protected harness/task/tests and license files match the authenticated read-only localhost commit `61f24f414b5ab91d81784eef99d373b167623a34`. No writes were made to localhost. The later Decentrathon speed fix `fc9b348` is a separate, unmeasured integration target.

The team head, attention, rotary, NF4, RMSNorm, startup module, registry and task settings are retained. The head still scales before FP16 gradient rounding. Starter SwiGLU kernels and all their specializations remain excluded from original-kernel credit. Existing team and donor attribution is preserved.

## Change

Canonical frozen BNB projections already cross FP16 boundaries. The native path writes the unchanged starter forward and backward results directly at those boundaries, avoiding redundant FP32 materialization. The exposed MLP result is explicitly widened to FP32 before residual accumulation. Gate and up retain separate input casts and independent gradient accumulation.

For the existing fixed policy above 1024 rows, a private first-order rematerialization helper replays only gate/up projections; it omits the unnecessary replayed SwiGLU forward. It restores RNG/autocast and releases projection output storage before nested backward. Down projection stays outside replay. Loaded executable provenance checks authenticate the BNB path without reading installed package source files; unsupported calls retain the existing fallback.

## Validation completed

- Exact assembled-source CPU suite: 242 passed, 274 skipped; logs retain skip reasons.
- Portable regression suite: 78 passed, 9 interpreter-only skips; fresh interpreter process: 87 passed.
- T4 prepared-model comparison: two-layer H1536, nested NF4, q/v LoRA, dropout 0.05, real GradScaler, sequential N1024/N1025 optimizer steps. Finite outputs, scaled/unscaled trainable gradients, updates, RNG, metrics and Rewind matched byte-for-byte under explicitly deterministic diagnostic settings. Production harness settings are unchanged.
- All 15 protected file hashes are preserved. Source and integration reviews found no blocking defects within the admitted contract.

## Component evidence and rejected alternatives

The incremental forward-half change passed all 65,536 FP16 encoding sweeps per operand and three scales through the real down boundary. Allocating MLP forward/backward median CUDA savings were 0.952/0.952/1.861/3.851 ms at N1024/1025/2048/4096 relative to the already-modified backward-half/remat stage. These are component results, not gains against the team baseline and not additive whole-command estimates.

Rematerialization alone did not establish a repeatable command-time gain. Backward-half/remat gave 4.54 seconds in one order but only 0.19 seconds in the reverse order; it was not promoted on that evidence. The supervised-logit cache failed its strict CUDA gradient oracle gate before timing; its tolerance was not relaxed. Repeated-K/V storage work was parked when the user prioritized time.

The exact teammate baseline measured 70.17 and 70.37 seconds, both 4421 MiB, in this session. Historical team measurements around 59.66/61.91 seconds used other session/data conditions; they cannot be subtracted from current candidate timings. The fixed target of 54.8 seconds has not been established.

## Matched whole-command comparison

The paired and prepared-model control is `b0fa8af` plus restored canonical-BNB executable guards (the added provenance module and checks in native_half/swiglu_pack); its admitted arithmetic is unchanged. It is distinct from the byte-identical main observations of 70.17/70.37 seconds.

Both orders used the same unchanged 64-step solution command, fresh compiler caches, production dropout and sequential execution in one T4 session. All source manifests and final cleanup/export receipts were verified.

| Order | Guarded teammate control | Candidate | Paired saving | Peak, both |
|---|---:|---:|---:|---:|
| Candidate then control | 72.82 s | 70.68 s | 2.14 s | 4421 MiB |
| Control then candidate | 71.26 s | 70.77 s | 0.49 s | 4421 MiB |

Candidate median is 70.725 s versus control 72.04 s: 1.315 s (1.83%) lower. Training-loop time also fell in both orders: 42.92 to 39.97 s and 42.28 to 41.40 s. The small reverse-order effect limits confidence in the gain magnitude. This passes the predeclared selection gate for full qualification; it does not establish the 54.8-second target or the official all-check result.

Full qualification completed against these exact source bytes; its raw evidence and separate final receipt are exported. Draft PR: https://github.com/alikhan-kai/decentrathon/pull/10. Published runtime commit: `713f143e711146aaac43b557c560a3c1132e37df`.

## Later upstream compatibility

Main advanced to `fc9b348` after source freeze. That commit adds a layer-budget policy and background precompilation; neither is included in this evidence. A bounded source audit reproduced 35 failures among 147 cases in its stated budget regression and found that precompile lacks this candidate's all-half pointer signatures. The budget failure demonstrates an infeasible estimate, not a measured OOM. Composing the branches requires resolving those issues and fresh qualification. No merge was performed.


## Final official qualification

The unchanged repeated harness completed all four 64-step runs and both evaluations. Official solution summary: **70.54 seconds and 4421 MiB**, from individual times **70.77/70.32 seconds**. The harness rounds its floating-point median to two decimals; the arithmetic mean of the displayed run times is 70.545 seconds. Base summary was 90.57 seconds/14345 MiB (88.96/92.18 seconds). The 1.284x whole-solution speedup versus base includes earlier team work and must not be attributed entirely to this MLP change.

All ten checks passed. Evaluation loss was 2.04944 versus base 2.04967. Tests: 381 passed/135 conditional skips in the full solution suite, 87 interpreter passes and a separately retained 79-passed regression transcript. Minimum useful original-kernel share was **10.295486%**, excluding every starter SwiGLU specialization.

The final receipt reports QUALIFIED with all 24 flags true, source integrity, bounded cleanup/export and completion at 2026-10-31T11:26:59.714807Z. The 146375137-byte raw ZIP has SHA-256 `7733edaf33a0e68939b4260247ae528df2e8cb73b9e91c30095426ad19dcdbe3`. Its 58 source-manifest entries were independently rehashed after download. The separate receipt is authoritative for final export status; the ZIP's earlier job-status snapshot predates export.

The **54.8-second target was not reached**. Memory is below 4693 MiB but did not improve relative to the guarded team control. Incremental time evidence remains the modest matched 1.315-second median gain described above. All artifacts are from one owned T4 session; the session was stopped after export. No automatic merge or localhost write occurred.
