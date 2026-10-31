# Final native-hybrid validation and overnight closure

Runtime commit: `59755b01e594b65ccb5830b97637417b687e50ea`, draft [PR #7](https://github.com/alikhan-kai/decentrathon/pull/7), stacked on [PR #5](https://github.com/alikhan-kai/decentrathon/pull/5). No overnight changes were merged. The final solution sources recovered from the notebook bundle match the candidate worktree exactly; all 15 protected files retain their recorded hashes.

## Final T4 measurement

The completed notebook records two base and two solution runs in ABBA order, with normal production settings and a fresh harness measurement cache. All ten official checks passed. This is the exact native-half / large-batch recomputation candidate, not the subsequent unmeasured attention prototypes.

| Run | Command seconds | Peak MiB |
|---|---:|---:|
| Base 1 | 75.09 | 14345 |
| Solution 1 | 59.40 | 4421 |
| Solution 2 | 59.65 | 4421 |
| Base 2 | 78.19 | 14345 |
| Base median, harness rounded | 76.64 | 14345 |
| Solution median, harness rounded | 59.52 | 4421 |

The harness reports **1.2876x speedup**, **3.2447x base/solution memory**, and public evaluation loss **2.04944 versus 2.04969 (-0.012%)**. That is about 22.3% less command time and 69.2% less peak memory than this same-session base. Minimum eligible original-kernel share is printed as **10.2%**, excluding all starter SwiGLU time. The printed precision does not establish an exact unrounded share.

The solution suite passed **298 tests**, with 126 skipped; the official patched Soup regression suite passed **79 tests**. Source-only staging, numerical component gates and deterministic prepared NF4/LoRA gradient/update/metric/Rewind gates were completed earlier and remain archived locally. Actual harness training used its normal settings; deterministic diagnostic settings were confined to comparison workers.

## Evidence recovered after interruption

The original completed notebook is [evidence/hybrid-final-r2.ipynb](evidence/hybrid-final-r2.ipynb), SHA256 `e290a582d7ca83749dd518fb643fe4aa970f4990d14818904796b8fe91abfd70`. Its outputs contain both repeats, the full checks JSON, successful exit codes and the test reports. The JSON/text files beside it are explicitly extracted notebook output, not replacements for the original raw result file.

Printed checks reference result SHA256 `5379fe6a891228b9bf86b1a10d4d4838caf07c606a4dacb440bf63d5ce25d41a` and solution digest `fc4d0ab09649d18205360f10a7a93174bb8b88ddeb56ac3d7b36341cf5ec6134`. The final result file has not been recovered, so its digest and the exact share cannot be independently recomputed here. The notebook records an **81,811,682-byte** final archive, but the stopped session's output browser no longer exposes it. Final raw per-step logs and final adapters therefore remain an export gap. Earlier component and single-run archives are retained under the local overnight evidence directory.

The embedded source bundle was independently decoded and checked against SHA256 `f03e8a0465a2035f0dd763044ba2c62f8daab07b04cd3adb86a5501075b425ad`. Its source manifest, LICENSE and NOTICE are intact. The exporter omitted extensionless notices from the final runtime archive; the source bundle and repository preserve them. No GPU restart or new experiment was used for recovery.

## Decisions and limits

- Retained: corrected scale-before-FP16 head, scoped teammate decoder compaction, T4 efficient-attention dispatch, original rotary, native FP16 MLP saves, and fixed N>1024 branch recomputation.
- Rejected as defaults: all-layer recomputation (78.70 s / 4707 MiB), FP32 restoration before SwiGLU backward (70.83 s / 6447 MiB), and full-length batched attention's component regression.
- Unpromoted: repeat-once attention, short-row batched integration, tied-embedding storage, and teammate startup optimizations. They lack complete candidate GPU validation. Tied-storage implementation was interrupted by the provider usage limit.
- The user-reported competitor 54.8 s / 4693 MiB is not an independently verified matched comparison. This candidate uses less memory but more raw seconds; no winning claim.
- This is a technical draft handoff, not a completed competition submission or public saved Kaggle version.

The authorized overnight cutoff was 2026-10-31 04:56:15 UTC. Provider usage exhaustion interrupted orchestration near 2026-10-30 23:03 UTC. On recovery at 06:31 UTC both recurring schedules were disabled. Kaggle showed the owned draft session **off** before any recovery action; it was not restarted. Its exact stop time is unknown, so compliance with the three-hour elapsed-session cap cannot be certified. Known final child tasks are terminal, with no pending runs. No optimization was resumed after cutoff.
