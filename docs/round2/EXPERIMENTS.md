# Round 2 optimization ledger

Status: experiments in progress; no new runtime default selected. The user selected
2026-10-31 06:38–08:38 UTC, at most two hours of owned T4 session time.
Target: at most 54.8 seconds and 4693 MiB under the unchanged official harness.
The opponent result is user-reported; its raw baseline is unavailable here.

The starting candidate is PR #7, runtime `59755b01e594b65ccb5830b97637417b687e50ea`,
source snapshot `2fb6e813fc6fba025655fedcf55f88386d880f96`.
Its previous ABBA notebook reported 59.52 seconds / 4421 MiB and all ten checks.
See `../overnight/FINAL_VALIDATION.md` for that evidence's export limitations.
A fresh same-session control is required for incremental claims.

## Repeat-once attention component

Existing donor idea: iencislennon / Ansar A2,
`375a0a4e8f61d8d0b155cac25ddf70b8883a5adb`. Repeat the full packed K and V once,
then retain four independent causal SDPA calls. The comparison includes copies,
casts, output layout materialization, gradient reductions and temporary allocations.
No new original-kernel credit is claimed.

Pinned T4, FP32 storage under FP16 AMP, Q12/KV2/D128; all Q/K/V require gradients.
Seven alternating blocks, two repetitions per block. Both implementations passed
an independent dense FP32 numerical oracle (maximum relative L2 0.00056544).
GPU traces confirmed efficient forward and backward for every row. Frozen K
retains the existing fallback. Attention dropout was zero in this component.

| Packed rows | Per-row repeat F+B | Repeat-once F+B | Paired median saving | Paired MAD |
|---|---:|---:|---:|---:|
| 519 | 3.0373 ms | 2.4514 ms | 0.5776 ms | 0.0614 ms |
| 666 | 2.8775 ms | 2.4934 ms | 0.3841 ms | 0.0517 ms |
| 1427 | 4.3157 ms | 5.2722 ms | -0.9455 ms | 0.0497 ms |

**Reject unconditional replacement.** A fixed short-input guard is only a candidate:
boundary correctness/timing, prepared quantized LoRA integration, and whole-command
measurements remain pending. Component allocation peaks increased by 6.56, 8.49,
and 18.29 MiB respectively; these are temporary component differences, not predictions
of the full model's external peak.

Raw JSONL, package/GPU metadata, source manifest and paired calculations are in
`evidence/`. The complete local export including six profiler traces has SHA256
`36d8f3a54427ecbc88254295e16670312fe99ba24506204aa882c309b344ed79`.
All 56 source entries matched the exported manifest. A sequential accidental repeat
of the component run occurred while selecting the notebook export cell; the full
raw export is that second run, not a pooled or best-selected result. Earlier partial
notebook output is retained in the local run ledger.

## Startup audit

Reject donor `78c907e` FAST_TRL. Its fake module attributes cause a fresh import to
fail with `AttributeError: type object '_Dummy' has no attribute 'endswith'`.
The 5.4-second claim is not supported by the inspected measurements. Pinned TRL
already lazily resolves trainer classes. Guarded CUDA-0 placement and dataset
construction are separate candidates, preserving the actual trainer and Arrow
normalization. No startup gain is claimed yet.

## Research

Firsthand reports supplied hypotheses, not transferable speed estimates. The
[PyTorch GQA memory report](https://github.com/pytorch/pytorch/issues/154363) and
[pinned SDPA dispatcher](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
support checking actual backend selection. [Reddit training discussion](https://www.reddit.com/r/LocalLLaMA/comments/1inimgo/does_flashattention_with_gqa_degrade_quality_or_i/)
also illustrates why a backend name alone is insufficient evidence.
X/Twitter searches were performed, but relevant originals returned 403 or no
results; no verified X performance claim is used. No third-party kernel code was imported.

## Fresh unchanged control

First solution-only control in the new session: **73.45 s / 4421 MiB**,
training loop 43.46 s, solution installation/import 14.02 s, evaluation 2.04941.
This is the exact starting implementation. It is not a regression attributed to
new code, and is not directly comparable to the prior formal ABBA median. The
previous run reported a 38.10-second training loop. A second cold-compiler control
and the unchanged official short profiler are running to investigate the difference.
The complete 36,733,228-byte export includes logs and adapter, SHA256
`79c1c0a0af5d2bad2ebe2c8c37b70c6f0b3e8ea155b3b9df5488004ebeb38d26`.
Raw result and metrics are in `evidence/`. All 15 protected files remain unchanged.


## Second control and fixed-boundary experiment

Second unchanged solution-only control: **68.39 s / 4421 MiB**, training40.10s,
eval2.04948. Raw83,497,812-byte export includes profile, logs and adapter; SHA256
`758a261d1a467b0b897468de7b70df761ef643e001e120cbac9f64709f02371c`.
Its separate instrumented profile is diagnostic, never score evidence.

The N<=768 repeat-once candidate passed component correctness across
N511/512/666/767/768/769/1024. Candidate/baseline F+B medians were
0.7655/0.8159/0.9053/0.9754/0.9502/1.0165/1.0093 respectively.
Larger inputs deliberately retained the unchanged per-row implementation.
The complete integration gate **failed in its baseline saved-tensor observer**
(`saved=[]`) and cannot qualify the candidate. Instrumentation repair is pending;
no runtime source or tolerance was changed in response. Full failure evidence
SHA256 `e76790c50bc735e0300743f70d2a3a5f74b085868d367cddd0c4cf2685bdce7d`.

## Tied-weight representation rejected

163 CPU tests included five demonstrations of unresolved lifecycle hazards.
The ordinary Accelerate-prepared forward declines compaction, and escaped method
references, aliases and raw mutations cannot safely restore the original semantics.
No GPU trial or measured memory claim. The disabled prototype remains scratch only.


## Startup diagnostics and review

The actual Kaggle datasets4.8.5/native cached Qwen snapshot gate passed all524rows
and264collated batches. Snapshot revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306
was resolved offline from the model cache, not an adapter tokenizer override.
Original/candidate inclusive map call in full setup: **2.6708 →0.5447s**.
Explicit device mapping only changed model setup4.2034→4.1487s; reject as too small
without further evidence. These are diagnostic setup times, not score results.

All control/device/map runs completed64steps. Parameter/buffer dtype, placement,
quantization metadata and aliases matched. Every recorded module had zero forward,
pre-forward, backward and Accelerate hooks; the generic comparator cannot prove
arbitrary hook equivalence. Rewind IDs/counts and cumulative token counts matched
for all64batches; row losses varied by at most.00710(device)/.00437(map). No claim
of bitwise whole-training equality is made. The complete110,222,300-byte export
includes all three adapters/logs; SHA256
`1ecf9626fd5b0c253e3240d526b6c59ea915dc9b3c40917b03fadf242fb881b7`.

Independent review found a real unsupported-formatter gap: same code with changed
global helpers could be accepted and lose added columns. This blocks promotion
until the helper/global provenance guard is repaired and retested. It also found
limitations in tokenizer provenance labeling, hook signatures, variant assertions
and accepted profile flag spellings. These are retained, not waived; the actual
runs used the model snapshot and ordinary full64step arguments.


## Completed repeat comparison and source-provenance repair

The corrected repeat gate passed in 72.57 seconds, including actual prepared NF4
LoRA gradients, optimizer updates, metrics, Rewind and native half saves. Its
subsequent control/repeat/repeat/control command times were 68.49/69.13/66.67/68.19
seconds. The 0.44-second median difference falls within the observed variability;
warmed training also did not improve. Reject this runtime change. All raw logs,
four adapters and source manifests are retained locally; archive SHA256
`e9256d844ed8d9f84f882294619d3f87a3554759ce3b43df02b4f710c4f488af`.

The native-half admission previously used `inspect.getsource` on installed BNB
functions. The task forbids reading unrelated package files. It now fingerprints
already-loaded Python code, defaults and class structure, and declines unsupported
versions. CPU mutation/file-access tests and independent review passed; actual
Python 3.13.15/BNB 0.50.2 T4 admission and bitwise MLP outputs/input gradients passed.
Native arithmetic and runtime-input guard ASTs are unchanged. This is a compliance
repair, with no performance claim. Full final configuration qualification remains
pending. The gate archive SHA256 is
`550ff11a8bd03552682ed099448b9a56e97eb093b5c4a1d190186b03f8155852`.

The separate rematerialization prototype passed isolated T4 numerical checks and
reduced component latency, but increased temporary allocation. It remains rejected
pending one bounded repair; no such runtime is installed in this branch.
