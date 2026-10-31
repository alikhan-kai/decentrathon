# Soup Kernel Challenge — Decentrathon 6.0

[English](#english) · [Русский](#russian) · [Submission](SUBMISSION.md)

<a id="english"></a>

## English

We optimize Qwen2.5-1.5B-Instruct LoRA training through custom Triton kernels, padding removal and smaller saved activations.
The final-source paired run measured **1.3954× speedup**, **3.2447× base/solution memory ratio**, and **−0.012% eval loss versus base**; all automatic checks 1–10 passed. Source: F below.
Submission Kaggle notebook: [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final).
Final submission commit: [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).

### Results

The fixed task uses Soup 0.77.0, NF4 weights, LoRA rank 16, 256 Kazakh instruction examples, batch size 4, maximum length 512, and one epoch of 64 steps. LoRA targets, dropout, optimizer, schedule and output metrics are preserved. Sources: [task configuration](task/task.yaml), [task data preparation](task/prepare_data.py), and [rules](docs/RULES.md).

**Final submission-source run (F), public rows, seed 1:**

| Measurement | Base Soup | Solution | Comparison |
|---|---:|---:|---|
| Whole command, s | 83.25 | 59.66 | Speedup 1.3954× |
| Peak GPU memory, MiB | 14345 | 4421 | Base/solution 3.2447× |
| Held-out loss | 2.04966 | 2.04942 | −0.012% |
| Own-kernel share, lowest repeat | Not applicable | 10.0% | Required ≥5% |

Source **F**: tracked [result.json](docs/measurements/submission_632851f/result.json), [checks.json](docs/measurements/submission_632851f/checks.json) and [notebook log](docs/measurements/submission_632851f/decentrathon-soup-kernel-final.log). The log identifies submission SHA `632851f0f2c3d202ae898d07b9b7b050eed61699`. All automatic checks 1–10 passed, including 79 Soup regressions. The full solution suite still reports 298 passed, 126 skipped and 5 failures. The same PEFT/torchao environment issue remains visible in the notebook log.

F uses the same recorded versions and solution SHA-256 as P/R below. Its ABBA command times are **85.13 / 59.36 / 59.97 / 81.36 s**. Result SHA-256: `77bb7816be769424343056cd1c4f38f1d9c67a5828452890bf20b15ee1a89845`. The runtime directories `solution/`, `harness/`, `task/`, `tests/` and `tools/` are unchanged between R and the submission SHA. F evidence and the configured notebook were recorded in later documentation commits; their existence does not change the benchmarked SHA.

**Earlier comparisons P and R, retained separately:**

| Dataset / measurement | Base Soup | Solution | Comparison |
|---|---:|---:|---|
| Public, seed 1: whole command, s | 89.66 | 67.08 | Speedup 1.3366× |
| Public: peak GPU memory, MiB | 14345 | 4421 | Base/solution 3.2447× |
| Public: held-out loss | 2.04967 | 2.04944 | −0.011% |
| Public: own-kernel share, lowest repeat | Not applicable | 8.9% | Required ≥5% |
| Other rows, seed 7: whole command, s | 81.68 | 61.91 | Speedup 1.3193× |
| Other rows: peak GPU memory, MiB | 14351 | 4405 | Base/solution 3.2579× |
| Other rows: held-out loss | 2.11513 | 2.11550 | +0.017% |
| Other rows: own-kernel share, lowest repeat | Not applicable | 10.0% | Required ≥5% |

**Evidence identifiers used throughout this README:**

- **P:** public `full_profile` at [e0eed88][public-commit]. Local raw files: `docs/measurements/full_profile_final_e0eed88/runs/result.json` and `docs/measurements/full_profile_final_e0eed88/runs/checks.json`. Tracked records: [experiment notes](notes/experiments.md) and [stage status](docs/measurements/full_profile_final_e0eed88/status.txt).
- **R:** additional-row run at [952f20a][robust-commit]. Local raw files: `docs/measurements/robust_final_952f20a/runs_robust/result.json` and `docs/measurements/robust_final_952f20a/runs_robust/checks.json`. Tracked records: [experiment notes](notes/experiments.md) and [stage status](docs/measurements/robust_final_952f20a/status.txt).

The raw P/R JSON files and training logs exist in the local evidence checkout, but are **not tracked by Git**: the nested `runs/` and `runs_robust/` directories match ignore rules. Their paths above are provenance labels, not public download links. The saved public notebook and its output archive must make them available before submission. The tables here transcribe those files; the tracked notes retain their summary. Publication work is recorded in [README_TODO](docs/README_TODO.md).

P and R use the same `solution/` source; the intervening change affects tools. Both checks files record solution SHA-256 `a223505358e52e36934f5c38f33a18c5e0046614170d27f145d517516ec17800`. P result SHA-256: `a36ee93dcd3ab219a5860eb8b5d33e419601080239f2a844cc9bd640fceb1969`. R result SHA-256: `88456f75fb533c644423e7a26954bebbf5b5a000f526e9f005198b48a46db1a9`.

Each comparison is from one `run_task.py --repeats 2` invocation: **base → solution → solution → base**, on one visible Tesla T4. P command times in that order were **89.09 / 66.83 / 67.33 / 90.24 s**; R times were **79.63 / 62.43 / 61.40 / 83.73 s**. Wall time is the median for each side; memory is the maximum peak for each side. Ratios are copied from `summary.ratios`, without additional favorable rounding. Evaluation uses the first successful adapter on each side. Sources: P/R and [run_task.py](harness/run_task.py).

Public training rows are 0–255 and evaluation rows are 40000–40255. R uses training rows 1000–1255 and evaluation rows 41000–41255, with seed 7, from the same dataset revision. R is a local robustness experiment, not the jury's hidden evaluation. Sources: [prepare_data.py](task/prepare_data.py), [make_robust_data.py](tools/make_robust_data.py), [notebook builder](tools/build_notebook.py).

| Recorded environment, identical in P/R | Version |
|---|---|
| Python | 3.13.15 |
| Soup | 0.77.0 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Transformers / PEFT / TRL | 5.16.1 / 0.20.0 / 0.29.1 |
| bitsandbytes / Accelerate / datasets | 0.50.2 / 1.14.0 / 4.8.5 |
| GPU / driver | Tesla T4 / 580.178.04 |

Versions come from P/R `versions`. The session provided two T4 GPUs, but the harness exposed only one for training. Compilation counts toward wall time and compiler caches are reset per run. Memory is sampled with `nvidia-smi`; own kernels are timed with CUDA events from step 3. The starter SwiGLU kernels are excluded from the eligible share. See [measurement code](harness/_train.py) and [common settings](harness/common.py).

| Automatic check | P | R |
|---|---|---|
| 1. All base and solution runs finish | PASS | PASS |
| 2. No prohibited kernel libraries or runtime installations | PASS | PASS |
| 3. Own Triton kernels execute in training | PASS | PASS |
| 4. Own-kernel share ≥5% | PASS: 8.9% | PASS: 10.0% |
| 5. Same task, adapter structure, metrics and rewind | PASS | PASS |
| 6. Training loss curve within tolerance | PASS | PASS |
| 7. Eval loss no more than 0.5% above base | PASS: −0.011% | PASS: +0.017% |
| 8. Time and memory no more than 10% worse | PASS | PASS |
| 9. Clean T4 measurement | PASS | PASS |
| 10. Soup regression suite | PASS: 79 tests | PASS: 79 tests |

Source: P/R `checks.json`; thresholds: [check_rules.py](harness/check_rules.py). Automatic passes do not replace jury code review. The submission-source run is recorded separately as F above.

### What changed

The baseline retains large vocabulary logits and repeatedly moves intermediate tensors through GPU memory.
We compute head loss and metrics in token chunks and recompute logits for backward.
We fuse NF4 unpacking, RMSNorm and rotary operations with custom Triton code.
We remove right padding inside each existing batch while keeping attention isolated per original example.
For supported MLP calls, we retain native FP16 activations or recompute the gate/up branch.
We also reduce dataset preparation and device-placement overhead.
The full-run gain is measured for the combined solution; individual component gains are not additive.

In the table, `mq` means `transformers.models.qwen2.modeling_qwen2`. Benefits describe the mechanism; an isolated final-candidate speed or memory saving for each module is **unknown**. It requires an otherwise identical on/off ABBA comparison in one T4 session.

| Module | Replacement / integration point | Original Triton work launched in P/R | Benefit and source |
|---|---|---|---|
| [head_loss.py](solution/head_loss.py) | `mq.Qwen2ForCausalLM.forward`; `trl.SFTTrainer.compute_loss` | Yes: `head_loss_kernel` | Avoid full retained logits; compute loss, entropy and accuracy together. Chunking idea: Soup [stream_loss_chunk][soup-chunks]. |
| [nf4.py](solution/nf4.py) | `bitsandbytes.backends.cuda.ops._dequantize_4bit_impl` | Yes: `nf4_dequant_kernel` | Faster weight unpacking. NF4 format: [QLoRA][qlora]; layout/reference: [bitsandbytes][bnb-source]. |
| [rmsnorm.py](solution/rmsnorm.py) | `mq.Qwen2RMSNorm.forward` | Yes: `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | Fuse reductions and elementwise operations. Semantics: [Qwen2 implementation][qwen-source]. |
| [rotary.py](solution/rotary.py) | `mq.apply_rotary_pos_emb` | Yes: `ember_rope_joint` | Joint Q/K forward or backward launch. Qwen2 semantics; research in [rotary notes](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/RESEARCH.md). |
| [packed.py](solution/packed.py) | `mq.Qwen2Model.forward`; `mq.Qwen2Attention.forward` | No additional kernel | Remove padding work without attention across examples. Team packed decoder; [integration provenance](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). |
| [attention_adapter.py](solution/attention_adapter.py), [attention_gqa.py](solution/attention_gqa.py) | Packed-row caller of PyTorch `scaled_dot_product_attention` | No | Explicit K/V repetition for eligible T4 calls; retains HF fallback. Team experiment in the integrated candidate. |
| [swiglu_pack.py](solution/swiglu_pack.py), [native_half.py](solution/native_half.py), [hybrid_mlp.py](solution/hybrid_mlp.py) | Guarded replacement of `mq.Qwen2MLP.forward` | No original kernel; reuse starter SwiGLU | FP16 saved activations or branch recomputation. Team `perf/packed-native-memory` work; [provenance](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). |
| [overhead.py](solution/overhead.py) | `soup_cli.trainer.sft.resolve_device_map`; `soup_cli.trainer.sft._map_text_sft_rows` | No | Direct GPU placement and dataset construction in memory. Alikhan-kai; [merge record](docs/MERGE_OVERHEAD.md). |
| [swiglu.py](solution/swiglu.py) | Initial `mq.Qwen2MLP.forward` replacement | Starter `example_swiglu_fwd_kernel` / `example_swiglu_bwd_kernel`; **not ours** | Fused MLP activation from the organizer kit. No originality or eligible-share credit claimed. |
| [lora.py](solution/lora.py), [residual_norm.py](solution/residual_norm.py) | Disabled: LoRA epilogue stub; optional decoder residual/norm fusion | Neither contributes eligible launches in P/R | LoRA optimization is unfinished; residual fusion has no demonstrated whole-training gain over rotary alone. [Residual evidence](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md). |

**Head kernel.** `head_loss_kernel` scans the vocabulary in blocks and uses a stable online softmax reduction. Forward emits per-token cross-entropy, entropy and argmax correctness. Backward recomputes chunks, applies the incoming loss scale in FP32, then stores FP16 logit gradients. The frozen head needs no weight gradient. `CHUNK=512` is in [head_loss.py](solution/head_loss.py). Tests compare metrics and hidden gradients against PyTorch in [test_head_loss.py](solution/test_head_loss.py); [fallback tests](solution/test_head_fallback.py) cover unsupported trainer/head behavior. The [head-correction record](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/head_correction/README.md) documents the scale-before-rounding fix and additional gradient tests.

**NF4 kernel.** `nf4_dequant_kernel` extracts the high and low nibbles of each byte, looks up the codebook, applies the block scale and writes FP16 weights. Nested scale decoding remains upstream. Supported blocks contain 64 weights; other cases use bitsandbytes. [test_nf4.py](solution/test_nf4.py) checks decoded values, odd lengths, dispatch and matmul gradients against bitsandbytes. [Experiment notes](notes/experiments.md) record bit-exact unit results for the earlier isolated NF4 run.

**RMSNorm kernels.** `rmsnorm_fwd_kernel` computes the row RMS, reciprocal scale and weighted output in one launch. `rmsnorm_bwd_kernel` computes the input gradient with FP32 reductions. A requested weight gradient uses PyTorch. [test_rmsnorm.py](solution/test_rmsnorm.py) compares forward/backward with the original module and exercises fallbacks.

**Rotary kernel.** `ember_rope_joint` processes Q and K in one launch per direction, preserving the original multiply-rounding boundaries. It supports strides and independent missing gradients; trig generation remains upstream. `_rope_rotate_block` is a device helper, not an additional separately timed launch. [test_rotary.py](solution/test_rotary.py) covers parity, layouts, gradient scales, shape reuse, tracing and fallback behavior.

### Integration

The harness calls `solution.apply()` before model and trainer construction. The registry order is `swiglu → head_loss → rmsnorm → nf4 → lora → rotary → overhead`; `lora.ENABLED=False`, so its patch is skipped. Then [stage_adapter.py](solution/stage_adapter.py) installs `packed` and, with `ENABLE_SAVE=True`, the saved-activation MLP path. Sources: [registry](solution/__init__.py), [stage options](solution/stage_options.py).

`swiglu_pack.ENABLED=False` and `native_half.ENABLED=False` do not disable this explicit installer. `residual_norm` is not in the registry. Importing `solution` alone does not patch the model; calling `apply()` does.

Guards check devices, dtypes, shapes, modes and supported model behavior. Unsupported calls generally use the saved original implementation. Attention dispatch requires the recorded PyTorch/Transformers versions and a T4. The MLP installer requires the pinned bitsandbytes/Transformers versions, known starter bytes and canonical frozen projections. Its rejection prints `SAVED_ACTIVATION_PACKING DISABLED` and leaves the previous MLP path active. This is a local fallback, not a global compatibility guarantee. Some import/API failures raise an error; rotary reports an unavailable patch. After a version or code change, use a fresh process and rerun the full checks.

### Reproduction

**Public Kaggle run:** [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final).
**Final commit:** [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).

The saved submission notebook must already contain the repository URL and final `COMMIT`, and must run with **Run All** without editing cells. The notebook URL and final SHA have been supplied. The configured [submission notebook](submission/kaggle_notebook.ipynb) pins that SHA and runs two repeats. Public visibility and a version-specific URL remain to be verified: the supplied URL has no saved-version ID, the page could not be inspected through the web tool, and local [metadata](submission/kernel-metadata.json) still says `is_private: true`. The repository's [organizer template](templates/kaggle_notebook.ipynb) still contains an unset repository/commit, and defaults to one repeat. It is not the ready-to-submit notebook.

To rerun the submission notebook or prepare another saved version:

1. Use a Kaggle notebook with **GPU T4 x2** and Internet enabled; the harness uses only one GPU. Keep the GPUs otherwise idle.
2. Open the supplied Kaggle notebook and use **Run All**. Its repository copy is [submission/kaggle_notebook.ipynb](submission/kaggle_notebook.ipynb), with the repository URL, final SHA and `--repeats 2` already set. For a fresh Kaggle copy, import this configured notebook. Do not edit the protected organizer template.
3. Alternatively, create a template-based notebook through [build_notebook.py](tools/build_notebook.py), using its `--commit` option and `--mode full_profile`. Its stages record failures in `status.txt` and continue, so inspect every status rather than relying on Kaggle's overall completion label.
4. Save a public version with **Save & Run All**. Verify the logged commit and all checks, and publish the output archive. Put the same saved-version URL in this README and `SUBMISSION.md`.

**Linux/CUDA reproduction of the measured source.** The following checkout pins the supplied final submission SHA measured in F. Start with a CUDA-enabled PyTorch/Triton environment matching the version table. macOS without CUDA cannot reproduce the T4 numbers.

```bash
git clone https://github.com/alikhan-kai/decentrathon.git
cd decentrathon
git checkout 632851f0f2c3d202ae898d07b9b7b050eed61699
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" \
  "accelerate==1.14.0" "datasets==4.8.5" pytest
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Review a failed test command before treating the run as validated; see the known environment failure below. The measured P/R training environment did pass the official checks. Package installation belongs to environment setup, outside the measured training process.

To reproduce R separately:

```bash
python tools/make_robust_data.py --out data_robust
python harness/run_task.py --repeats 2 --seed 7 \
  --train-file data_robust/train.jsonl --eval-file data_robust/eval.jsonl \
  --out runs_robust
python harness/check_rules.py --result runs_robust/result.json
```

Expected output: `runs/result.json`, `runs/checks.json`, per-run `train.log`, metrics and saved adapters; R uses `runs_robust/`. Optional profiling: add `--profile` to `run_task.py`. `python tools/summarize.py runs` prints the measurement tables. Measured individual command durations are listed above. **I do not know the complete full-profile workflow duration** from the retained P evidence; it needs a timed clean full-profile run including installation, downloads, tests and evaluation. The separately recorded F notebook is a non-profiled submission run. The organizer template's approximate 10-minute estimate describes its simpler run, not a measurement of this full-profile workflow.

### Correctness validation

Kernel tests compare forward values and backward gradients with PyTorch or bitsandbytes using explicit per-test tolerances. They cover masked targets, chunk tails, layouts and fallback paths. The shared [kcheck.py](solution/kcheck.py) is a helper; individual tests define their own appropriate tolerances.

The recorded P suite had **298 passed, 126 skipped, 5 failed**. The five failures were PEFT `ImportError`s involving installed `torchao 0.10.0`, where PEFT required at least `0.16.0`. They affect three head-fallback cases and two residual/LoRA integration cases. Source: local `kaggle/out_full_profile_20261031_040542/out/pytest_solution.log`, summarized in [experiment notes](notes/experiments.md); [tracked status](docs/measurements/full_profile_final_e0eed88/status.txt) records the failed stage. This is not a fully passing solution suite. The separate **79 Soup regression tests passed** in both P and R.

Checks 5–7 compare against the base within the same run. In P, the solution's first-step loss gap is **0.03%** and its mean step-loss gap is **0.05%**. In R these are **0.00%** and **0.05%**, as rounded by the checks file. Limits are **1%** and **3%**. Eval-loss changes are **−0.011%** and **+0.017%**, against a **+0.5%** upper limit. Check 5 reports matching task/output structure and metrics within tolerance. Sources: P/R checks; [comparison definitions](harness/check_rules.py).

R rewind mean absolute relative row-loss gaps against `base_1` are **0.225% / 0.221%** for the solution repeats. A short row with **7 answer tokens** reaches **7.04%** in the first repeat. Check 5 limits the **mean** relative gap to **3%**, not every row's maximum. Source: R `runs.*.rewind`, [notes](notes/experiments.md), and `compare_outputs` in the checker. P's separate rewind files were not collected, although its result JSON contains the rewind summaries used by the checks.

### Limitations and unsuccessful work

- **Specific workload.** Results apply to the fixed Qwen/T4 FP16 task and recorded libraries. Other architectures, BF16 execution and GPU types need new measurements. Unsupported paths may fall back or fail to install.
- **Starter hash guard.** MLP storage optimization accepts only two known starter sources. The current `swiglu.py` SHA-256 is `e1dc639838c4c46ea0ed0aec3f0d904d360c92a91c601b943c9cfef85ca21c64`; even a source-only change can disable it. See [guard code](solution/swiglu_pack.py).
- **Attention fallback.** Each P solution run reports **6912 dispatched / 256 fallback calls**, or **256/7168 = 3.5714% fallback**. The fallback occurs at layer 0 with `k-no-grad`. This is a fraction of calls, not GPU time. Source: P solution `train.log`, `COMBINED_STAGE_STATS`; counts also appear in [notes](notes/experiments.md).
- **MLP tradeoff.** The fixed threshold is **1024 token rows**. P records **1344 native-FP16 calls and 448 recomputed calls**. Recomputing saves memory at the cost of more work. It is restricted to first-order eager training. Sources: [installer](solution/swiglu_pack.py), [hybrid helper](solution/hybrid_mlp.py), P stage statistics.
- **Numerical equivalence has limits.** Packing changes dropout RNG assignment. Short-row losses can differ more than the mean check threshold. Passing task checks does not establish bitwise equality or general gradient correctness; an earlier head scale-before-cast defect required a separate fix. See [correction evidence](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/head_correction/README.md).
- **Environment test failures remain.** The five PEFT/torchao failures above are recorded, not hidden. A clean full solution-suite rerun is still needed. The official training checks are a separate test surface.
- **No isolated final-module speedups.** Earlier component profiles in [FINDINGS](docs/FINDINGS.md) belong to older source snapshots. Cross-session T4 timings are indicative only. Thermal/session noise is documented in [rotary experiments](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md); do not sum component gains or compare P's wall time directly with R's.
- **Unfinished/disabled work.** `lora.py` is a stub. Residual/norm fusion remains disabled because a whole-training improvement over rotary alone was not demonstrated. The private batched-attention prototype was excluded from the integrated candidate. Sources: [LoRA file](solution/lora.py), [rotary evidence](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md), [candidate record](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md).
- **Release metadata is incomplete.** The final SHA and notebook URL are supplied, and F evidence is tracked. Public visibility, a saved-version URL, the official roster and media links still need verification or completion. Historical P/R raw evidence remains locally ignored. Recorded results do not certify the hidden jury run. Historical research links are pinned to their earlier Git snapshot because the submission SHA removed those documents.

### Next steps

| Experiment | Evidence and expected benefit | Required measurement |
|---|---|---|
| Reduce NF4 unpacking cost | It accounts for 6.3609–6.4231% of the P timed loop, derived from `100 × own_kernel_ms / (1000 × timed_loop_s)`. Potential time benefit; exact gain unknown. | One-session ABBA, including compilation, kernel parity and all checks. |
| Reduce attention fallbacks | P has 256 fallback calls per solution run. Potential time benefit; exact gain unknown and call share is not time share. | Profile layer 0 and validate Q/K/V gradients before broadening guards. |
| Revisit the fixed MLP recomputation threshold | P contains 448 recomputed MLP calls. Potential memory/time tradeoff; exact gain unknown. | Compare fixed policies on both datasets, preserving RNG and all task settings. |
| Optimize LoRA scale/cast/add | Older [module profile](docs/FINDINGS.md) attributes 9.7 ms/step to the PEFT epilogue. This is an old-workload opportunity, not a current gain prediction. | Implement the missing epilogue without changing dropout, then profile the final workload and run ABBA. |

### Credits and sources

Official team name: TODO(TEAM_NAME).
Confirmed roster, roles, GitHub/Kaggle accounts and captain contact: TODO(TEAM_ROSTER_AND_CAPTAIN_CONTACT).

The following are **code contribution credits**, established from Git history and provenance records; they do not substitute for the confirmed competition roster.

| Contributor | Recorded work | Evidence |
|---|---|---|
| Ansar / `iencislennon` | Head loss, NF4, RMSNorm, packed decoder, integration and measurements | Commits `69a595d`, `ca90121`, `c0cc1c8`; [candidate provenance](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md) |
| `stritphatographir` | Rotary work, corrected head gradients and guards, native/hybrid integration | Commits `05543d7`, `5b4dd1f`, `59755b0`; [rotary record](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md) |
| `Alikhan-kai` | Startup device placement and dataset preparation | Commit `44d1a47`; [overhead merge record](docs/MERGE_OVERHEAD.md) |

- **Makazhan Alpamys and Soup contributors:** [Soup][soup] and the [challenge starter kit][starter]. The head-chunking idea follows Soup's `stream_loss_chunk`; our head Triton kernel is an original implementation. The source link points to upstream `main`; the benchmark package is pinned to Soup 0.77.0. See [source][soup-chunks] and [NOTICE](NOTICE).
- **Starter SwiGLU:** organizer code retained in [swiglu.py](solution/swiglu.py). We claim **no original-kernel credit** for it, its renamed variants, activation-save wrappers, recomputation wrappers or `do_not_specialize` decorators. Its time is excluded from check 4.
- **Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer:** [QLoRA][qlora] defines NF4. The bitsandbytes contributors provide the [format implementation and reference behavior][bnb-source]. We wrote the Triton unpacking kernel.
- **Hugging Face Transformers contributors:** the [pinned Qwen2 implementation][qwen-source] defines RMSNorm, RoPE and attention semantics. The solution uses PyTorch SDPA for attention; it does not claim a custom attention kernel.
- **Team `perf/packed-native-memory` work:** packed decoder and native/hybrid activation integration are recorded in [CANDIDATE](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md), including donor `3e48f6a` and integration `59755b0`. Team/Codex assistance is acknowledged in that record. [Research notes](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/RESEARCH.md) retain additional leads and primary references.

Code and starter tests are Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The [Qwen model][model] is Apache-2.0. The [Kazakh dataset][dataset] card lists MIT and describes its derivation from [Stanford Alpaca][alpaca], whose card lists CC BY-NC 4.0. Data and model weights are downloaded at runtime, not bundled in this submission.

### Repository layout

```text
solution/    Runtime patches, Triton kernels and component tests; jury copies this directory.
task/        Fixed training configuration and public data preparation.
harness/     Organizer measurement, adapter evaluation, checks and scoring.
tests/       Soup regression tests with optional solution activation.
tools/       Notebook generation, local lint, profiling and result summaries.
kaggle/      Development notebook and metadata; downloaded outputs are ignored.
docs/        Research, provenance, measurement records and submission follow-up.
notes/       Chronological experiment summaries.
templates/   Organizer notebook and submission templates.
.github/     Repository CI and pull-request template.
```

---

<a id="russian"></a>

## Русский

Проект ускоряет обучение LoRA для Qwen2.5-1.5B-Instruct с помощью своих Triton-кернелов, удаления padding и экономии памяти активаций.
В парном замере финального кода: **ускорение 1.3954×**, **отношение памяти базы к решению 3.2447×**, **eval loss −0.012% относительно базы**; автоматические проверки 1–10 прошли. Источник: F ниже.
Kaggle-ноутбук для сдачи: [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final).
Финальный коммит для сдачи: [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).

### Результаты

Фиксированная задача: Soup 0.77.0, веса NF4, LoRA ранга 16, 256 казахских инструкций, батч 4, длина до 512 токенов, одна эпоха и 64 шага. Сохраняются настройки LoRA и dropout, оптимизатор, расписание и выходные метрики. Источники: [конфигурация](task/task.yaml), [подготовка данных](task/prepare_data.py), [правила](docs/RULES.md).

**Замер финального кода для сдачи (F), публичные строки, seed 1:**

| Показатель | База Soup | Решение | Сравнение |
|---|---:|---:|---|
| Вся команда, с | 83.25 | 59.66 | Ускорение 1.3954× |
| Пик памяти GPU, MiB | 14345 | 4421 | База/решение 3.2447× |
| Eval loss | 2.04966 | 2.04942 | −0.012% |
| Доля своих кернелов, худший повтор | Не применяется | 10.0% | Требуется ≥5% |

Источник **F**: отслеживаемые [result.json](docs/measurements/submission_632851f/result.json), [checks.json](docs/measurements/submission_632851f/checks.json) и [лог ноутбука](docs/measurements/submission_632851f/decentrathon-soup-kernel-final.log). Лог подтверждает запуск SHA `632851f0f2c3d202ae898d07b9b7b050eed61699`. Все автоматические проверки 1–10 прошли, включая 79 регрессионных тестов Soup. Полный набор решения по-прежнему содержит 298 passed, 126 skipped и 5 ошибок. Та же несовместимость PEFT/torchao видна в логе ноутбука.

Версии окружения и SHA-256 решения F совпадают с P/R ниже. Времена ABBA: **85.13 / 59.36 / 59.97 / 81.36 с**. SHA-256 результата: `77bb7816be769424343056cd1c4f38f1d9c67a5828452890bf20b15ee1a89845`. Папки `solution/`, `harness/`, `task/`, `tests/` и `tools/` между R и сдаваемым SHA не изменились. Доказательства F и настроенный ноутбук добавлены последующими коммитами документации; это не меняет SHA измеренного кода.

**Ранние сравнения P и R сохранены отдельно:**

| Данные / показатель | База Soup | Решение | Сравнение |
|---|---:|---:|---|
| Публичные строки, seed 1: вся команда, с | 89.66 | 67.08 | Ускорение 1.3366× |
| Публичные строки: пик памяти GPU, MiB | 14345 | 4421 | База/решение 3.2447× |
| Публичные строки: eval loss | 2.04967 | 2.04944 | −0.011% |
| Публичные строки: доля своих кернелов, худший повтор | Не применяется | 8.9% | Требуется ≥5% |
| Другие строки, seed 7: вся команда, с | 81.68 | 61.91 | Ускорение 1.3193× |
| Другие строки: пик памяти GPU, MiB | 14351 | 4405 | База/решение 3.2579× |
| Другие строки: eval loss | 2.11513 | 2.11550 | +0.017% |
| Другие строки: доля своих кернелов, худший повтор | Не применяется | 10.0% | Требуется ≥5% |

**Источники P и R:**

- **P:** публичный `full_profile` на [e0eed88][public-commit]. Локальные исходные файлы: `docs/measurements/full_profile_final_e0eed88/runs/result.json` и `docs/measurements/full_profile_final_e0eed88/runs/checks.json`. В Git сохранены [журнал экспериментов](notes/experiments.md) и [статус этапов](docs/measurements/full_profile_final_e0eed88/status.txt).
- **R:** проверка на других строках на [952f20a][robust-commit]. Локальные исходные файлы: `docs/measurements/robust_final_952f20a/runs_robust/result.json` и `docs/measurements/robust_final_952f20a/runs_robust/checks.json`. В Git сохранены [журнал экспериментов](notes/experiments.md) и [статус этапов](docs/measurements/robust_final_952f20a/status.txt).

Исходные JSON и логи P/R есть локально, но **не отслеживаются Git**: вложенные каталоги `runs/` и `runs_robust/` попадают под правила игнорирования. Эти пути обозначают происхождение чисел, а не публичные ссылки для скачивания. Перед сдачей файлы должны быть доступны в выходных данных публичной версии ноутбука. Таблицы переписаны из этих JSON; сводка также есть в отслеживаемом журнале. Задачи публикации перечислены в [README_TODO](docs/README_TODO.md).

В P и R одинаковая папка `solution/`; промежуточный коммит меняет инструменты. Оба `checks.json` записали SHA-256 решения `a223505358e52e36934f5c38f33a18c5e0046614170d27f145d517516ec17800`. SHA-256 результата P: `a36ee93dcd3ab219a5860eb8b5d33e419601080239f2a844cc9bd640fceb1969`. Для R: `88456f75fb533c644423e7a26954bebbf5b5a000f526e9f005198b48a46db1a9`.

Каждое сравнение сделано одним вызовом `run_task.py --repeats 2`: **база → решение → решение → база**, одна видимая Tesla T4. Время команд в P в этом порядке: **89.09 / 66.83 / 67.33 / 90.24 с**; в R: **79.63 / 62.43 / 61.40 / 83.73 с**. В таблице времени используется медиана каждой стороны, памяти — максимальный пик. Отношения взяты непосредственно из `summary.ratios`. Для оценки качества берётся первый успешный адаптер каждой стороны. Источники: P/R, [run_task.py](harness/run_task.py).

Публичные строки обучения: 0–255, оценки: 40000–40255. R использует строки обучения 1000–1255 и оценки 41000–41255 с seed 7 и той же ревизией датасета. Это дополнительный эксперимент, а не скрытый тест жюри. Источники: [prepare_data.py](task/prepare_data.py), [make_robust_data.py](tools/make_robust_data.py), [сборщик ноутбука](tools/build_notebook.py).

| Одинаковое окружение P/R | Версия |
|---|---|
| Python | 3.13.15 |
| Soup | 0.77.0 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Transformers / PEFT / TRL | 5.16.1 / 0.20.0 / 0.29.1 |
| bitsandbytes / Accelerate / datasets | 0.50.2 / 1.14.0 / 4.8.5 |
| GPU / драйвер | Tesla T4 / 580.178.04 |

Источник версий — поле `versions` в P/R. В сессии были две T4, но обучению доступна только одна. Компиляция входит во время команды, кэши компилятора очищаются перед каждым прогоном. Память измеряет `nvidia-smi`, время своих кернелов — CUDA events начиная с шага 3. Кернелы примера SwiGLU в зачётную долю не входят. См. [измерительный код](harness/_train.py) и [настройки](harness/common.py).

| Автоматическая проверка | P | R |
|---|---|---|
| 1. Завершение всех запусков | PASS | PASS |
| 2. Нет запрещённых библиотек и установки пакетов во время прогона | PASS | PASS |
| 3. Свои Triton-кернелы работают в обучении | PASS | PASS |
| 4. Доля своих кернелов ≥5% | PASS: 8.9% | PASS: 10.0% |
| 5. Та же задача, структура адаптера, метрики и rewind | PASS | PASS |
| 6. Кривая ошибки обучения в пределах допуска | PASS | PASS |
| 7. Eval loss не выше базы более чем на 0.5% | PASS: −0.011% | PASS: +0.017% |
| 8. Время и память не хуже более чем на 10% | PASS | PASS |
| 9. Чистый замер на T4 | PASS | PASS |
| 10. Регрессионные тесты Soup | PASS: 79 тестов | PASS: 79 тестов |

Источник: `checks.json` P/R; пороги: [check_rules.py](harness/check_rules.py). Прохождение автоматики не заменяет код-ревью жюри. Запуск сдаваемого кода отдельно зафиксирован как F выше.

### Что сделано

База сохраняет большие таблицы оценок словаря и многократно перемещает промежуточные массивы через память GPU.
Мы считаем loss и метрики головы порциями токенов и пересчитываем logits для backward.
Свои Triton-кернелы объединяют операции распаковки NF4, RMSNorm и RoPE.
Padding удаляется внутри исходного батча; внимание между разными примерами не возникает.
Для подходящих MLP-вызовов сохраняются активации FP16 либо пересчитывается ветка gate/up.
Дополнительно сокращены расходы подготовки датасета и размещения модели.
Итог измерен для всей комбинации; выигрыши отдельных частей нельзя складывать.

В таблице `mq` означает `transformers.models.qwen2.modeling_qwen2`. **Не знаю отдельный выигрыш каждого модуля в финальной комбинации**: для этого нужен ABBA с включением/выключением только этого модуля в одной сессии T4. Указанная польза описывает механизм.

| Модуль | Точка замены | Свой Triton в P/R | Польза и источник |
|---|---|---|---|
| [head_loss.py](solution/head_loss.py) | `mq.Qwen2ForCausalLM.forward`, `trl.SFTTrainer.compute_loss` | `head_loss_kernel` | Не хранить все logits, совместно считать loss/entropy/accuracy. Идея порций: [Soup stream_loss_chunk][soup-chunks]. |
| [nf4.py](solution/nf4.py) | `bitsandbytes.backends.cuda.ops._dequantize_4bit_impl` | `nf4_dequant_kernel` | Ускорить распаковку весов. Формат: [QLoRA][qlora], семантика: [bitsandbytes][bnb-source]. |
| [rmsnorm.py](solution/rmsnorm.py) | `mq.Qwen2RMSNorm.forward` | `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | Объединить редукции и поэлементные операции по семантике [Qwen2][qwen-source]. |
| [rotary.py](solution/rotary.py) | `mq.apply_rotary_pos_emb` | `ember_rope_joint` | Общий запуск для Q/K, forward и backward. [Исследование](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/RESEARCH.md). |
| [packed.py](solution/packed.py) | `mq.Qwen2Model.forward`, `mq.Qwen2Attention.forward` | Нет нового кернела | Удалить работу по padding, сохранив независимость примеров. [Происхождение](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). |
| [attention_adapter.py](solution/attention_adapter.py), [attention_gqa.py](solution/attention_gqa.py) | Вызов PyTorch `scaled_dot_product_attention` для отдельного примера | Нет | Явное повторение K/V в подходящих случаях T4 и возврат к HF в остальных. Командный эксперимент. |
| [swiglu_pack.py](solution/swiglu_pack.py), [native_half.py](solution/native_half.py), [hybrid_mlp.py](solution/hybrid_mlp.py) | Защищённая замена `mq.Qwen2MLP.forward` | Нет своего; используется SwiGLU примера | Сохранение FP16 или пересчёт ветки. Ветка команды `perf/packed-native-memory`, [происхождение](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). |
| [overhead.py](solution/overhead.py) | `soup_cli.trainer.sft.resolve_device_map`, `soup_cli.trainer.sft._map_text_sft_rows` | Нет | Прямое размещение на GPU и сборка датасета в памяти. Alikhan-kai; [запись интеграции](docs/MERGE_OVERHEAD.md). |
| [swiglu.py](solution/swiglu.py) | Начальная замена `mq.Qwen2MLP.forward` | `example_swiglu_*` — **не наши** | Объединённая активация из стартового набора; авторство и зачётную долю не заявляем. |
| [lora.py](solution/lora.py), [residual_norm.py](solution/residual_norm.py) | Выключены: заготовка LoRA и эксперимент объединения residual/norm | В P/R зачётных запусков нет | LoRA не реализована; общий выигрыш residual fusion не доказан. [Измерения](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md). |

**Голова и loss.** `head_loss_kernel` проходит словарь блоками и устойчиво вычисляет softmax. Forward выдаёт ошибку, энтропию и попадание argmax для каждого токена. Backward пересчитывает порции, применяет входящий масштаб градиента в FP32 и только затем сохраняет FP16-градиенты logits. Замороженной голове градиент весов не нужен. Размер порции `CHUNK=512` задан в [коде](solution/head_loss.py). [Основной тест](solution/test_head_loss.py) сверяет метрики и градиенты с PyTorch; [fallback-тесты](solution/test_head_fallback.py) проверяют нестандартные случаи. [История исправления](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/head_correction/README.md) описывает ошибку масштаба перед округлением и дополнительные проверки.

**NF4.** `nf4_dequant_kernel` извлекает две половины байта, обращается к таблице кодов, применяет масштаб блока и записывает FP16. Вложенные масштабы распаковываются вышестоящим кодом. Поддерживаются блоки по 64 веса, остальные случаи передаются bitsandbytes. [Тесты](solution/test_nf4.py) проверяют значения, нечётные длины, подключение и градиенты matmul. В [журнале](notes/experiments.md) записано побитовое совпадение в раннем отдельном тесте NF4.

**RMSNorm.** `rmsnorm_fwd_kernel` за один запуск считает RMS строки, обратный масштаб и взвешенный результат. `rmsnorm_bwd_kernel` получает градиент входа с редукциями FP32. Если нужен градиент веса, он вычисляется PyTorch. [Тесты](solution/test_rmsnorm.py) сравнивают оба прохода с оригиналом и проверяют fallback.

**RoPE.** `ember_rope_joint` обрабатывает Q и K общим запуском в каждом направлении, сохраняя границы округления умножений. Поддерживаются strides и отсутствующие отдельные градиенты. Синусы и косинусы вычисляет исходный код. `_rope_rotate_block` — внутренняя GPU-функция, а не отдельный измеряемый запуск. [Тесты](solution/test_rotary.py) покрывают значения, layouts, масштаб градиентов, повторное использование компиляции, tracing и fallback.

### Как это встроено

До построения модели и тренера harness вызывает `solution.apply()`. Порядок: `swiglu → head_loss → rmsnorm → nf4 → lora → rotary → overhead`. У `lora` стоит `ENABLED=False`, её патч пропускается. Затем [stage_adapter.py](solution/stage_adapter.py) подключает `packed` и при `ENABLE_SAVE=True` оптимизацию сохранения активаций. Источники: [реестр](solution/__init__.py), [настройки](solution/stage_options.py).

Флаги `ENABLED=False` в `swiglu_pack` и `native_half` не отключают этот явный установщик. `residual_norm` не входит в реестр. Сам импорт `solution` ещё не подменяет методы модели: нужен вызов `apply()`.

Проверяются устройства, типы чисел, формы, режимы и поведение модулей. Неподходящие вызовы обычно направляются в сохранённый оригинал. Ветка attention требует определённых версий PyTorch/Transformers и T4. MLP требует закреплённых версий bitsandbytes/Transformers, известных байтов SwiGLU и стандартных замороженных проекций. При отказе выводится `SAVED_ACTIVATION_PACKING DISABLED`, остаётся предыдущая реализация MLP. Это локальный fallback, а не гарантия общей совместимости. Некоторые ошибки импорта/API завершают запуск; rotary сообщает о невозможности установки. После смены версии или кода нужны новый процесс и повторная полная проверка.

### Как запустить

**Публичный Kaggle-ноутбук:** [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final).
**Финальный коммит:** [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).

Сохранённая версия для сдачи должна заранее содержать URL репозитория и финальный `COMMIT`, чтобы жюри нажало **Run All** без редактирования ячеек. Ссылка на ноутбук и финальный SHA получены. Настроенный [ноутбук сдачи](submission/kaggle_notebook.ipynb) закрепляет этот SHA и выполняет два повтора. Публичность и ссылка на конкретную сохранённую версию ещё требуют проверки: в полученном URL нет ID версии, веб-инструмент не смог открыть страницу, а локальные [метаданные](submission/kernel-metadata.json) содержат `is_private: true`. В [шаблоне организатора](templates/kaggle_notebook.ipynb) репозиторий и коммит ещё не заполнены, а по умолчанию выполняется один повтор. Сам шаблон не является готовой сдаваемой версией.

1. В Kaggle выберите **GPU T4 x2** и Internet; harness использует одну GPU. Не запускайте параллельную работу на GPU.
2. Откройте предоставленный Kaggle-ноутбук и нажмите **Run All**. Его копия в репозитории — [submission/kaggle_notebook.ipynb](submission/kaggle_notebook.ipynb); URL репозитория, финальный SHA и `--repeats 2` уже заполнены. Для нового ноутбука импортируйте эту настроенную копию. Защищённый шаблон организатора не изменяйте.
3. Альтернатива — [build_notebook.py](tools/build_notebook.py) с `--commit` и `--mode full_profile`. Он использует ячейки шаблона и фиксирует ошибки этапов в `status.txt`, продолжая работу. Поэтому проверяйте каждый статус, а не только общую отметку завершения Kaggle.
4. Сохраните публичную версию через **Save & Run All**. Проверьте SHA в логе и все проверки, опубликуйте архив результатов. Одна и та же ссылка на сохранённую версию должна быть здесь и в `SUBMISSION.md`.

**Локально на Linux/CUDA.** Команды ниже закрепляют предоставленный финальный SHA, измеренный в F. Заранее подготовьте CUDA-окружение PyTorch/Triton из таблицы версий. На macOS без CUDA числа T4 не воспроизвести.

```bash
git clone https://github.com/alikhan-kai/decentrathon.git
cd decentrathon
git checkout 632851f0f2c3d202ae898d07b9b7b050eed61699
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" \
  "accelerate==1.14.0" "datasets==4.8.5" pytest
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Если команда тестов завершилась ошибкой, разберите её до признания запуска проверенным; известная проблема окружения описана ниже. Официальные проверки измеренного обучения P/R прошли. Установка пакетов относится к подготовке окружения и выполняется вне измеряемого процесса.

Для отдельного воспроизведения R:

```bash
python tools/make_robust_data.py --out data_robust
python harness/run_task.py --repeats 2 --seed 7 \
  --train-file data_robust/train.jsonl --eval-file data_robust/eval.jsonl \
  --out runs_robust
python harness/check_rules.py --result runs_robust/result.json
```

Результаты: `runs/result.json`, `runs/checks.json`, отдельные `train.log`, метрики и сохранённые адаптеры; у R каталог `runs_robust/`. Для профиля добавьте `--profile` к `run_task.py`. `python tools/summarize.py runs` выводит сводку. Время отдельных измеренных команд приведено выше. **Не знаю полную длительность full-profile процесса** по сохранённым данным P: нужен отдельный замер чистого full-profile запуска с установкой, загрузками, тестами и оценкой. Отдельно записанный ноутбук F выполняет сдаваемый запуск без профилирования. Примерная оценка организатора в 10 минут относится к более простому шаблону, а не к измерению данного full-profile процесса.

### Как проверяли корректность

Тесты сверяют прямой проход и градиенты с PyTorch или bitsandbytes. Допуски явно заданы в каждом тесте. Проверяются маски, неполные порции, layouts и запасные пути. [kcheck.py](solution/kcheck.py) — общий помощник; конкретные тесты могут задавать свои допуски.

В P набор `solution/` завершился с **298 passed, 126 skipped, 5 failed**. Пять падений — `ImportError` PEFT из-за установленного `torchao 0.10.0` при требовании минимум `0.16.0`: три случая head fallback и два residual/LoRA. Источник: локальный `kaggle/out_full_profile_20261031_040542/out/pytest_solution.log`, [сводка](notes/experiments.md), [отслеживаемый статус ошибки](docs/measurements/full_profile_final_e0eed88/status.txt). Полностью успешным этот набор считать нельзя. Отдельные **79 регрессионных тестов Soup прошли** и в P, и в R.

Проверки 5–7 используют базу из той же сессии. В P отклонение loss первого шага решения — **0.03%**, среднее отклонение по шагам — **0.05%**. В R — **0.00%** и **0.05%** с округлением из `checks.json`. Пределы — **1%** и **3%**. Изменение eval loss — **−0.011%** и **+0.017%** при верхнем пределе **+0.5%**. Проверка 5 подтверждает структуру задачи/выходов и метрики в пределах допуска. Источники: P/R и [определения проверок](harness/check_rules.py).

В R среднее абсолютное относительное отклонение row loss от `base_1` — **0.225% / 0.221%** в двух повторах решения. Для короткой строки с **7 токенами ответа** максимум первого повтора — **7.04%**. Проверка 5 ограничивает **среднее** отклонение величиной **3%**, а не максимум каждой строки. Источники: `runs.*.rewind` в R, [журнал](notes/experiments.md), `compare_outputs` в checker. В P отдельные файлы rewind не собраны, но их сводные данные есть в result JSON и использованы проверками.

### Ограничения и что не получилось

- **Конкретная задача.** Замеры относятся к Qwen/T4 FP16 и указанным библиотекам. Для других моделей, BF16 и GPU нужны новые проверки. Неподдерживаемый путь может перейти к оригиналу или не установиться.
- **Привязка к исходнику SwiGLU.** Принимаются только два известных хэша. Текущий SHA-256: `e1dc639838c4c46ea0ed0aec3f0d904d360c92a91c601b943c9cfef85ca21c64`. Даже изменение исходного текста может отключить экономию активаций. См. [защиту](solution/swiglu_pack.py).
- **Fallback внимания.** В каждом P-прогоне решения: **6912 ускоренных вызовов и 256 fallback**, то есть **256/7168 = 3.5714% вызовов**. Причина — `k-no-grad` в слое 0. Это доля вызовов, не времени. Источник: `COMBINED_STAGE_STATS` в логах P и [журнал](notes/experiments.md).
- **Компромисс MLP.** Порог — **1024 токен-позиции**. В P зафиксированы **1344 native-FP16 вызова и 448 пересчётов**. Пересчёт экономит память, но добавляет работу; поддерживается обучение первого порядка в eager-режиме. Источники: [установщик](solution/swiglu_pack.py), [hybrid helper](solution/hybrid_mlp.py), статистика P.
- **Нет побитовой эквивалентности всего обучения.** Packing меняет назначение случайных масок dropout. У коротких строк отклонение loss может превышать порог средней ошибки. Ранняя ошибка масштаба градиента головы потребовала отдельного исправления даже при проходивших автоматических проверках. См. [историю исправления](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/head_correction/README.md).
- **Пять ошибок окружения не устранены в доказательствах.** Нужен повтор полного набора тестов решения в совместимом окружении. Проверки обучения и регрессионные тесты Soup не заменяют этот набор.
- **Нет отдельных финальных ускорений модулей.** [FINDINGS](docs/FINDINGS.md) содержит профили более раннего кода. Сравнения разных сессий T4 ориентировочны; тепловой шум описан в [экспериментах](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md). Нельзя складывать выигрыши частей или напрямую сравнивать время P с R.
- **Незавершённые направления.** `lora.py` — заготовка. Residual/norm fusion выключена: общий выигрыш над rotary-only не доказан. Отдельный прототип batched attention не включён в кандидата. Источники: [LoRA](solution/lora.py), [residual evidence](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md), [кандидат](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md).
- **Оформление сдачи не завершено.** Финальный SHA и URL ноутбука получены, доказательства F отслеживаются Git. Публичность, ссылка на сохранённую версию, состав и медиассылки требуют проверки или заполнения. Исходные исторические результаты P/R по-прежнему игнорируются Git. Замеры не сертифицируют скрытый тест жюри. Ссылки на исторические исследования закреплены на прежнем коммите: в сдаваемом SHA эти документы удалены.

### Что дальше

| Эксперимент | Основание и ожидаемая польза | Нужный замер |
|---|---|---|
| Сократить стоимость NF4 | В P доля 6.3609–6.4231%, рассчитанная как `100 × own_kernel_ms / (1000 × timed_loop_s)`. Возможен выигрыш времени; точную величину не знаю. | ABBA в одной сессии, компиляция включена, сверка значений и все проверки. |
| Уменьшить fallback внимания | В P по 256 fallback-вызовов. Возможен выигрыш времени; точную величину не знаю, доля вызовов не равна доле времени. | Профиль слоя 0 и проверка градиентов Q/K/V до расширения условий. |
| Пересмотреть фиксированный порог MLP | В P 448 пересчётов. Возможен другой баланс времени и памяти; точную пользу не знаю. | Сравнить фиксированные политики на обеих выборках, сохранив RNG и настройки задачи. |
| Реализовать эпилог LoRA | В старом [профиле](docs/FINDINGS.md) PEFT-эпилог занимал 9.7 мс/шаг. Это основание для эксперимента, не прогноз текущего выигрыша. | Сохранить dropout, реализовать scale/cast/add, профилировать финальную задачу и выполнить ABBA. |

### Участники и источники

Название команды: TODO(TEAM_NAME).
Подтверждённый состав, роли, GitHub/Kaggle и контакт капитана: TODO(TEAM_ROSTER_AND_CAPTAIN_CONTACT).

Ниже указано авторство изменений по Git и документам. Оно не заменяет подтверждённый список участников соревнования.

| Автор | Вклад | Основание |
|---|---|---|
| Ansar / `iencislennon` | Head loss, NF4, RMSNorm, packed decoder, интеграция и измерения | `69a595d`, `ca90121`, `c0cc1c8`, [происхождение кандидата](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md) |
| `stritphatographir` | Rotary, исправление градиентов и guards головы, native/hybrid-интеграция | `05543d7`, `5b4dd1f`, `59755b0`, [rotary record](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/README.md) |
| `Alikhan-kai` | Размещение модели и подготовка датасета | `44d1a47`, [запись интеграции](docs/MERGE_OVERHEAD.md) |

- **Makazhan Alpamys и участники Soup:** [Soup][soup] и [стартовый набор][starter]. Идея порционного вычисления головы взята из `stream_loss_chunk`; Triton-кернел головы написан командой. [Ссылка на исходник][soup-chunks] ведёт на upstream `main`; в замере закреплён Soup 0.77.0. См. [NOTICE](NOTICE).
- **SwiGLU организатора:** сохранён в [swiglu.py](solution/swiglu.py). Авторство кернелов, обёрток хранения/пересчёта как новых кернелов и декораторов `do_not_specialize` **не заявляется**. Время `example_swiglu_*` не входит в проверку 4.
- **Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer:** формат NF4 из [QLoRA][qlora]. Авторы bitsandbytes предоставили [формат и эталонное поведение][bnb-source]. Triton-кернел распаковки написан командой.
- **Участники Hugging Face Transformers:** [закреплённая реализация Qwen2][qwen-source] задаёт семантику RMSNorm, RoPE и attention. Основное внимание вычисляет PyTorch SDPA; собственный attention-кернел не заявляется.
- **Работа команды `perf/packed-native-memory`:** донор packed decoder `3e48f6a` и интеграция `59755b0` описаны в [CANDIDATE](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). Там же отмечена помощь Team/Codex. Дополнительные идеи и первичные ссылки сохранены в [исследовании](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/rotary/RESEARCH.md).

Код и тесты стартового набора — Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE). [Модель Qwen][model] — Apache-2.0. Карточка [казахского датасета][dataset] указывает MIT и происхождение из [Stanford Alpaca][alpaca], чья карточка указывает CC BY-NC 4.0. Данные и веса скачиваются при запуске, в сдачу не включены.

### Структура репозитория

```text
solution/    Патчи, Triton-кернелы и тесты компонентов; эту папку копирует жюри.
task/        Фиксированная конфигурация и подготовка публичных данных.
harness/     Замеры, оценка адаптеров, проверки и подсчёт баллов организатора.
tests/       Регрессионные тесты Soup с возможностью подключения решения.
tools/       Сборка ноутбука, локальные проверки, профили и сводки.
kaggle/      Рабочий ноутбук и настройки; скачанные результаты игнорируются.
docs/        Исследования, происхождение кода, замеры и задачи по сдаче.
notes/       Хронологический журнал экспериментов.
templates/   Шаблоны ноутбука и описания сдачи.
.github/     CI и шаблон pull request.
```

[public-commit]: https://github.com/alikhan-kai/decentrathon/commit/e0eed882bc468f3403e0249ebec38db7f84210c7
[robust-commit]: https://github.com/alikhan-kai/decentrathon/commit/952f20a356fc5bf9ae3182567b6aba64d53610b1
[soup]: https://github.com/MakazhanAlpamys/Soup
[starter]: https://github.com/MakazhanAlpamys/soup-kernel-challenge
[soup-chunks]: https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/utils/stream_loss_chunk.py
[qlora]: https://arxiv.org/abs/2305.14314
[bnb-source]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py
[qwen-source]: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
[model]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
[dataset]: https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2
[alpaca]: https://huggingface.co/datasets/tatsu-lab/alpaca
