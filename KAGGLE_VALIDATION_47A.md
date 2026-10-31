# Kaggle validation and sub-40 speed search

**The sub-40-second target was not achieved.** Current main source `47aede7e529db245b93d16597ef5cedb38211ace` measured **50.83 seconds / 3039 MiB** on Kaggle T4, with all 10 automatic checks passing. This evidence branch changes documentation only; its runtime code remains that exact commit.

The separately published [3029 MiB memory branch](https://github.com/alikhan-kai/decentrathon/tree/perf/head-memory-runtime-20261031) contains runtime commit `d3ca849` and evidence commit `e4a2dc7`. It measured **58.06 s / 3029 MiB in Colab**, not Kaggle. Both versions remain independently reproducible.

## Repeated Kaggle results

Public task rows, seed 1, official unchanged harness, two cold-cache repeats in ABBA order, one Tesla T4. Whole command time includes compilation and adapter saving. Training-loop time is not used as the sub-40 metric.

| Source / hypothesis | Whole command median, s | Peak MiB | Eval loss | Checks |
|---|---:|---:|---:|---|
| Original baseline `fabc67d` | 52.23 | 3279 | 2.04945 | 10/10 |
| Earlier precompile (V8) | 51.75 | 3279 | 2.04944 | 10/10 |
| Selective cache on old baseline (V11) | 51.22 | 3621 | 2.04945 | 10/10 |
| Current main `47aede7` | 50.83 | 3039 | 2.04983 | 10/10 |

Current main's individual solution runs: **51.01 s** and **50.66 s**, both **3039 MiB**. Its paired stock runs have median **81.44 s / 14345 MiB**. Own-kernel share: **11.3% of the training time (lowest of 2 runs, steps 3+)**. All 79 Soup regressions pass.

The startup-only change's initial 49.61 s screen did not repeat: its full median was 51.75 s. The old-baseline selective cache traded 342 MiB for about one second; it was not promoted. Neither supports a sub-40 claim.

## Rejected speed experiments

These are single training screens, not full admission results. They are not enabled in this branch.

| Experiment | Platform | Whole s | MiB | Decision |
|---|---|---:|---:|---|
| Full decoded-weight cache | Kaggle T4 | 46.65 | 6121 | Reject: own-kernel share 3.404%, below 5% |
| Cached weights with native autograd | Kaggle T4 | 48.01 | 6121 | Reject: own-kernel share 3.173% |
| Custom frozen-MLP gradient GEMM | Kaggle T4 | 95.91 | 6121 | Reject: much slower despite 27 GPU tests passing |
| Manual LoRA backward | Colab T4 | 56.90 | 3299 | No improvement over the 56.95 s Colab baseline |
| Current main, unchanged | Colab T4 | 53.07 | 3039 | Refreshed baseline; 160 focused tests passed |
| Current main, all-native MLP budget | Colab T4 | 53.19 | 4159 | Reject: no speed gain, much more memory |
| Current main, selective decode cache | Colab T4 | 55.55 | 3341 | Reject: slower and more memory |

Generated PTX for the attempted custom GEMM had no `mma.sync` Tensor Core instructions and substantial register spills on this T4 target. Five fixed tile layouts were checked; all lost to native PyTorch GEMM. No autotuning or compiler-cache reuse was introduced into the solution.

The memory-focused V1–V6 Colab trials are retained in the evidence archive. V4 (3029 MiB) passed all 10 checks with two full solution repeats; higher activation budgets did not approach 40 seconds. Cross-platform results are kept separate.

## Reproduce current main

```bash
git checkout 47aede7e529db245b93d16597ef5cedb38211ace
python task/prepare_data.py
python harness/run_task.py --repeats 2 --out runs_47a_validation
python harness/check_rules.py --result runs_47a_validation/result.json
```

Use the pinned dependencies and an idle T4. The task, harness and Soup tests were not changed. Baseline and candidate runs used the same Kaggle session. No new paid resources were used.

Kaggle environment: Python 3.13, torch 2.11.0+cu128, Triton 3.6.0, Soup 0.77.0. Exact library versions and GPU metrics are in `latest-main-full/result.json` within the archive. Colab used torch 2.11.0+cu130; its timings are not interchangeable with Kaggle's.

## Evidence and provenance

[Download all raw measurements and exact experimental payloads](GPU_VALIDATION_20261031.zip). The bundle contains separate Kaggle and Colab archives, results/checks JSON, per-run metrics, training logs, source payloads and the experiment ledger. It excludes adapters, model weights, credentials and compiler caches.

- Bundle SHA-256: `315b2fff786db73c9f228cb4cfa67f48b5bfaa3d24c5807cbbdb51f51b12a942`
- Current-main result SHA-256: `82cbd69d233ba6850732483d5a20a4e9f9c7fe5230795fdff0b2c22b14d432af`
- Current-main solution SHA-256: `57e0f198eb59b4b12c62ee5134c58d3a52b1464a251413292bcaab8555c364a2`
- Kaggle notebook: [Ember Score Validation 20261031](https://www.kaggle.com/code/tukhtakhunovamir/ember-score-validation-20261031) (account access may be required).

The existing `benchmark-optimization-loop` skill guided correctness gates and repeated measurements. Experimental speed changes that lost or failed admission remain unpromoted. The existing static lint reports pre-existing lazy-BNB compile-wrapper warnings; the official task checks pass. These public-row results are not hidden-jury measurements and do not prove a global optimum.
