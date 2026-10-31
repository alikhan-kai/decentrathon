# 3029 MiB memory experiment

Runtime source: [`d3ca849ee0c7fdec69d43ecaacd814893a0a1e97`](https://github.com/alikhan-kai/decentrathon/commit/d3ca849ee0c7fdec69d43ecaacd814893a0a1e97), branch `perf/head-memory-runtime-20261031`.

Google Colab Tesla T4, public task rows, seed 1, two cold-cache repeats in official ABBA order. This is a measured memory alternative, not the sub-40-second speed candidate. Kaggle measurement is still pending; do not describe these as Kaggle numbers.

| Metric | Stock Soup | This branch |
|---|---:|---:|
| Whole command, median seconds | 87.47 | 58.06 |
| Peak GPU memory, MiB | 14345 | 3029 |
| Eval loss | 2.04968 | 2.04944 |

Individual solution runs: **58.42 s / 3029 MiB** and **57.70 s / 3029 MiB**. All 10 official checks pass, including the 79 Soup regression tests. Focused GPU tests cover variable-length attention, compact grouped K/V storage, lossless FP16 rematerialization boundaries and exact dropout RNG/gradient behavior.

## Changes

- Run native PyTorch variable-length efficient attention across the packed batch.
- Retain compact grouped K/V storage instead of materializing repeated K/V heads.
- Save frozen-MLP recomputation inputs at their exact FP16 compute boundary; keep independent FP32 gradient accumulation boundaries.
- Preserve native dropout output and RNG while bit-packing its saved Boolean mask.

The fixed model, data, row order, optimizer, LoRA dropout, training steps, task and harness are unchanged. Starter SwiGLU kernels remain excluded from original-kernel credit. Runtime source is byte-identical to the measured Colab V4 payload.

## Reproduce

Use the pinned dependencies in the original README, prepare the unchanged task data, and run from this checkout on an idle T4:

```bash
python task/prepare_data.py
python harness/run_task.py --repeats 2 --out runs_memory_3029
python harness/check_rules.py --result runs_memory_3029/result.json
```

Do not use a warm compiler cache or change the harness. New hardware/session timing may differ.

## Evidence

[Download raw results, checks, per-run metrics and logs](ember-memory-3029-evidence.zip).

- Evidence ZIP SHA-256: `c755d15c01899af5f3817598c9c91d4bfe5dda2b79dcb6b8c2d6cea4eca5e1a5`
- Result SHA-256: `a8f8db86d329b1a383fb555cc56f8fe9ff2fc4f09700a1c709847cbbb2735dac`
- Solution tree SHA-256: `7bc118b3e34f3baf5f2fa39abc310f7110c0fef1f0270a9bae5e62a5e9730b34`
- Validation: 12.5% of the training time (lowest of 2 runs, steps 3+)

The ZIP also contains the exact source payload used in Colab. Documentation-only commits after the runtime SHA do not alter measured solution code.
