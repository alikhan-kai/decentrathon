# Exact-commit T4 retest: 2585520

Tested Decentrathon `25855208695fd113fa1897385abccc73f71a19cd` unchanged on 2026-10-31. No PR10 or experimental overlays were applied. These are measurements of the team's commit, not a new implementation or an incremental gain attributed to this review.

| Metric | Solution | Base |
|---|---:|---:|
| Whole-command median | **67.65 s** | 89.80 s |
| Individual command times | 66.80 / 68.50 s | 86.94 / 92.66 s |
| External peak, both repeats | **5327 MiB (5.20 GiB)** | 14345 MiB |
| Training-loop median | 37.98 s | 59.26 s |
| Evaluation loss | 2.04947 | 2.04959 |

All ten official checks passed, including 79 Soup regressions. Minimum useful original-kernel share is **9.5232122800%**, excluding every starter SwiGLU specialization. Official whole-solution speedup is 1.3274x. All four runs completed 64 steps; packages remained unchanged.

The external peak did not reproduce a result just over 4 GiB. PyTorch's allocated high-water was **4917.8 MiB (4.80 GiB)** and reserved high-water **5184 MiB (5.06 GiB)** in both solution runs. These are different metrics; the harness scores the external nvidia-smi peak summed over GPUs. This result remains 12.85 seconds above the 54.8-second target and 634 MiB above 4693 MiB.

## Conditions and reproduction

One visible Tesla T4 (`CUDA_VISIBLE_DEVICES=0`), idle initial aggregate usage 3 MiB, standard harness allocator, unchanged task, production dropout and seed 1. Python 3.13.15; package versions are recorded in runtime.json. The fixed public train and eval data contain 256 rows each, with SHA256:

- train: `93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd`
- eval: `b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e`

From the exact source and these data files under data/:

```sh
python -B harness/run_task.py --repeats 2 --timeout 240
python -B harness/check_rules.py
```

Every training run gets fresh compiler caches. Compilation and imports inside the training process count in command time. Package/model download warmup occurs outside that measured command, following the unchanged harness. Both orders are represented by its base, solution, solution, base sequence. No CUDA graphs, autotuning, task modification or external precompiled cache was used.

## Evidence and limits

Raw archive: 146352845 bytes, SHA256 `2c2cbc68c285e4d6684049fcd0fcf196fa59ea82a70dbc2875a96f31f864b939`. Local raw evidence is at `/home/homelab/.t3/experiments/decentrathon-time-2026-10-31/evidence/latest-2585520/raw.zip`. Its 52 source-manifest entries (48 Git source files, two datasets and two runner files) and 15 protected harness/task/test files were verified. The archive's job-status snapshot predates export; final-job-status.json was extracted from the completed notebook output and records successful completion, cleanup, export and source verification. Source identity is bound to the exact commit.

LORA_REVIEW.md and the CPU reproduction record a separate precision issue: the new forward casts FP32 activations to BF16 before dropout in the task's FP16-autocast path. This changes adapter gradients, including the first B update when the initial forward output still matches. The official checks passed within their tolerances; they do not establish bitwise or gradient equivalence. This CPU result does not quantify a CUDA gradient error. The scaled loss-head implementation remains intact.

No same-session matched comparison against PR10 was run. Do not attribute a difference from its historical 70.54 s / 4421 MiB directly to this commit. The source was not patched during the measurement, and no shared branch was merged.

The owned Kaggle session was stopped at **12:40:05 UTC** after evidence download and checksum verification. GPU priority was handed to the user's memory-task Codex; this thread will not start another GPU job without coordination.
