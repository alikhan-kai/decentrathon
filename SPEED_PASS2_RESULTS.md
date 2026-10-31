# H16 exact code and measurements — 2026-10-31

Runtime commit: `2e32ef05b6487a2c58608e10a886a6f9a94fd1e5` on `perf/colab-speed-pass2-20261031`, based on main `47aede7e529db245b93d16597ef5cedb38211ace`.

**Kaggle validation completed: 46.05 s median / 3325 MiB, all 10 official checks passed.** This exact runtime is preserved on the experiment branch; main is unchanged. No sub-40-second result has been measured. Further work targets both time and memory, toward approximately 3 GiB and 40 seconds.

| Exact code | Platform / measurement | Whole soup train wall time | Peak GPU memory | Training loop only |
|---|---|---:|---:|---:|
| H16 2e32ef0 | Kaggle T4, official harness, 2 repeats | **46.05 s** (45.82, 46.28) | **3325 MiB** | **29.27 s** |
| Main 47aede7 | Kaggle T4, official harness, 2 repeats | **50.83 s** (51.01, 50.66) | **3039 MiB** | See prior evidence |
| Main 47aede7 | Colab T4, fresh screen in this pass | 58.42 s | 3039 MiB | 35.17 s |
| H16 2e32ef0 | Colab T4, single screen | 55.26 s | 3325 MiB | 34.79 s |

Whole-command wall time includes startup and cold compiler caches; training-loop time is not the target metric. Kaggle H16 eval loss was 2.04988 versus official base 2.04965 (+0.011%). Original kernels accounted for at least 12.4% of training time. All 79 Soup regressions passed. The prior main result is a separate session; a same-session main comparison is currently running. Historical Colab screens below are not used to claim a Kaggle speedup.

## Exact changes

- Bounded 512 MiB FP16 head-logit cache avoids repeated backward projection when the activation fits. Larger inputs retain recomputation. Incoming loss scaling occurs before FP16 gradient rounding, and retained backward does not mutate saved logits.
- Native PyTorch packed causal attention replaces separate row calls, with an unsupported-metadata fallback. This native operator is not claimed as an original kernel.
- Existing fused MLP path is preserved. Harness, task, training data, and official tests are unchanged.
- The slower NF4 scale-cache experiment is excluded.

## Completed checks and evidence

199 focused GPU tests passed on Kaggle T4 (20 warnings), and previously on Colab T4 (6 warnings):

```sh
python -m pytest -q solution/test_head_cache_probe.py solution/test_head_loss.py solution/test_head_fallback.py solution/test_varlen_attention.py solution/test_attention_frozen_key.py solution/test_mlp_budget.py -k 'not speed'
```

Kaggle used the unchanged official harness with two repetitions, followed by all 10 rules checks. The same commands were used in the earlier interrupted Colab validation:

```sh
python harness/run_task.py --repeats 2 --out runs/h16-full
python harness/check_rules.py --result runs/h16-full/result.json
```

Kaggle summary and complete check details: [kaggle-h16-validation.json](kaggle-h16-validation.json). Result SHA256: `f5514328d3d5b5e2f52ec551b395744f58412288fa651d28cbe1fc7eb460f54f`. The result and solution digests were verified by the rules checker. Raw Kaggle logs will be exported with the ongoing comparison pass.

Historical Colab logs/screens: [colab-speed-pass2-evidence.zip](colab-speed-pass2-evidence.zip). This archive includes partial interrupted full-run logs, not the completed Kaggle result. Archive SHA256: `afb56a28e00f57cbe2ced0f834c6a1d7d7350743074f13efb8502d83e825a62f`.

[h16-source-manifest.json](h16-source-manifest.json) records the exact published source hashes. The five uploaded files were byte-verified against this runtime commit. [h16-screen.json](h16-screen.json) and [baseline47-repeat.json](baseline47-repeat.json) contain the screen measurements.

Prior fully validated Kaggle baseline: [KAGGLE_VALIDATION_47A.md](https://github.com/alikhan-kai/decentrathon/blob/validation/kaggle-47a-20261031/KAGGLE_VALIDATION_47A.md), all 10 official checks and 79 focused regression tests passed. That result belongs to main 47aede7, not H16.

Rollback: use main commit `47aede7e529db245b93d16597ef5cedb38211ace`. H16 remains available as the faster validated option while lower-memory variants are measured separately.
