# Kaggle balanced optimization pass — 2026-10-31

Exact balanced runtime: `2e7df7b8abf89879b8825ea297afbbcbbaf363f7`. Faster preserved runtime: `2e32ef05b6487a2c58608e10a886a6f9a94fd1e5`. Main remains `47aede7e529db245b93d16597ef5cedb38211ace`.

| Runtime | Whole soup train median | Peak GPU memory | Training loop only | Official checks |
|---|---:|---:|---:|---|
| Main 47aede7 | 49.89 s | 3039 MiB | 32.98 s | 10/10 |
| H16 preserved faster version | 46.05 s | 3325 MiB | 29.27 s | 10/10 |
| h20-head-partial balanced version | 52.62 s | 3065 MiB | 35.70 s | 10/10 |

Each official result uses two cold-compiler-cache repetitions in the same Kaggle session, unchanged harness and task. The timing target is the whole command, not training-loop time. **Sub-40 seconds has not been achieved in this pass.** No GPU jobs ran concurrently.

Fresh unchanged-H16 control screens: [52.39, 51.48], median 51.94 s / 3325 MiB. The balanced official median differs by +0.68 s while saving 260 MiB. These later measurements are the relevant memory/time tradeoff; do not combine the earlier 46.05 s with the later lower-memory result. Fresh unchanged-main control: 54.49 s / 3039 MiB (single screen).

The balanced change retains a bounded prefix of labelled FP16 logits and recomputes the remainder during backward. Forward still computes every loss, entropy and accuracy row; incoming loss scaling and FP16 rounding boundaries are preserved. Cache data is not mutated by backward. Attention and the existing fused MLP remain from H16.

## Screens and rejected experiments

| Variant | Whole command seconds | Peak MiB | Gate |
|---|---:|---:|---|
| h18-head-wide | 50.68 | 3325 | screen_passed |
| h19-head-compact | 51.63 | 3225 | screen_passed |
| h20-head-partial | 52.58 | 3065 | screen_passed |
| h16-control | 52.39 | 3325 | screen_passed |
| h16-control-2 | 51.48 | 3325 | screen_passed |
| baseline47a-control | 54.49 | 3039 | screen_passed |

Grouped gate/up BMM was rejected before a full-training screen: it was slower than separate GEMMs on the T4. Later screens showed substantial timing drift: unchanged H16 took 52.39 s / 3325 MiB, versus H20's 52.58 s / 3065 MiB. H18's 50.68 s and H19's 51.63 s therefore do not establish standalone regressions against the earlier 46.05 s H16 median. H20 was selected for full validation because it reduced memory by 260 MiB while staying within 0.19 s of the fresh H16 control. Screens are exploratory, not full rules/evaluation results. Only the table above marked 10/10 represents complete validation.

## Reproduction and evidence

51 focused GPU tests passed, including cache-boundary, ignored-target, scaled-gradient and repeated-backward cases. All 79 official Soup regressions passed through the rules checker.

Repository static checks are a separate gate: all 48 Python files in `solution/` and `tools/` parse, but `tools/lint_rules.py` reports 37 inherited findings, identical to H16, with zero new findings. CI is therefore not fully green despite the 10/10 official benchmark checks. Details: [static-checks.json](static-checks.json).

```sh
git checkout 2e7df7b8abf89879b8825ea297afbbcbbaf363f7
python -m pytest -q solution/test_head_partial_cache.py solution/test_head_cache_probe.py solution/test_head_loss.py solution/test_head_fallback.py -k 'not speed'
python task/prepare_data.py
python harness/run_task.py --repeats 2 --out runs/balanced
python harness/check_rules.py --result runs/balanced/result.json
```

Environment: Python 3.13.15, PyTorch `2.11.0+cu128`, Kaggle Tesla T4. Remaining dependency versions and GPU inventory are in `environment.json` in the archive. Original harness allocator settings and seed 1 are unchanged.

Exact files: [kaggle-balanced-source-manifest.json](kaggle-balanced-source-manifest.json). Raw full result: [kaggle-balanced-result.json](kaggle-balanced-result.json). All checks: [kaggle-balanced-checks.json](kaggle-balanced-checks.json). Raw evidence: [kaggle-pass3-evidence.zip](kaggle-pass3-evidence.zip), SHA256 `bc72875b471e7751d57bee98ec1a3380fc0f0eba2983212914218d877518f50e`.

Upstream `db19acab9cfcb9944a7ffcd7737383b588247212` was inspected. Its individual change moves precompilation earlier; H16 already starts it first. Its full source snapshot lacks the fused MLP, head cache and newer packed-attention improvements, so it was not substituted or claimed as measured. [Review](upstream-db19-review.json).

Rollback options are the exact H16 and main commits above. This experiment branch does not alter main.
