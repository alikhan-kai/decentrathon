# Measurement Evidence and Provenance

## Final submission-source run (`F`, commit `fabc67d`)

- **Benchmarked runtime commit:** [`fabc67d4b0667be70b846aac29d9be21cb31a212`](https://github.com/alikhan-kai/decentrathon/commit/fabc67d4b0667be70b846aac29d9be21cb31a212) (mirrored in [`BAITC-Hacks/hack-d2097b50-localhost`](https://github.com/BAITC-Hacks/hack-d2097b50-localhost/commit/fabc67d4b0667be70b846aac29d9be21cb31a212)).
- **Protocol:** `python harness/run_task.py --repeats 2` on one visible Kaggle Tesla T4 (`base_1 -> solution_1 -> solution_2 -> base_2`), followed by `python harness/check_rules.py` and `python harness/score.py runs/result.json`.
- **`solution_sha256`:** `7caf827d0fb87cc23400afb227f2efb17e21fcca5ebd94867732f71f0171285c`
- **`result_sha256`:** `fbf7553e8ca4df347be80f567bf311e92cec21275c65270767cf2d7a89fc8585`
- **Tracked checks record:** [`docs/measurements/submission_fabc67d/checks.json`](measurements/submission_fabc67d/checks.json)

### ABBA per-run breakdown (`F`)

| Run | Side | Wall time (s) | Peak GPU memory (MiB) | `timed_loop_s` (s) |
|---|---|---:|---:|---:|
| `base_1` | base | 84.18 | 14345 | — |
| `solution_1` | solution | 52.74 | 3279 | 31.660 |
| `solution_2` | solution | 53.64 | 3279 | 32.218 |
| `base_2` | base | 85.01 | 14345 | — |

### Summary (`runs/result.json -> summary`)

- **Base median wall time:** `84.59 s` (`train_loop_s`: `62.54 s`), peak memory `14345 MiB`, `eval_loss`: `2.04965`
- **Solution median wall time:** `53.19 s` (`train_loop_s`: `35.65 s`), peak memory `3279 MiB` (`torch_max_allocated_mib`: `3002.5 MiB`, `torch_max_reserved_mib`: `3136.0 MiB`), `eval_loss`: `2.04941`
- **Ratios:** `speedup = 1.5903×`, `memory_ratio = 4.3748×`, `quality_delta_pct = -0.012%`

### Per-kernel timing in `F` (CUDA events, steps 3..64, `TIMED_FROM_STEP = 2`)

| Kernel | File | `solution_1` (ms) | `solution_2` (ms) | Share range (`100 × ms / (1000 × timed_loop_s)`) | Eligible |
|---|---|---:|---:|---:|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | 2630.099 | 2663.093 | 8.2659–8.3073% | Yes |
| `head_loss_kernel` | `solution/head_loss.py` | 242.312 | 247.840 | 0.7654–0.7693% | Yes |
| `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | 231.180 | 230.618 | 0.7158–0.7302% | Yes |
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | 204.817 | 201.079 | 0.6241–0.6469% | Yes |
| `ember_rope_joint` | `solution/rotary.py` | 181.907 | 178.034 | 0.5526–0.5746% | Yes |
| `example_swiglu_fwd_kernel` | `solution/swiglu.py` | 454.564 | 463.157 | excluded | Starter |
| `example_swiglu_bwd_kernel` | `solution/swiglu.py` | 885.853 | 907.255 | excluded | Starter |

Combined eligible custom-kernel share: `11.0244%` (`solution_1`) and `10.9276%` (`solution_2`), reported by `harness/check_rules.py` check 4 as **`10.9%`** (lowest of 2 runs, required `>= 5.0%`).

---

## Earlier milestones (`632851f`, `P`, `R`)

- **Earlier baseline candidate (`632851f`):** [`measurements/submission_632851f/`](measurements/submission_632851f/) — `83.25 s` vs `59.66 s` (`1.3954×`), `14345 MiB` vs `4421 MiB` (`3.2447×`), eval loss `-0.012%`, check 4 `10.0%`.
- **Public profile (`P`, `e0eed88`):** `89.66 s` vs `67.08 s` (`1.3366×`), `14345 MiB` vs `4421 MiB` (`3.2447×`), eval loss `-0.011%`, check 4 `8.9%` ([status](measurements/full_profile_final_e0eed88/status.txt), [notes](../notes/experiments.md)).
- **Additional-row robustness check (`R`, `952f20a`, rows 1000–1255 / eval 41000–41255, seed 7):** `81.68 s` vs `61.91 s` (`1.3193×`), `14351 MiB` vs `4405 MiB` (`3.2579×`), eval loss `+0.017%`, check 4 `10.0%` ([status](measurements/robust_final_952f20a/status.txt), [notes](../notes/experiments.md)).

To reproduce a robustness run on alternate rows and seed:

```bash
python tools/make_robust_data.py --out data_robust
python harness/run_task.py --repeats 2 --seed 7 \
  --train-file data_robust/train.jsonl --eval-file data_robust/eval.jsonl \
  --out runs_robust
python harness/check_rules.py --result runs_robust/result.json
```
