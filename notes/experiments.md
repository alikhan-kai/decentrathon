# Experiments

| date | commit | MODE | key numbers | what broke / fix |
|---|---|---|---|---|
| 2026-10-30 | 69a595d (code) | quick | pytest solution 5 passed; pytest tests 79 passed; wall 81.9 s; nvidia-smi peak 8385 MiB; torch reserved 8242 MiB; head_loss_kernel 138 launches, 200.4 ms of 44.924 s timed loop (0.45%); example swiglu 2972.7 ms; eval loss 2.04966 | nothing broke; check 4 expected to fail (own kernels < 5%). Numbers: kaggle/out/out/files/runs_try/ |
| 2026-10-30 | fe282f8 | full_profile | base 84.7 s / 14345 MiB, solution 73.38 s / 8385 MiB, speedup 1.154x, memory 1.711x, eval loss 2.04966 -> 2.04969 (+0.001%); own kernels 0.4% (head_loss 3.2 ms/step) | check 4 fails as expected (0.4% < 5%), checks 1-3, 5-10 pass. Numbers: kaggle/out_full_profile_20261030_151022/ |
| 2026-10-30 | 9995a69 | check_and_ops | pytest solution 7 passed 9 skipped; pytest tests 79 passed; wall 77.37 s; peak 8385 MiB; eval loss 2.04967; head_loss_kernel 3.21 ms/step (0.46%); op profile: total GPU 740.6 ms/step, RMSNorm 62.1, rotary 32.0, NF4 dequant ~40 | nothing broke; scaffold (ENABLED, apply) works on T4. Data: docs/measurements/check_and_ops_9995a69/ |
| 2026-10-31 | 4c51afb | quick | solution.apply() total 8.151s: swiglu patch 2.986s (modeling_qwen2 import), head_loss patch 5.162s (trl import) | added microsecond instrumentation to apply() to profile startup latency breakdown |
| 2026-10-31 | 44d1a47 | quick | base 78.44 s / 14345 MiB; solution 71.36 s / 8385 MiB; speedup 1.099x, memory 1.711x; eval loss 2.04964 -> 2.04964 (+0.000%); pytest solution 8 passed; Soup tests 79 passed | solution_1 reduced by 2.37 s (73.73 s -> 71.36 s) via fast in-memory dataset mapping and direct single-GPU device mapping in solution/overhead.py |
