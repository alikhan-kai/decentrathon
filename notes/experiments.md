# Experiments

| date | commit | MODE | key numbers | what broke / fix |
|---|---|---|---|---|
| 2026-10-30 | 69a595d (code) | quick | pytest solution 5 passed; pytest tests 79 passed; wall 81.9 s; nvidia-smi peak 8385 MiB; torch reserved 8242 MiB; head_loss_kernel 138 launches, 200.4 ms of 44.924 s timed loop (0.45%); example swiglu 2972.7 ms; eval loss 2.04966 | nothing broke; check 4 expected to fail (own kernels < 5%). Numbers: kaggle/out/out/files/runs_try/ |
| 2026-10-30 | fe282f8 | full_profile | base 84.7 s / 14345 MiB, solution 73.38 s / 8385 MiB, speedup 1.154x, memory 1.711x, eval loss 2.04966 -> 2.04969 (+0.001%); own kernels 0.4% (head_loss 3.2 ms/step) | check 4 fails as expected (0.4% < 5%), checks 1-3, 5-10 pass. Numbers: kaggle/out_full_profile_20261030_151022/ |
| YYYY-MM-DD | <short sha> | quick/full/full_profile | wall, peak MiB, own_kernel_ms by kernel, share % | what broke and how it was fixed |
