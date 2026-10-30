# Experiments

| date | commit | MODE | key numbers | what broke / fix |
|---|---|---|---|---|
| 2026-10-30 | 69a595d (code) | quick | pytest solution 5 passed; pytest tests 79 passed; wall 81.9 s; nvidia-smi peak 8385 MiB; torch reserved 8242 MiB; head_loss_kernel 138 launches, 200.4 ms of 44.924 s timed loop (0.45%); example swiglu 2972.7 ms; eval loss 2.04966 | nothing broke; check 4 expected to fail (own kernels < 5%). Numbers: kaggle/out/out/files/runs_try/ |
