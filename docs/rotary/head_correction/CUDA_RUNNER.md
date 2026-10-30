# Bounded CUDA reproduction

Copy this entire directory and use the pinned CUDA environment described in
the parent rotary README. From this directory, select one idle T4 and run:

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=2s 117s \
  python -B -u run_cuda.py --budget-seconds 110 \
  > cuda-results.jsonl 2> cuda-stderr.log

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=2s 117s \
  python -B -u run_cuda_tails.py --phase validate --budget-seconds 110 \
  > cuda-tail-results.jsonl 2> cuda-tail-stderr.log
```

These are separate sequential runs. Keep the external timeout: a stalled CUDA
call may not respond to a Python exception. Do not interpret timeout, missing
final summary, or partial counts as a pass. The scripts do not install packages.

The main runner compares the corrected all-row backward against the corrected
supervised-row backward at H=1536, V=151936 and CHUNK=512 with frozen fp32 weights.
It requires 32 dense-oracle comparisons, including actual GradScaler optimizer
updates, and 48 timing samples. Three alternating pairs per case are a component
comparison, not a statistically established whole-training speedup. Timings include
gather/scatter and CUDA `nonzero` synchronization.

The tail runner requires 24 oracle comparisons around 511, 512 and 513 selected
rows out of 1024, at scales 1 and 65536, including N=513 and N=1024 cases. It is
validation-only with the command above. Recorded results and exact source hashes
are linked from README.md. The official full-training check is separate.
