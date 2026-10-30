# Correct loss gradients with supervised-row-only backward

The actual `solution/head_loss.py` and registry remain unchanged on this branch.
This directory supplies a narrow correction for the integrator. Passing the
standard admission checks did not detect the demonstrated gradient defect.

The original source (`9692acc5d32318872f7b50b67e4e4c8d61d86a33dea2458f06443eddfd1c3934`)
rounds unscaled logit and hidden gradients to fp16, then applies the upstream
GradScaler multiplier. The missing small gradients cannot be recovered.
The candidate (`3699eb819b6d3bd415980e2ab5e73030b10babb26a4fa5ca2c908264474b92e1`)
recomputes projection chunks during backward and scales before either fp16
boundary. It processes only supervised rows in backward and scatters into a
zero-initialized full hidden gradient. Forward still processes every attended
position required for entropy, plus shifted supervised boundary positions.

It also preserves zero ignored-target gradients with zero denominator and fixes
supervised targets at left-padding boundaries. The public API, projection dtype,
trainer hooks, registry and Rewind interface are preserved. The saved fp16
hidden/weight buffers replace the original unscaled hidden-gradient buffer.
No full vocabulary-sized logits buffer is retained across forward/backward.
The independent `head_loss.correct-r6.py` snapshot is the straightforward correct
version that recomputes every retained row; it is the performance comparison.

## Validation and measured limits

The original source fails 11 assertions/subtests on T4, including the real
GradScaler test. The straightforward repair passes all 10 original regressions;
the supervised-row candidate passes all 18 extended CUDA regressions. Tests
cover mutation detection, repeated backward, mask densities, real tiny Qwen2
parameter gradients, and Rewind row counts/identity. See the `r6-head-*` and
`r8-head-*` logs in [evidence](../evidence/).

At H=1536, V=151936, CHUNK=512 and fp32 frozen weights, all **32** comparisons
against dense fp16 projection plus fp32 CE passed. They use N=513/1024, scales
1/65536, actual CUDA GradScaler unscale/optimizer-step/update, and supervised
fractions 0%, 1%, 25%, 100%. Metrics and ignored-row zero gradients are checked.
Gradient tolerances are recorded in the raw JSONL; this is not a bitwise-parity
claim. The benchmark includes gather/scatter and `nonzero` synchronization.

For N=1024, **three alternating measured pairs per case** follow one
warmup per implementation. Complete allocating forward/backward medians:

| Supervised rows | Straightforward correction | Selected-row correction | Ratio |
| --- | ---: | ---: | ---: |
| 0% | 76.11 ms | 31.95 ms | 2.38x |
| 1% | 67.93 ms | 30.98 ms | 2.19x |
| 25% | 70.19 ms | 37.81 ms | 1.86x |
| 100% | 70.02 ms | 70.98 ms | 0.986x |

These are head-component results versus the corrected recompute-all version,
not speedups versus the original incorrect head or full training. The recorded
training workload averages 63.54% supervised positions; its actual effect must
be measured with the full harness. Raw results: [JSONL](../evidence/r8-head-cuda-results.jsonl).

## Repeated full-training validation

The corrected candidate passed [all ten official checks](../evidence/r8-final-checks.json)
across two baseline/solution repeats. The [raw result](../evidence/r8-final-result.json)
records median command time **78.07 s versus 83.31 s** (1.0671x), peak GPU memory
**8367 versus 14345 MiB** (1.7145x), and evaluation loss **2.04966 for both**.
Minimum eligible original-kernel share is **7.0106%**. Solution command times
were 76.92/79.22 s; baseline times were 79.37/87.24 s.

The selected training configuration also passed **270 tests, 126 skipped**, and
**79 official regressions**. The staged 34-file manifest matches the measured
snapshot exactly; its only difference from the rotary-only validation is
`solution/head_loss.py`. These totals include all team optimizations and rotary.
They do not establish that the correction is faster than the original defective
head. Its median is slightly higher than the separate earlier run; numerical
correctness is the reason for this change. No residual candidate is enabled.

## Reproduce

From the pinned T4 environment:

```bash
CUDA_VISIBLE_DEVICES=0 python docs/rotary/head_correction/run_smoke.py --device cuda
CUDA_VISIBLE_DEVICES=0 python docs/rotary/head_correction/run_cuda.py --budget-seconds 110
CUDA_VISIBLE_DEVICES=0 python docs/rotary/head_correction/run_cuda_tails.py --phase validate --budget-seconds 110
```

The last command adds scattered 511/512/513-of-1024 masks around the production
chunk boundary. All **24 additional T4 oracle comparisons passed**; see
[raw tail results](../evidence/r9-head-tail-results.jsonl). CPU mode uses a documented interpreter scalar compatibility
adjustment and does not establish CUDA execution.

For an isolated training snapshot:

```bash
python docs/rotary/stage.py --variant rotary --head-correction --out /kaggle/working/corrected-rotary
```

The script verifies the original file hash, substitutes only the candidate in
the new directory, and writes its source manifest. Permanent integration stays
with the module owner:

```bash
git apply --check docs/rotary/head_correction/head_loss.patch
git apply docs/rotary/head_correction/head_loss.patch
```

Existing eligibility limitations and higher-order derivatives are outside this
narrow correction. All-ignored zero-divisor scalar loss remains NaN, matching
the original reduction; its gradients are zero. CUDA compaction synchronizes
once per backward, and at full supervision adds some overhead. Retaining the
445.125 MiB fp16 weight copy is an allocation fact, not a claim about peak memory.
