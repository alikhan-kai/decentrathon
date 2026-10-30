# Further optimization review — 30 October 2026

The target remains the fixed T4 training command. Original-kernel share is an
admission gate, not a score that grows indefinitely with more kernel time.
An optimization must do useful work, preserve the training contract, and improve
the complete measured command or memory use. No external kernel implementation
was imported in this review.

## Firsthand experiments found on Reddit

These reports supplied hypotheses, not validated speed estimates for this project.

- [T4 RMSNorm/RoPE experiment](https://www.reddit.com/r/LocalLLaMA/comments/1q71xc6/getting_30k_tokenssec_on_t4_with_14m_moe_model_is/):
  faster isolated kernels did not improve training throughput; the author later
  identified inefficient memory accesses. This is the closest hardware and
  operation match, but it supplies no demonstrated post-fix training result.
- [Architecture-specific Triton launch choices on RDNA3](https://www.reddit.com/r/ROCm/comments/1v1fe7z/rdna34tuned_int8int4_triton_kernels_for_comfyui/):
  a reason to test fixed tile/warp choices. AMD settings, inference GEMMs and
  autotuning results do not transfer directly to our T4 task.
- [ApexW optimizer microbenchmark versus full training](https://www.reddit.com/r/learnmachinelearning/comments/1xgqh8q/i_built_a_custom_tritonbased_pytorch_optimizer/):
  highlights host overhead and many small operations. Its claimed local gains
  are not evidence of a gain in our training loop.
- [Qwen3-TTS fusion experiment](https://www.reddit.com/r/speechtech/comments/1s0zbqj/project_i_built_a_triton_kernel_fusion_library/):
  similar RMSNorm/RoPE/residual operations, but its RTX 5090 headline combines
  CUDA graphs with kernels. Neither its result nor its implementation is used.

## Transferable mechanisms

The [Triton fused-softmax tutorial](https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/python/tutorials/02-fused-softmax.py)
and [NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy)
support investigating memory traffic, coalescing and register/occupancy tradeoffs.
They do not predict the winning launch configuration for this workload.
[NVIDIA's overhead explanation](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
also motivates separating host dispatch cost from GPU execution.

The bounded experiments are:

1. Use simpler address arithmetic for canonical dense B,T,H,D storage viewed
   as B,H,T,D, retaining the generic stride/broadcast path. Compare fixed
   blocks 256 and 512; keep sequence lengths dynamic.
2. Give residual kernels valid hidden-width alignment information while retaining
   dynamic sequence lengths and correct combined-gradient casting. Remove an
   immediately repeated eligibility check only after the first check succeeds.
3. Investigate the existing loss path's scale-before-cast correctness separately
   from performance. The integrator owns that module; provide a narrow patch and
   regression evidence rather than changing its ownership or registry here.

Each performance candidate starts with correctness tests and interleaved component
measurements. Only a promising candidate advances to compilation-inclusive full
training. Residual fusion remains disabled unless matched evidence supports it.
Rejected experiments and unfavorable measurements are retained.

## RoPE experiment result

The runtime packed-index candidate passed **198 tests on T4, 65 skipped**.
The component benchmark also checked exact eager forward/backward parity before
seven interleaved timing blocks for each dtype, length and gradient layout.
[Raw measurements](evidence/r5-rope-benchmark.json) and
[complete output](evidence/r5-rope-output.txt) are retained.

**Rejected:** packed indexing and block 512 do not show a consistent improvement.
For the task's fp32 path, packed indexing ranged from approximately 1.00 to 1.07
of baseline latency; increasing the block size also produced mixed results.
GPU events enclose the complete allocating autograd operation and can include
host dispatch gaps. These are component observations, not training-speed claims.
The first-call timings are not a fair compilation comparison: the first baseline
timer also drains queued oracle work. Warmed measurements are unaffected.
Generic-forward performance was not measured; no candidate was promoted.

The archived candidate and benchmark are under `experiments/r5/` solely for
reproduction. They are not imported by the solution. The benchmark expects the
candidate at `solution/rotary.py`, and baseline copies named `baseline.py` and
`baseline512.py`; the latter changes only block 256 to 512. The raw JSON records
all source hashes. Do not copy this experimental candidate into the submission.

## Residual alignment result

The width-specialized candidate passed **62 T4 tests, 77 skipped**, including
unaligned storage offsets. Its dense-oracle benchmark passed scale/cancellation
checks and retained compilation-cache keys across sequence lengths.
[Raw measurements](evidence/r6-residual-benchmark.json) and
[unit results](evidence/r6-residual-cuda.log) are retained.

**Not promoted:** the paired baseline/candidate latency ratios were 0.997,
0.953 and 1.005 at T=127, 256 and 512 respectively (>1 favors the candidate).
The static register reduction did not produce a consistent complete-operation
improvement. Both residual implementations can beat team add-plus-norm at the
largest shape; that does not establish an incremental benefit from this change
or an end-to-end training benefit. The existing optional residual source and
its disabled default are unchanged. The candidate remains reproducible under
`experiments/residual_r6/`.

## Why supervised-only backward is worth testing

The latest 64-step run records 33,608 supervised targets and 52,890 attended
positions: approximately 63.54% supervision. Per-step fractions range from
24.80% to 81.46%. These counts come from the existing Rewind token counts and
TRL cumulative token metric in `r5-final-result.json`; no task data is changed.
Skipping mathematically zero ignored-row gradients therefore removes real work,
but the gain depends on mask density and must include gather/scatter and CUDA
`nonzero` synchronization. Forward must still process attended prompt positions
for entropy. This is a separate optimization of the corrected backward, not
permission to omit forward diagnostics.

## Retained result

Keep the RoPE real-FX fallback fix. Recommend the separately staged head-gradient
repair with supervised-row-only backward: it passes 18 CUDA regressions, 32
production-dimension dense-oracle comparisons, and all ten official full-training
checks across two repeats. The aggregate configuration measures 78.07 s median,
8367 MiB and 7.0106% minimum eligible share, with unchanged reported eval loss.
See [full evidence](evidence/r8-final-result.json) and
[correction details](head_correction/README.md). This fixes a real gradient defect
while bounding its cost; it does not establish a full-training speedup over the
previous numerically defective team head. Neither rejected kernel experiment is
installed, and residual fusion remains disabled.
