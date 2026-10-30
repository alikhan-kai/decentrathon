# Track 1 decision: original Triton training kernels for T4

Decision record, 2026-10-30. The research below predates implementation. Current evidence: original vocabulary/RMSNorm kernels pass initial T4 tests; one official development run used 10,223 MiB versus 14,345 MiB, with matching quality, but only 2.1% original-kernel share. Admission fails. See README and docs/validation for current measurements.

The user has experience with GPU testing/local LLMs, direct RTX 5090 access, and T4 access through Colab. This changes the earlier track recommendation: pursue the Soup Kernel Challenge. Kaggle T4 access and the user-designated repository alikhan-kai/decentrathon are now available. This workspace itself has no NVIDIA device.

## Central bet and scope

Build a bounded-memory output-head and loss path with original Triton reductions and correct fp16 backward semantics. Preserve entropy, accuracy, per-row rewind loss, token counts, sampler identity and the standard saved adapter. The intended substantial kernel contribution is tiled output projection and joint statistics. A focused NF4 input-gradient kernel is the competing high-upside direction if early T4 profiling supports it. Do not attempt two complete tiled engines simultaneously within the deadline.

The memory foundation may start with cuBLAS projections on 64–128 total-token chunks and original Triton statistics/gradient kernels. This alone is not assumed to satisfy admission. Measure original kernel share immediately; faster reductions can fall below the mandatory 5% of training wall time.

## Originality and inspected baseline

- Starter revision: `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. The 15 files under `harness/`, `tests/`, and `task/` match `starter-integrity.json`.
- Soup 0.77.0 already implements chunked head/loss with diagnostic and rewind integration in `soup_cli/utils/stream_loss_chunk.py`. Streaming and metric preservation are not new ideas. Do not enable the built-in switch or present existing code as original work.
- CCE and Liger establish related approaches. Credit ideas; no importing/copying their kernels. CCE's documented custom implementation requires Ampere or newer; original T4 work remains a relevant engineering target.
- bitsandbytes 0.50.2 has hardware/shape-dependent fused NF4 forward dispatch. Its backward still dequantizes before computing input gradients. It retains packed weights in autograd context, not a persistent dense copy of all weights.
- NF4 dX is therefore a cleaner high-ceiling experiment than assuming both directions are unoptimized. Dense weight temporaries are much smaller than the full vocabulary buffers; do not misattribute gigabytes of savings to removing one weight temporary.

## Correctness contract

1. Loss and accuracy use causally shifted supervised labels; ignore `-100`.
2. Entropy includes every attention-valid position, including prompts and the last real token. Do not project assistant tokens only.
3. Normalize summed token loss using the actual `num_items_in_batch` denominator when supplied. Do not average chunk or row means.
4. Keep row-level loss/count reductions and Rewind sampler FIFO behavior intact.
5. Verify the dtype boundary through Accelerate. Inner head logits are fp16, while returned outputs can be promoted to fp32 before TRL/Rewind consume them.
6. Apply the upstream loss gradient, including GradScaler scaling, before casting logit gradients to fp16. Precomputing unscaled fp16 gradients during forward can destroy small contributions.
7. Keep LoRA q/v branches, dropout, parameters, optimizer, schedule, batches and Soup switches unchanged.
8. Use ordinary shape/device capability guards and preserve generation, prediction and unsupported configurations through saved original methods. No test/harness detection.

## Analytical checks completed

For each token, stable streaming summaries of maximum, exponential sum, exponential-weighted logit sum, target logit and argmax recover cross-entropy and entropy. A local NumPy fp64 check over 13 synthetic rows, vocabulary 1003 and tiles of 37 matched dense loss within 1.8e-15 and entropy within 5.8e-15; argmax and ignored-label handling matched. This is algebra evidence only, not fp16 or GPU validation.

A separate illustrative uniform-probability gradient with vocabulary 151936 and 1024 supervised tokens is about 6.43e-9: casting it to fp16 before scaling yields zero, while scaling by 65536 first yields about 4.21e-4. This motivates backward-time scaling; it is not a model training result.

At the maximum fixed shape B=4, T=512, H=1536, V=151936:

- One full logits tensor: 593.5 MiB fp16 or 1187 MiB fp32.
- One 128-total-token logits chunk: 37.09 MiB fp16 or 74.19 MiB fp32.
- Hidden states or dH: 6 MiB fp16 or 12 MiB fp32.
- Largest individual MLP weight, 8960 by 1536: 26.25 MiB fp16.

These are sizes of individual tensors, not measured run peaks or additive peak-memory savings. Recomputed head projections cost real compute and can hurt time.

## First hardware experiments

1. Run the unchanged baseline on Colab T4 with pinned packages. Capture actual padded token counts, tensor dtypes, GradScaler state, operation profile and memory lifetime. Verify a route to the required clean Kaggle run immediately.
2. Compare dense and chunked head outputs/gradients on identical hidden states. Include ignored labels, variable lengths, large logits, argmax ties and several upstream scales. Compare unscaled adapter gradients as well as loss and metrics.
3. Time a small original tiled head projection/statistics prototype against its real baseline, including launch and compilation overhead.
4. Time an NF4 dX prototype on actual Qwen shapes (1536→1536, 1536→256, 1536→8960, 8960→1536). Preserve original NF4 groups, nibble order, nested scales and rounding. Keep the existing forward and PEFT LoRA branch initially.
5. Select one major GEMM direction based on T4 results. Keep the NF4 spike bounded to roughly four hours; remove it if it loses overall.
6. Integrate and measure original-kernel share from the unchanged harness. A result just over 5% is fragile; the harness warns below 6%. Never add busywork to meet the threshold.

Use RTX 5090 for explicit fp16 development tests. Do not select a T4 design using 5090 timings or a default bf16 training run. Colab T4 evidence is provisional; the final deliverable is Kaggle T4.

Approximately 30 hours remained at the start of this assessment. Reserve at least the final 6–8 hours for complete cold-cache runs, all regression tests, adapter evaluation, Kaggle reproduction, documentation, presentation and the ≤3 minute video. Do not spend the entire window on a fully fused backward kernel.

## Evidence and sources

- [Official starter and rules](https://github.com/MakazhanAlpamys/soup-kernel-challenge)
- [Soup loss chunking documentation](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/performance-and-quantization.md) — implementation also inspected in the exact 0.77.0 PyPI wheel.
- [Cut Your Losses paper](https://arxiv.org/abs/2411.09009)
- [CCE implementation requirements](https://github.com/apple-aiml-research/ml-cross-entropy)
- [Liger paper](https://arxiv.org/abs/2410.10989)
- [Pinned bitsandbytes autograd](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py)
- [Pinned bitsandbytes CUDA dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py)
- [Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)

Historical research plan above; resume from docs/STATE.md and current measured evidence. The immediate blocker is useful original kernel coverage, not GPU access.
