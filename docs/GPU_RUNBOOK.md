# GPU validation runbook

Run all commands from the repository root with the intended GPU environment's
`python`. The development workspace has CPU-only PyTorch and no NVIDIA GPU;
it cannot validate CUDA kernels or produce performance results. Initial Kaggle results and the current admission failure are recorded in
`docs/validation/`; final reproduction still follows this runbook.

Use the RTX 5090 for explicit fp16 correctness checks and iteration, Colab T4
for provisional integration evidence, and a clean Kaggle T4 session for the
submission. The unchanged Soup harness chooses bf16 on newer hardware, so a
5090 harness run does not establish the T4 fp16 path or an official score.

## 1. Verify the GPU environment

The pinned application stack is `soup-cli==0.77.0`, `transformers==5.16.1`,
`peft==0.20.0`, `trl==0.29.1`, `bitsandbytes==0.50.2`, and
`accelerate==1.14.0`. The development environment has `torch==2.11.0+cpu`
and `triton==3.6.0`; copying that CPU environment onto a GPU host is insufficient.
Use a CUDA-enabled PyTorch build and driver compatible with the target GPU,
and record the actual torch/CUDA/Triton versions. Prepare dependencies before
any measured harness run; do not install packages during a run.

The first actual Kaggle probe exposed an unrelated preinstalled `torchao==0.10.0`
that PEFT 0.20 rejects during adapter setup. Remove that unused optional package
in notebook setup, before running either baseline or candidate:

```bash
python -m pip uninstall -y torchao
```

This is an environment setup fix for both implementations; no package changes
are made inside the measured training commands. The provided notebook includes
this step and readiness guards so a failed setup cannot produce benchmark claims.

```bash
nvidia-smi
python - <<'PY'
import importlib.metadata as md
import torch

for package in ('torch', 'triton', 'soup-cli', 'transformers', 'peft', 'trl',
                'bitsandbytes', 'accelerate'):
    print(package, md.version(package))
print('torch CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA PyTorch and a visible NVIDIA GPU are required'
print('GPU:', torch.cuda.get_device_name(0))
print('compute capability:', torch.cuda.get_device_capability(0))
PY
```

In Kaggle select **GPU T4 x2** and enable Internet for initial dependencies,
model and public-data downloads. The account must have GPU access. Use the
platform-created team repository as the authoritative submission checkout;
record its exact revision before the final run. In Colab select a GPU runtime
and verify the allocated GPU name with `nvidia-smi`.

Run diagnostics, tests and training sequentially. **Never leave this runner,
another training process, or a notebook-held CUDA model occupying a GPU during
the official harness.** Finish diagnostic processes and release notebook GPU
objects (restart the kernel if needed), then inspect `nvidia-smi`. Both Kaggle
GPUs must be idle before measurement: the harness sums GPU memory across
devices, even though training is restricted to GPU 0. Do not modify the
harness's allocator settings, device selection, compilation caches or task.

## 2. Run the bounded component diagnostics

```bash
python -m solution.benchmark --help
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark --component head --json runs_diagnostics/head_m128.json
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark --component rmsnorm --json runs_diagnostics/rmsnorm_fp32.json
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark --component rmsnorm --input-dtype fp16 --json runs_diagnostics/rmsnorm_fp16.json
```

Each invocation selects exactly one component. Defaults are 128 total token
rows, two warmup invocations and five timed invocations per implementation.
`--warmup` is bounded to 1–20 and `--iterations` to 1–100; there is no adaptive
timing loop. Initial compilation and parity checks add untimed work, so these
limits are invocation bounds, not a wall-clock timeout. `--device` selects a
logical CUDA index after `CUDA_VISIBLE_DEVICES` filtering.

All projection/timing paths explicitly use CUDA fp16 autocast, including on
RTX 5090. Head and RMSNorm inputs default to fp32, matching the frozen-fp32
normalization/residual boundary; `--input-dtype fp16` also exercises the half
input boundary. Autocast does not turn every RMSNorm operation into fp16:
the independent oracle preserves Qwen's fp32 normalization and dtype casts.
TF32 and reduced-precision fp16 matmul reductions are disabled for these
diagnostics. This runner does not change the official training configuration
or call `solution.apply()`.

The head uses the actual Qwen width 1536 and vocabulary 151936. `M` means
total tokens: default `--m 128 --batch-size 4` is `[4,32,1536]`, and
`--m 512` is `[4,128,1536]`. Projection chunks default to 128 token rows.
The frozen fp32 vocabulary weight and the dense oracle are deliberately real
allocations; peak memory is not limited to the chunk alone. Increase `M`
only after the default case passes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark --component head --m 512 --json runs_diagnostics/head_m512.json
```

`--m 2048 --batch-size 4` selects the maximum `[4,512,1536]` shape and costs
substantially more memory and compute. `--hidden-size`, `--vocab-size` and
`--chunk-size` can reduce a failing case; results from reduced shapes are
not full-Qwen evidence.

Before any timings, head parity compares against independent dense PyTorch
projection, cross-entropy, entropy, argmax and autograd. The synthetic fixture
includes prompt masking, right padding and a prompt-only row when batch > 1.
It checks:

- Causally shifted supervised labels for loss, accuracy and row token counts.
- Entropy and attention-token counts over every attention-valid position,
  including prompt positions and the last real token.
- Row loss sums/means, an explicit `num_items_in_batch` different from the
  local supervised count, and the default denominator.
- Input gradients with upstream multipliers 1 and **65536**, comparing the
  unscaled gradients after backward. Timed head calls use scale 65536.

This is a loss-scaling contract check, not an Accelerate/GradScaler training
run. RMSNorm separately checks its output and input gradient against a plain
PyTorch graph. Full model, adapter and sampler behavior require the tests and
official harness below.

### Optional NF4 experiment

NF4 is experimental and disabled in the default solution integration. It runs
only when explicitly selected, without monkeypatching bitsandbytes or training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark --component nf4 --nf4-shape 1536x1536 --json runs_diagnostics/nf4_q_m128.json
```

`--nf4-shape` specifies packed weight shape `[N,K]`, with
`dY[M,N] @ W[N,K] -> dX[M,K]`. Available real Qwen shapes are `1536x1536`
(q/o), `256x1536` (k/v), `8960x1536` (gate/up), and `1536x8960` (down).
For example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark --component nf4 --nf4-shape 256x1536 --m 512 --json runs_diagnostics/nf4_v_m512.json
```

The runner quantizes random fp16 weights through bitsandbytes with NF4,
blocksize 64 and double quantization. It checks the original input-gradient
kernel against both bitsandbytes dequantization plus fp16 matmul and an
independent fp32 PyTorch matmul of the bitsandbytes-dequantized weights.
The timed baseline recreates the dense temporary on every call. Quantization
setup is excluded; this compares only the input-gradient operation, not an
NF4 layer's forward pass or end-to-end training. Existing tests cover additional
packing, scale rounding, layout and non-nested cases.

### Read the diagnostic report correctly

The runner writes JSON to stdout after all parity checks and timing samples
succeed. Progress/errors go to stderr. `--json PATH` saves the same report;
on failure it does not create a success report or replace a previous file.
Always check the exit code and report timestamp before using an existing file.
No-CUDA exits with code 2 and no numbers; parity/runtime failure exits nonzero.

The report records GPU name, UUID, compute capability, total device memory,
driver inventory, Python/package versions, actual CUDA runtime, shapes,
precision settings, seed, tolerances and maximum observed absolute errors.
Argmax, accuracy and token counts require exact agreement; other tolerances
are explicit in each parity entry. A mismatch stops the run before timing.

CUDA-event samples cover one complete selected operation per sample, after
warmup, with synchronization after each sample. Head and RMSNorm include
forward and input backward; NF4 includes only input backward. These are
synthetic steady-state diagnostics, not whole-command wall time. Cold
compilation, model loading, optimizer work and adapter saving are not timed.

`peak_allocated_bytes` is PyTorch's peak allocated tensor memory after warmup;
it includes the shared resident input fixture. `allocated_before_timing_bytes`
records that fixture and `peak_increment_over_inputs_bytes` reports the
additional allocated peak. Peaks are reset per implementation. These fields
exclude reserved allocator memory, CUDA context and other processes. They
are **not the official nvidia-smi peak memory metric**, and neither their
ratios nor CUDA-event ratios establish an official score or kernel share.

## 3. Run kernel and Soup regression tests

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
CUDA_VISIBLE_DEVICES=0 SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

Read the skip summary: a passing CPU suite with CUDA tests skipped does not
validate GPU kernels. On the 5090, the head CUDA tests use explicit fp16 AMP.
The patched Soup regression suite is also required because its lightweight
fixtures exercise fallback guards. Do not modify `tests/` to make it pass.

After the bounded NF4 experiment passes, the larger optional Qwen tests can
be run separately (these use up to 2048 token rows):

```bash
CUDA_VISIBLE_DEVICES=0 NF4_FULL_GPU_TESTS=1 python -m pytest solution/test_nf4.py -q
```

## 4. Run the unchanged end-to-end harness on T4

Download only the prescribed public data, then stop all other GPU work before
the harness. Keep `harness/`, `task/` and `tests/` unchanged.

```bash
python task/prepare_data.py
nvidia-smi
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

`prepare_data.py` writes `data/train.jsonl` and `data/eval_public.jsonl`.
`--repeats 2` runs base, solution, solution, base, then evaluates the saved
adapters. It measures the complete train subprocess with fresh compiler
caches. Use the harness's own single-GPU/allocator setup without overrides.
`check_rules.py` reads `runs/result.json`, reruns Soup regression tests and
writes `runs/checks.json`; do not use `--skip-tests` for final evidence.
Run these commands sequentially and inspect every exit code/output, including
the check table. A result file alone does not mean the checks passed.

For provisional iteration, use a separate output directory:

```bash
python harness/run_task.py --only solution --out runs_try
```

That run lacks a paired baseline and cannot satisfy all comparison checks.
For optional operation profiling, budget an additional training invocation:

```bash
python harness/run_task.py --repeats 2 --profile --out runs_profile
python harness/check_rules.py --result runs_profile/result.json
```

Read the resulting `runs_profile/solution_profile/profile.json`; the extra
profile run is not a replacement for the paired measured runs. Kernel share
must come from launches that perform real training work in the unchanged
harness, not this diagnostic CLI. Check the mandatory 5% original-kernel share
on every solution repeat (a result close to the boundary is fragile).

If any parity, output-equivalence, quality, kernel-share or memory/time check
fails, retain the failure logs and fix the implementation before reporting
admission or gains. Increasing artificial kernel work is not a valid fix.

## 5. Preserve the final evidence

Keep `runs/result.json`, `runs/checks.json`, per-run `train.log`, evaluation
logs, the saved adapters and their trainer state/rewind files. Keep optional
diagnostic JSON in `runs_diagnostics/`, separate from official result files.
Record the exact source revision and package/GPU versions with the evidence.
Do not revise solution code after a final run without rerunning the checks
and measurements for the new code.

The submission workflow also covers README/STATE/SUBMISSION, notebook creation,
publication links, presentation and video. Final submission evidence must
come from the pinned revision in a saved Kaggle T4 notebook. Neither a 5090
diagnostic nor a Colab T4 run substitutes for that requirement. This runbook
contains commands and acceptance criteria; it asserts no GPU performance.
