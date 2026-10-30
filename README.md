<img src="assets/icon.svg" width="64" height="64" alt="">

# Ember — diagnostic-preserving QLoRA kernels

Original Triton work for the [Soup Kernel Challenge](https://github.com/MakazhanAlpamys/soup-kernel-challenge) at Decentrathon 6.0. The target is the unchanged Qwen2.5-1.5B QLoRA training task on a single Kaggle T4.

**Status: initial Kaggle T4 run preserves training quality and reduces peak memory by 28.7%, but fails the minimum original-kernel share (2.1% versus 5%). Not submission-ready. These are one-repeat development results.**

The output head processes 128 tokens at a time. Original split-vocabulary kernels compute loss, entropy and argmax; backward recomputes each chunk and applies the actual upstream loss scale before converting gradients to fp16. Original RMSNorm kernels preserve Qwen's intermediate dtype boundaries. A separate NF4 input-gradient kernel is experimental and is not enabled by `solution.apply()`.

The implementation preserves the standard LoRA parameters, dropout, optimizer, batches, token counts and per-row Rewind records. Entropy covers every attention-valid position, including prompt tokens. Ordinary model calls and evaluation retain the original logits path. No Soup optimization switch is enabled.

## Run

Use the platform-created team repository and [the Kaggle notebook](notebooks/ember_kaggle.ipynb). Fill in its repository URL, select GPU T4 ×2 and Internet on, and run all cells. For submission, pin the final commit and save a public executed notebook version. **Public version link: pending.**

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" \
  "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python task/prepare_data.py
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

CUDA PyTorch and Triton are required for GPU runs. Local reference checks used CPU PyTorch 2.11.0 and Triton 3.6.0. The official harness records actual runtime versions. Only the prescribed Kaggle T4 measurements count toward the challenge.

## Evidence so far

- All **79 official regression tests pass** with `solution.apply()` enabled locally and on Kaggle T4.
- On Kaggle T4, the solution suite reports **254 passed, 17 skipped** (optional interpreter/full-shape experiments). This includes CUDA kernel checks and a prepared fp16 training step using Accelerate's GradScaler. [Recorded environment and evidence](docs/validation/2026-10-30-t4-tests.json).
- Head checks match independent dense PyTorch loss, all-position diagnostics and hidden-state gradients, including non-default normalization and fully ignored rows.
- Real Qwen2 + PEFT + TRL integration checks pass, including dropout-seeded adapter gradients, Rewind row identities, evaluation fallback, one training step and saved adapter files.
- Vocabulary and RMSNorm kernels also compile offline for sm75. The initial end-to-end run passes quality and diagnostic checks, but fails original-kernel share. Final repeated measurements remain pending.
- `harness/`, `tests/`, and `task/` match the official starter revision recorded in [the integrity manifest](docs/starter-integrity.json).

| Official Kaggle measurement | Baseline | Ember |
|---|---:|---:|
| Full training command, seconds | 88.27 | 86.91 |
| Peak GPU memory, MiB | 14,345 | 10,223 |
| Held-out adapter loss | 2.04962 | 2.04963 |
| Original kernel share | — | 2.1% — fails |

One baseline and one candidate run through the unchanged official harness on Kaggle T4, source bundle `abdc0fc4256a968cfebb7bce915b9666708e2c469b3dba6a0ebe7f435ee2abf9`. Runtime improvement (1.016×) is below the 1.05× scoring threshold. All checks except the original-kernel share pass; neither submission readiness nor a repeatable gain is established.

## Limits and credit

Streaming linear cross-entropy is established by [CCE](https://arxiv.org/abs/2411.09009), [Liger](https://arxiv.org/abs/2410.10989), and Soup's existing chunked head. We claim original kernel code and integration work, not invention of chunked loss. We do not import their kernels or enable Soup's built-in chunked implementation. The existing per-token Rewind interface is reused.

The current head uses ordinary GEMMs and recomputes the projection during backward. That adds compute and may fail the time gate. Reductions plus RMSNorm account for only 2.1% of training wall time in the first run, below the required 5%; useful additional original kernel work is necessary. The NF4 experiment must beat the actual bitsandbytes backward before promotion; Triton 3.6 emits scalar FMA for its sm75 dot operation, so it stays disabled. T4 measurements determine which path survives.

The head supports a frozen, bias-free Qwen2 head with fp16/fp32 projection, ordinary single-device training and first-order gradients. Unsupported training configurations use the original trainer path. See [technical direction](docs/TECHNICAL_DIRECTION.md), [starter rules](docs/STARTER_README.md), and [submission draft](SUBMISSION.md).
