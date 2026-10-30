# Ember — submission draft, not ready to submit

Initial Kaggle T4 tests and training-quality checks pass. The first measured run reduces peak memory by 28.7%, but fails the original-kernel-share gate (2.1% versus 5%). This draft is not eligible yet.

## Links

- User-designated team repository: https://github.com/alikhan-kai/decentrathon (final commit pending).
- Public Kaggle notebook saved version: pending final pinned execution.
- Demo video (up to 3 minutes): pending validated measurements.
- Presentation: pending validated measurements.
- Team names and public contact information: pending.

## Original kernels

| Kernel | Source | Purpose | Measured share |
|---|---|---|---|
| `ember_vocab_partials`, `ember_vocab_merge` | `solution/vocab.py` | Joint loss, entropy, argmax and log-sum-exp statistics | pending |
| `ember_vocab_backward_mass`, `ember_vocab_backward_write` | `solution/vocab.py` | Scaled fp16 logit gradients with stable normalization | pending |
| `_rmsnorm_forward`, `_rmsnorm_backward` | `solution/rmsnorm.py` | Frozen-weight Qwen RMSNorm with preserved cast boundaries | pending |
| `_decode_nested_scales`, `_nf4_input_grad_kernel` | `solution/nf4.py` | Experimental NF4 dX; disabled in submission entry point | not active |

Chunked loss ideas: [CCE](https://arxiv.org/abs/2411.09009), [Liger](https://arxiv.org/abs/2410.10989), and [Soup](https://github.com/MakazhanAlpamys/Soup). Kernel code was written independently. NF4 format and baseline behavior were checked against [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2).

## Kaggle T4 results

One development repeat of the unchanged official harness on Kaggle T4; final repeated and pinned submission results remain pending.

| Metric | Base | Ember | Ratio |
|---|---:|---:|---:|
| Full command seconds | 88.27 | 86.91 | 1.016× speedup |
| Peak GPU memory MiB | 14,345 | 10,223 | 1.403× base/candidate |
| Public held-out loss | 2.04962 | 2.04963 | effectively unchanged |

- [ ] All official rule checks pass on the final Kaggle run.
- [ ] Original kernels exceed the required share in every repeat.
- [ ] Public notebook is pinned to the final platform repository commit.
- [ ] README and this submission contain verified results and artifact links.
- [ ] Presentation and video accurately show benefits and limitations.

## Remaining risks

Original kernels currently occupy only 2.1% of training time, which fails eligibility. Repeated final measurements and public notebook reproduction are still required. The optional NF4 direction has a larger compute scope but no demonstrated speed benefit.
