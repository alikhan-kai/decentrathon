> **Current source validation:** this branch's runtime is `47aede7`, measured at **50.83 s / 3039 MiB on Kaggle T4**, all 10 checks passing. **Sub-40 seconds is not achieved.** See [the full report and raw evidence](KAGGLE_VALIDATION_47A.md). Original submission documentation below describes separately pinned `fabc67d`, not this branch's current runtime.

# Submission / Сдача: localhost (TODO: confirm official team name)

## English

### Links

- **Repository (team public):** [alikhan-kai/decentrathon](https://github.com/alikhan-kai/decentrathon)
- **Repository (hackathon platform):** [BAITC-Hacks/hack-d2097b50-localhost](https://github.com/BAITC-Hacks/hack-d2097b50-localhost)
- **Final benchmarked runtime commit:** [`fabc67d4b0667be70b846aac29d9be21cb31a212`](https://github.com/alikhan-kai/decentrathon/commit/fabc67d4b0667be70b846aac29d9be21cb31a212) ([hackathon commit link](https://github.com/BAITC-Hacks/hack-d2097b50-localhost/commit/fabc67d4b0667be70b846aac29d9be21cb31a212))
- **Public Kaggle notebook (`COMMIT` pinned to final SHA):** [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final)
- **Demo video (up to 3 minutes):** TODO(DEMO_VIDEO_URL)
- **Presentation:** TODO(PRESENTATION_URL)

The Kaggle notebook URL and final benchmarked SHA match [README.md](README.md) and [README.ru.md](README.ru.md). Run **F** is the paired two-repeat (`--repeats 2`, ABBA) measurement of `fabc67d4b0667be70b846aac29d9be21cb31a212` (`solution_sha256`: `7caf827d0fb87cc23400afb227f2efb17e21fcca5ebd94867732f71f0171285c`, `result_sha256`: `fbf7553e8ca4df347be80f567bf311e92cec21275c65270767cf2d7a89fc8585`); its [checks.json](docs/measurements/submission_fabc67d/checks.json) and full breakdown in [docs/EVIDENCE.md](docs/EVIDENCE.md) are tracked in Git, and those measurements apply to the pinned `fabc67d` source; this branch uses `47aede7`.

### Description

Custom `@triton.jit` kernels accelerate nested double-quantized NF4 unpacking (`nf4_dequant_kernel`), forward/backward RMSNorm (`rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel`), joint Q/K split-half rotary position embedding (`ember_rope_joint`) and chunked LM-head cross-entropy/entropy/accuracy (`head_loss_kernel`). Together with within-batch padding removal, native FP16 QKV / fused LoRA epilogue / RoPE / batched `repeat_kv`, lossless FP16 tied embedding/head storage, budgeted FP16 MLP activation saves (`BUDGET_ROWS=940`), startup overhead elimination and background 9-specialization Triton precompilation, the solution achieves **1.5903× speedup** and **4.3748× lower peak GPU memory (`3 279 MiB` vs `14 345 MiB`)** while preserving the exact Soup/Qwen2.5-1.5B LoRA task and standard PEFT adapter output.

### Team (up to 4 people)

| Name / Handle | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Ansar | Integrator, Head Loss / NF4 / RMSNorm / Packed Decoder | `iencislennon` | `iencis` | TODO(CAPTAIN_CONTACT) |
| TODO(MEMBER_2) | Rotary Kernel, Lossless Tied Weight, Native/Hybrid MLP & Precompile | `stritphatographir` | TODO | — |
| Alikhan | Startup Overhead, Fused LoRA Epilogue & Nested NF4 Integration | `alikhan-kai` | TODO | — |
| TODO(MEMBER_4_IF_ANY) | Documentation, Video & Presentation | TODO | TODO | — |

### Our kernels

| Kernel (`@triton.jit` name) | File | What it replaces | Share of timed training in F, two repeats (Check 4) |
|---|---|---|---:|
| `nf4_dequant_kernel` | [`solution/nf4.py`](solution/nf4.py) | bitsandbytes flat & nested double-quantized NF4 unpacking | 8.2659–8.3073% |
| `head_loss_kernel` | [`solution/head_loss.py`](solution/head_loss.py) | Full-vocab logits materialization + separate CE loss, entropy, accuracy & `dlogits` | 0.7654–0.7693% |
| `rmsnorm_bwd_kernel` | [`solution/rmsnorm.py`](solution/rmsnorm.py) | Qwen2 RMSNorm backward input-gradient elementwise/reduction chain | 0.7158–0.7302% |
| `rmsnorm_fwd_kernel` | [`solution/rmsnorm.py`](solution/rmsnorm.py) | Qwen2 RMSNorm forward variance, `rsqrt` & weight-multiply chain | 0.6241–0.6469% |
| `ember_rope_joint` | [`solution/rotary.py`](solution/rotary.py) | Separate Q and K split-half rotary forward/backward elementwise chains | 0.5526–0.5746% |

Per-kernel shares are computed per repeat as `100 × own_kernel_ms / (1000 × timed_loop_s)` from `F` (`solution_1`: `timed_loop_s = 31.660 s`, total `11.0244%`; `solution_2`: `timed_loop_s = 32.218 s`, total `10.9276%`). **Combined eligible share in the lowest repeat: `10.9%`** (Check 4 requires `≥ 5.0%`). Starter `example_swiglu_fwd_kernel` and `example_swiglu_bwd_kernel` are **excluded** from our kernel share; `_rope_rotate_block` is an inlined device helper; `residual_norm_*` is disabled.

Ideas taken from other kernels / references (all Triton kernels above were written by our team):

- **Chunked LM-head + loss:** Makazhan Alpamys and Soup contributors, [`stream_loss_chunk`](https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/utils/stream_loss_chunk.py) (Apache-2.0).
- **NF4 & double quantization:** Tim Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314); reference layout in [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py).
- **RMSNorm & RoPE arithmetic:** Hugging Face Transformers contributors, [`modeling_qwen2.py` (v5.16.1)](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py).
- **Starter SwiGLU (`example_swiglu_*`):** Organizer [starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge) (no original-kernel credit claimed).

### Results on Kaggle T4 (`--repeats 2`, ABBA, public rows, seed 1)

| Metric | Base Soup | Ours (`fabc67d`) | Ratio / Delta |
|---|---:|---:|---|
| Time of `soup train`, s (median of 2) | 84.59 | 53.19 | **Speedup 1.5903×** |
| Peak GPU memory, MiB (max of 2) | 14345 | 3279 | **Base/ours 4.3748×** |
| Eval loss (public held-out rows) | 2.04965 | 2.04941 | **−0.012% vs base** |

- [x] All 10 checks of `harness/check_rules.py` passed (`2 base + 2 solution` runs, 79 Soup regression tests passed in `0.49s`).
- [x] `README.md` (and `README.ru.md`) rewritten for our solution: what we did, how to run it, results and limitations.
- [x] `submission/kaggle_notebook.ipynb` updated with `COMMIT = "fabc67d4b0667be70b846aac29d9be21cb31a212"` and `--repeats 2`.
- [ ] Verify public Kaggle saved-version URL (`?scriptVersionId=...`) after final **Save & Run All** and fill remaining `TODO(...)` fields (captain contact, video, presentation).

### Presentation covers

1. Bottleneck profile of baseline Qwen2.5-1.5B NF4 LoRA training on T4 (`14.3 GiB` VRAM spike from full-vocabulary FP32 logits and padded FP32 activations; Python/import/compilation startup overhead; 25K NF4 dequantization launches).
2. Architecture of our 5 custom Triton kernels (`nf4_dequant_kernel` with inline-PTX IEEE-754 nested scale fusion, `head_loss_kernel` online softmax + scaled backward compaction, `rmsnorm_fwd/bwd_kernel`, and FP16 `ember_rope_joint`).
3. Memory & compute co-design: within-batch padding removal with isolated causal SDPA, lossless FP16 tied embedding/head compaction (`-467 MiB`), budgeted native-FP16 / rematerialized MLP saves (`3 279 MiB` steady peak), and 9-spec background Triton precompilation.
4. Verified 2-repeat ABBA results (`1.5903×` speedup, `4.3748×` memory reduction, `−0.012%` eval loss, `10.9%` own-kernel share, 10/10 checks PASS), limitations and next steps.

### Limitations

Fast paths are specialized and guarded for Tesla T4 (`sm_75`), FP16 autocast and the pinned challenge package versions, falling back cleanly otherwise. Above `BUDGET_ROWS = 940` packed tokens, upper MLP layers trade extra backward recomputation for the `3 279 MiB` memory ceiling. A 3D `[16, 32, 2]` NF4 tile with register-tree lookup regressed coalescing on `sm_75` and was reverted; `residual_norm` fusion remains disabled. Full details are in [README.md](README.md#limitations-and-unsuccessful-work).

---

## Русский

### Ссылки

- **Репозиторий команды:** [alikhan-kai/decentrathon](https://github.com/alikhan-kai/decentrathon)
- **Репозиторий платформы хакатона:** [BAITC-Hacks/hack-d2097b50-localhost](https://github.com/BAITC-Hacks/hack-d2097b50-localhost)
- **Финальный измеренный коммит кода:** [`fabc67d4b0667be70b846aac29d9be21cb31a212`](https://github.com/alikhan-kai/decentrathon/commit/fabc67d4b0667be70b846aac29d9be21cb31a212) ([ссылка в репо хакатона](https://github.com/BAITC-Hacks/hack-d2097b50-localhost/commit/fabc67d4b0667be70b846aac29d9be21cb31a212))
- **Публичный ноутбук Kaggle (`COMMIT` закреплён на финальном SHA):** [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final)
- **Демо-видео (до 3 минут):** TODO(DEMO_VIDEO_URL)
- **Презентация:** TODO(PRESENTATION_URL)

### Наши кернелы

| Имя `@triton.jit` | Файл | Что заменяет | Доля времени обучения в F, 2 повтора (Check 4) |
|---|---|---|---:|
| `nf4_dequant_kernel` | [`solution/nf4.py`](solution/nf4.py) | Обычную и вложенную (`double-quant`) распаковку NF4 в bitsandbytes | 8.2659–8.3073% |
| `head_loss_kernel` | [`solution/head_loss.py`](solution/head_loss.py) | Полные logits словаря + раздельные loss, entropy, accuracy и `dlogits` | 0.7654–0.7693% |
| `rmsnorm_bwd_kernel` | [`solution/rmsnorm.py`](solution/rmsnorm.py) | Цепочку операций градиента входа Qwen2 RMSNorm | 0.7158–0.7302% |
| `rmsnorm_fwd_kernel` | [`solution/rmsnorm.py`](solution/rmsnorm.py) | Прямой проход Qwen2 RMSNorm (`var`, `rsqrt`, умножение на вес) | 0.6241–0.6469% |
| `ember_rope_joint` | [`solution/rotary.py`](solution/rotary.py) | Раздельные поэлементные цепочки RoPE для Q и K (forward и backward) | 0.5526–0.5746% |

**Суммарная зачётная доля наших кернелов в худшем повторе: `10.9%`** (при требовании Check 4 `≥ 5.0%`). Кернелы стартового примера `example_swiglu_*` в зачёт не входят.

### Результаты на Kaggle T4 (`--repeats 2`, ABBA, публичные строки, seed 1)

| Показатель | База Soup | Наше решение (`fabc67d`) | Сравнение |
|---|---:|---:|---|
| Время всей команды `soup train`, с | 84.59 | 53.19 | **Ускорение 1.5903×** |
| Пик памяти GPU, MiB | 14345 | 3279 | **База/решение 4.3748×** |
| Eval loss (отложенная выборка) | 2.04965 | 2.04941 | **−0.012% (лучше базы)** |

- [x] Все 10 проверок `harness/check_rules.py` пройдены (`2 base + 2 solution`, 79 тестов Soup за `0.49 с`).
- [x] `README.md` и `README.ru.md` полностью обновлены под финальное решение.
- [x] В `submission/kaggle_notebook.ipynb` закреплён `COMMIT = "fabc67d4b0667be70b846aac29d9be21cb31a212"` и `--repeats 2`.
- [ ] После финального **Save & Run All** сделать ноутбук Public на Kaggle и вставить ссылки на сохранённую версию, видео, презентацию и контакт капитана.
