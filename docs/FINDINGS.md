# Что мы знаем (цифры только из логов и json запусков на T4)

Источники лежат в `docs/measurements/`:
- **FP** = `docs/measurements/full_profile_fe282f8/`: `full_profile` на коммите fe282f8 (база и решение по 2 запуска, проверки, профили).
- **OP** = `docs/measurements/check_and_ops_9995a69/`: `check_and_ops` на коммите 9995a69 (pytest, один запуск решения, профиль по операциям и модулям).
Железо: Tesla T4, torch 2.11.0+cu128, triton 3.6.0, transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2 (`FP/result.json`, `versions`).

## 1. База и решение (FP/result.json, summary)
| | base | solution (fe282f8) |
|---|---|---|
| wall, с (медиана 2 запусков) | 84.7 (запуски 86.9 и 82.5) | 73.38 (72.78 и 73.99) |
| пик памяти nvidia-smi, MiB | 14345 | 8385 |
| train_loop, с | 56.23 | 49.52 |
| медианный шаг, с | 0.918 | 0.7288 |
| eval loss | 2.04966 | 2.04969 (+0.001%) |

speedup 1.1543, memory_ratio 1.7108. Проверки (FP/checks.json): 1, 2, 3, 5, 6, 7, 8, 9, 10 PASS; **4 FAIL: 0.4%** (нужно 5%).
Накладные расходы вне цикла (wall - train_loop): base 32.34 и 24.6 с, solution 23.58 и 24.15 с; `apply_s` решения 8.04 и 8.09 с.
Повторный запуск решения в OP: wall 77.37 с (один запуск), пик 8385 MiB, eval loss 2.04967, медианный шаг 0.7136 с.
Разброс wall между запусками решения 72.78 ... 77.37 с: выигрыш в секундах меньше этого разброса не отличим от шума.

Тесты (OP/status.txt и логи): `pytest solution` 7 passed, 9 skipped (шаблоны rmsnorm, nf4, lora пропущены, пока кернелов нет),
`pytest tests` с `SOUP_APPLY_SOLUTION=1` 79 passed.

## 2. Наши кернелы (OP/metrics.json, шаги с 3-го, 62 шага, timed_loop_s 43.15)
| кернел | запусков | мс | мс/шаг | доля |
|---|---|---|---|---|
| `head_loss_kernel` (наш) | 138 | 199.0 | 3.21 | 0.46% |
| `example_swiglu_fwd_kernel` (пример) | 1792 | 1108.3 | 17.88 | не считается |
| `example_swiglu_bwd_kernel` (пример) | 1792 | 1874.1 | 30.23 | не считается |

5% от шага = 34.80 мс/шаг, не хватает 31.59 мс/шаг.

## 3. Профиль по группам кернелов, мс/шаг (FP, 16 шагов, с накладными профайлера)
| группа | база | решение |
|---|---|---|
| matmul | 283.1 | 310.7 |
| elementwise/cast | 361.9 | 285.0 |
| softmax/CE | 55.6 | 3.0 |
| attention | 46.5 | 52.8 |
| NF4 dequant (`kDequantizeBlockwise<__half,512,64,8,2>`) | 35.8 | 39.8 |
| прочее | 27.8 | 21.5 |
| всего GPU | 810.6 | 712.9 |
Группы считает `tools/summarize.py` по подстрокам имён кернелов (правила выводятся в отчёте).

## 4. Профиль по модулям-источникам, решение (OP/op_profile.txt; мс/шаг, 16 шагов)
Всего GPU 740.6 мс/шаг, атрибутировано 740.8 (покрытие 1.000). Операция относится к самому внутреннему модулю, внутри
которого она запущена; backward привязан к forward-модулю по номеру последовательности autograd. «Исключая» значит без
вложенных модулей.

| модуль | вызовов/шаг | fwd | bwd | всего | доля |
|---|---|---|---|---|---|
| bnb `Linear4bit` (все matmul базы + NF4 dequant) | 196 | 227.0 | 204.3 | 431.3 | 58.2% |
| `Qwen2Attention` исключая вложенные (SDPA и прочее) | 28 | 26.0 | 51.7 | 77.7 | 10.5% |
| `Qwen2RMSNorm` | 57 | 16.0 | 46.1 | 62.1 | 8.4% |
| `Qwen2MLP` исключая Linear4bit (SwiGLU примера) | 28 | 17.6 | 29.5 | 47.0 | 6.4% |
| fused head + loss (`_FusedHeadLoss`, включая matmul головы) | 1 | 41.8 | 0.1 | 41.9 | 5.7% |
| `apply_rotary_pos_emb` | 28 | 13.6 | 18.4 | 32.0 | 4.3% |
| LoRA `nn.Linear` (lora_A, lora_B) | 112 | 6.3 | 11.0 | 17.3 | 2.3% |
| `Dropout` (LoRA dropout 0.05) | 56 | 5.1 | 7.4 | 12.5 | 1.7% |
| peft `Linear4bit` (эпилог LoRA: clone, cast, scale, add) исключая вложенные | 56 | 6.9 | 2.8 | 9.7 | 1.3% |
| `Qwen2DecoderLayer` исключая вложенные (остаточные сложения) | 28 | 6.0 | 1.5 | 7.5 | 1.0% |
| optimizer, embedding, прочее | | | | 1.8 | 0.2% |

Топ ATen-операций (мс/шаг): `aten::mm` 318.5 (643 вызова), `aten::copy_` 130.0 (1660), `aten::mul` 46.9 (850),
`_efficient_attention_backward` 38.4, `aten::add_` 33.9 (664), `SwiGLUBackward` 29.5, `bitsandbytes::gemm_4bit` 22.7 (196),
`bitsandbytes::dequantize_4bit` 18.1 (193), `SwiGLU` 17.6, `_efficient_attention_forward` 16.7, `aten::add` 14.3, `aten::addmm` 12.8.
Полный список и формы входов: OP/op_profile.txt.

## 5. dtype и вызовы за шаг (OP/op_profile.json)
- Остаточный поток fp32: выход `DecoderLayer` и `Embedding` fp32. Выход `RMSNorm` fp32; вес `RMSNorm` fp32, `requires_grad=False`.
- Выход `Attention` fp16, `MLP` fp32, peft `Linear4bit` fp32, bnb `Linear4bit` fp16 и fp32 (оба встречались), lora_A/lora_B
  fp16-выход, их вес fp32 с `requires_grad=True`; выход `Dropout` fp32; выход `apply_rotary_pos_emb` fp32.
- Вызовов за шаг: DecoderLayer 28, RMSNorm 57 (28 x 2 + финальный), Attention 28, MLP 28, rotary 28, bnb Linear4bit 196,
  peft Linear4bit 56 (q_proj и v_proj), lora Linear 112, Dropout 56, head_loss 1, Embedding 1.
- Вес bnb `Linear4bit` uint8 (упакованный NF4).

## 6. Чего мы НЕ знаем
- Из чего состоит self-время `bitsandbytes::gemm_4bit` (22.7 мс/шаг): код CUDA-реализации (`backends/cuda/ops.py:934`) не читали.
  Надёжная цифра по NF4 это 39.8 мс/шаг на уровне кернелов (раздел 3); aten-уровень даёт 18.1 + 22.7 = 40.8.
- Входной dtype `RMSNorm`: не записан; по цепочке (выход Embedding и DecoderLayer fp32) он fp32, это вывод, не измерение.
- Профиль по модулям для БАЗЫ (его снимали только для решения): раскладка базы по модулям не знаем.
- Откуда конкретно берётся пик 8385 MiB и вклад SwiGLU и head_loss в экономию памяти по отдельности: не замеряли.
- Профайлер добавляет накладные расходы: суммы по профилю (740.6 мс/шаг) выше реального шага (713.6 мс, OP) и это разные величины.
- Реальное время будущих кернелов (rmsnorm, nf4, lora): не знаю, пока не написаны; потолки в `docs/tasks/`.
- Поведение на скрытых строках жюри и разброс между сессиями Kaggle: не знаю.
