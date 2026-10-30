# Контракты модулей

Общая схема: `harness/_train.py` импортирует `solution` и вызывает `solution.apply()` до `soup train`. `apply()` идёт по
`MODULES = ["swiglu", "head_loss", "packed", "rmsnorm", "nf4", "lora"]`: для каждого модуля, у которого `ENABLED = True`, вызывает
`patch()`. Ошибка в `patch()` не глотается (иначе не заметим, что кернел не работает). Список `MODULES` меняет только integrator.

## Правила для любого модуля
- Только свой файл `solution/<модуль>.py` и `solution/test_<модуль>.py`. Не править чужие файлы, `__init__.py`, `harness/`, `tests/`, `task/`.
- `ENABLED = False`, пока кернел не готов. `patch()` идемпотентен (флаг `_patched`) и узкий: подменяет только свой метод/класс.
- В подменённой функции: если входы не подходят (не CUDA, другой dtype, пустой тензор, другой адаптер), вызвать оригинал.
  `apply()` выполняется и перед регрессионными тестами Soup (проверка 10), где transformers/peft подменены заглушками.
- Нет `@triton.autotune`, CUDA graphs, `os.environ`/`os.getenv`, поведения, зависящего от замера (`tools/lint_rules.py` это ловит).
- Каждый `@triton.jit` кернел запускается как `kernel[grid](...)` в обучении (холостые запрещены, проверка 3).
- Не добавлять обучаемые параметры; градиенты те же (проверка 5).
- Сравнение с PyTorch: `solution/kcheck.py` (`check_fwd_bwd`), допуски явные: fp16 atol = rtol = 2e-3, fp32 1e-4.

## Кто что патчит (имена из исходников transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2)
| модуль | патчит | не пересекается с |
|---|---|---|
| `swiglu` (пример) | `modeling_qwen2.Qwen2MLP.forward` | |
| `head_loss` | `modeling_qwen2.Qwen2ForCausalLM.forward` (только размеченный, обучение) и `trl.SFTTrainer.compute_loss` | их не трогает никто другой |
| `rmsnorm` | `modeling_qwen2.Qwen2RMSNorm.forward`; шаг 2: `Qwen2DecoderLayer.forward` (остаточные сложения), `Qwen2Model.forward` (финальная норма) | |
| `packed` | `modeling_qwen2.Qwen2Model.forward` (в обучении упаковывает реальные токены в `[1, N, H]`, возвращает `packed_keep`), `Qwen2Attention.forward` (SDPA по строкам) | `head_loss` читает `packed_keep` из выхода `self.model(...)`; с `rmsnorm` шаг 2 (`Qwen2Model.forward`) договариваться через integrator |
| `nf4` | `bitsandbytes.backends.cuda.ops._dequantize_4bit_impl` (общая точка: forward через `gemm_4bit` -> `_dequant_linear_fallback`, backward через `F.dequantize_4bit` -> cuda-операция) | |
| `lora` | `peft.tuners.lora.bnb.Linear4bit.forward` | не трогать `lora_dropout` |
| (кандидат) rotary | `modeling_qwen2.apply_rotary_pos_emb` (функция модуля, `Qwen2Attention.forward` берёт её по имени при вызове) | |

`Qwen2DecoderLayer.forward` может патчить только владелец `rmsnorm` (шаг 2). Если двум модулям нужен один метод, договариваются
через integrator, а не правят оба.

## dtype-контракт (OP/op_profile.json на T4; см. `docs/FINDINGS.md`, раздел 5)
- Остаточный поток fp32 (выход `DecoderLayer` и `Embedding`). `RMSNorm`: выход fp32, вес fp32, `requires_grad=False`;
  входной dtype не записывали (по цепочке fp32, это вывод). Кернел должен вернуть ровно тот dtype, что и база.
- `Attention` выход fp16. `MLP` выход fp32. peft `Linear4bit` выход fp32. bnb `Linear4bit` выход fp16 или fp32.
- LoRA: `lora_A`, `lora_B` вес fp32, `requires_grad=True`, выход fp16; `scaling` = alpha / r = 2.0 (32 / 16); dropout 0.05.
- NF4: вес `uint8` (упакован), двойное квантование: `quant_state.nested`, `state2.blocksize == 256`; на выходе fp16 `[out, in]`.
- `apply_rotary_pos_emb` выход fp32.
Если кернел меняет dtype на границе, он должен вернуть его обратно на выходе, иначе меняются значения (проверки 5-6).

## Как проверить, что кернел реально используется
После `quick`: `metrics.json` решения (`runs_try/solution_1/metrics.json`): `triton_launches[<имя кернела>] > 0`,
`triton_files[<имя>] == "solution/<файл>.py"`, `own_kernel_ms[<имя>]`. `tools/summarize.py <папка вывода>` печатает таблицу «наши кернелы»
с мс/шаг и долей. Ожидаемые числа запусков за шаг: rmsnorm 57 + 57 (fwd + bwd), nf4 около 389, lora 56 + 56, head_loss около 2
(138 запусков за 64 шага). Харнесс считает запуски за все 64 шага, время кернелов только с 3-го.
Если нужно увидеть, в какой модуль идёт время: `MODE=check_and_ops` (профиль по модулям, `tools/op_profile.py`).

## Как не затереть метрики Soup и rewind
- `head_loss.py` уже владеет путём loss: `Qwen2ForCausalLM.forward` (без logits) и `SFTTrainer.compute_loss`. Он пишет
  `_soup_chunked_step` (его читает рекордер rewind) и метрики `entropy`, `num_tokens`, `mean_token_accuracy` по формулам TRL.
  Остальные модули не меняют ни `compute_loss`, ни `forward` каузальной модели, ни возвращаемые объекты (`CausalLMOutputWithPast`).
- Не менять число токенов на шаг, порядок и содержимое батчей, форму выходов слоёв (проверка 5: токены, rewind, entropy и
  accuracy в пределах 3%, row loss тоже).
- Конфиг адаптера и имена тензоров LoRA остаются прежними: никаких новых `nn.Parameter` и переименования модулей.
