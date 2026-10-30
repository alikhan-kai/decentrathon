# Задача nf4: Triton-распаковка NF4 вместо bitsandbytes

Файл `solution/nf4.py`, тест `solution/test_nf4.py` (шаблон), ветка `feat/nf4`. **Критический путь к проверке 4.**

## Что в профиле
- Кернелы `kDequantizeBlockwise<__half, 512, 64, 8, 2>`: **39.8 мс/шаг** (5.6% GPU-времени решения), 6224 запуска за 16 шагов = 389 за шаг
  (FP/solution_profile.json). База: 35.8 мс/шаг.
- На уровне ATen (OP): `bitsandbytes::dequantize_4bit` 18.1 мс/шаг (193 вызова) и `bitsandbytes::gemm_4bit` 22.7 мс/шаг (196 вызовов),
  сумма 40.8; состав self-времени `gemm_4bit` не знаем (код `backends/cuda/ops.py:934` не читали).
- 196 bnb `Linear4bit` за шаг: q, k, v, o + gate, up, down x 28 слоёв; в backward каждая считает `dequantize_4bit` заново для `grad_A`.

## Цель по профилю
- Свой кернел не медленнее bnb: **около 39.8 мс/шаг своего времени** на тех же 389 запусках. Только он закрывает проверку 4
  (нужно 34.8 мс/шаг, FINDINGS §2): 39.8 / 34.8 = 114%, запас малый (профиль снят с накладными профайлера).
- Если кернел быстрее bnb, его доля падает вместе со временем шага: быстрее вдвое (около 20 мс/шаг) уже не хватит одному, нужен 4-й кернел.
- Не делай кернел медленнее ради доли: за это жюри снижает баллы за скорость и ревью увидит.

## Точки патча (bitsandbytes 0.50.2)
- `bitsandbytes.functional.dequantize_4bit(A, quant_state, absmax, out, blocksize, quant_type)` и операция
  `torch.ops.bitsandbytes.dequantize_4bit` (cuda: `backends/cuda/ops.py:423`, `_dequantize_4bit_impl` вызывает `lib.cdequantize_blockwise_*_nf4`).
- `bitsandbytes.autograd._functions.MatMul4Bit`: forward идёт в `torch.ops.bitsandbytes.gemm_4bit.default`, backward зовёт
  `F.dequantize_4bit(B, ctx.state)` и `torch.matmul(grad_output, W)`. Для forward проверить, какой путь реально берёт обучение
  (cuda `gemm_4bit` или `default` = dequantize + `F.linear`).
- `Linear4bit.forward` (`bitsandbytes/nn/modules.py`) приводит x к `compute_dtype` и зовёт `bnb.matmul_4bit`.

## Первый шаг (день 1)
1. Прочитать `backends/cuda/ops.py` (`gemm_4bit`, `_dequantize_4bit_impl`) и `backends/default/ops.py` (`_gemm_4bit_default_impl`): как считается
   двойное квантование: `absmax = code[absmax_8bit] * state2.absmax[i // 256] + offset`.
2. Кернел: распаковать два nibble на байт, таблица NF4, умножить на scale, записать fp16 `[out, in]`.
3. `test_nf4.py`: сравнить с `bnb.functional.dequantize_4bit` на реальных `quantize_4bit(..., compress_statistics=True)` (формы 1536x1536, 8960x1536, 1000x1000),
   ожидаем побитовое равенство. Затем `patch()`, `ENABLED = True`, `quick`.

## Риски
- Двойная квантизация и `quant_state` (nested absmax, `state2`, `offset`): ошибка даёт тихо неверные веса, проверки 5-7 это увидят.
- Быть не медленнее bnb на форме 6881280 / 215040 (так выглядят q/k/v/o и MLP): мерить `own_kernel_ms` в `quick`.
- bnb может звать dequantize из нескольких мест (`gemm_4bit`, backward): патч на уровне `torch.ops` или `F.dequantize_4bit`, проверить оба.
- Без `@triton.autotune`: подобрать BLOCK вручную на T4.

## Критерий готовности
`test_nf4` зелёный на T4; `quick`: nf4-кернел запускается около 389 раз за шаг, `own_kernel_ms` не выше 39.8 мс/шаг (не медленнее bnb)
и вместе с head_loss не меньше 34.8 мс/шаг (иначе 5% закроют другие кернелы); wall не выше 77.4 с, пик не выше 8385 MiB, eval loss не выше 2.0599.
