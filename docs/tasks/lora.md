# Задача lora: слияние масштаба, приведений типов и сложения LoRA на q_proj и v_proj

Файл `solution/lora.py`, тест `solution/test_lora.py` (шаблон), ветка `feat/lora`. Dropout (0.05) НЕ менять, только слияние операций.

## Что в профиле (OP, решение, мс/шаг)
- peft `Linear4bit` исключая вложенные модули (эпилог LoRA: `result.clone()`, приведение x, `* scaling`, `.to(dtype)`, `result + output`): **9.7 мс/шаг**
  (fwd 6.9, bwd 2.8), 56 вызовов за шаг.
- Остальное LoRA-пути, которое сливать нельзя или не нужно: lora_A и lora_B (`nn.Linear`) 17.3 мс/шаг (matmul), `Dropout` 12.5 мс/шаг (должен остаться тем же).
- Типы: результат базового слоя fp32, lora_A/B вес fp32 (обучаемый), выход lora fp16, итог fp32, `scaling = 2.0`.

## Цель по профилю
- Потолок: 9.7 мс/шаг. Свой кернел займёт меньше (читает result и lora_out, пишет сумму): реальное время **не знаю**; измерить микробенчмарком
  CUDA events на `[4, 512, 1536]`, затем `quick`. Оценка по трафику (не из логов): единицы мс/шаг.
- Вклад в проверку 4 малый; главная польза в скорости шага (минус clone и приведения).

## Точки патча (peft 0.20.0)
- `peft.tuners.lora.bnb.Linear4bit.forward` (`peft/tuners/lora/bnb.py`, класс внутри `if is_bnb_4bit_available():`; патчить после `import peft.tuners.lora.bnb`).
  Путь без адаптеров/слияния: `result = self.base_layer(x); result = result.clone()`, затем для активного адаптера
  `x = self._cast_input_dtype(x, lora_A.weight.dtype)`, `output = lora_B(lora_A(dropout(x))) * scaling`, `output = output.to(expected_dtype)`, `result = result + output`.
- `dispatch_bnb_4bit` создаёт этот слой только для `bnb.nn.Linear4bit` в целях `q_proj`, `v_proj` (README: `target_modules: auto`).

## Первый шаг (день 1)
1. Определить границу слияния: `result + (lora_out * scaling)` с приведением dtype в одном кернеле, forward и backward (градиент по lora_out = grad * scaling, по result = grad).
2. `lora_epilogue(result, lora_out, scaling)` как `autograd.Function`; `test_lora.py` зелёный на T4.
3. `patch()`: заменить только эпилог внутри `forward`, dropout и lora_A/lora_B вызывать теми же вызовами в том же порядке (тот же расход RNG).

## Риски
- Dropout и RNG: другой порядок вызовов меняет маску и loss (проверка 5-6). Не трогать.
- `.clone()` в базе нужен из-за autograd bnb: убирать только после проверки backward.
- Другие адаптеры, merged-режим, `adapter_names`: оставить оригинальный путь.

## Критерий готовности
`test_lora` зелёный на T4; `quick`: около 56 + 56 запусков кернела за шаг, `own_kernel_ms` > 0, меньше 9.7 мс/шаг; wall не выше 77.4 с, пик не выше 8385 MiB,
eval loss не выше 2.0599; `lora_dropout` и конфиг адаптера не изменены (проверка 5).

## Статус (ветка feat/check4): НЕ НАЧАТО
Проверка 4 уже проходит без lora-кернела (F2: 6.06%). Потолок задачи прежний, 9.7 мс/шаг. Если нужен запас по доле, первым кандидатом по профилю остаётся rotary (32.0 мс/шаг, см. `integrator.md`), а не LoRA.
