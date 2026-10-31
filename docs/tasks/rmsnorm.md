# Задача rmsnorm: fused RMSNorm (forward + backward), затем слияние с остатком

Файл `solution/rmsnorm.py`, тест `solution/test_rmsnorm.py` (шаблон, пропускается без кернела), ветка `feat/rmsnorm`.

## Что в профиле (OP/op_profile.txt, решение, мс/шаг)
- `Qwen2RMSNorm`: **62.1 мс/шаг** (8.4% GPU-времени): forward 16.0, backward 46.1, 57 вызовов за шаг (28 x 2 + финальный).
- Выход fp32, вес fp32 с `requires_grad=False`, остаточный поток fp32.
- Остаточные сложения вокруг норм (`DecoderLayer` исключая вложенные): 7.5 мс/шаг (шаг 2 этой задачи).
- Это основная часть fp32 `mul`/`copy_`/`pow`/`mean`/`rsqrt` из elementwise-группы (285.0 мс/шаг, FP).

## Цель по профилю
- Потолок: 62.1 мс/шаг, это то, что сегодня тратят нормы на многих мелких операциях. Свой кернел займёт заметно меньше.
- Реальное время своего кернела: **не знаю**. Оценка по пропускной способности (не из логов): вход `[4, 512, 1536]` fp32 = 12.6 MB,
  forward читает и пишет ~25 MB, backward ~38 MB; при ~320 GB/s T4 (по спецификации) это порядка 11 мс/шаг на 57 вызовов.
  Как измерить: микробенчмарк CUDA events на `[4, 512, 1536]` и `[4, 512, 1001]`, затем `quick` (`own_kernel_ms`).
- Вклад в проверку 4: ожидаем около 10 мс/шаг (оценка), вместе с head_loss этого мало: нужен nf4 (см. `docs/tasks/nf4.md`).
- Главная выгода здесь в скорости шага: убираем до ~50 мс/шаг лишних операций (оценка: 62.1 минус время своего кернела).

## Точки патча (transformers 5.16.1)
- `transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward`: база: `x.to(float32)`, `var = x.pow(2).mean(-1, keepdim=True)`,
  `x * torch.rsqrt(var + self.variance_epsilon)`, `return self.weight * x.to(input_dtype)`. Класс декорирован
  `@use_kernel_forward_from_hub("RMSNorm")`: проверить на T4, что подмена `forward` у класса работает и `kernelize` его не заменяет.
- Шаг 2: `Qwen2DecoderLayer.forward` (`residual + hidden`, затем `post_attention_layernorm`) и финальная `self.norm` в `Qwen2Model.forward`.

## Первый шаг (день 1)
1. `rmsnorm(x, weight, eps)` как `torch.autograd.Function`: forward один кернел (строка на программу, fp32-аккумулятор, сохранить rstd),
   backward один кернел (dx; вес заморожен, `weight.requires_grad` проверить на T4).
2. Вернуть dtype выхода как у базы (fp32), входы fp32: проверить на T4 печатью dtype в юнит-тесте.
3. `test_rmsnorm.py` зелёный на T4 (`tools/kaggle_run.sh unit <ветка> solution/test_rmsnorm.py`), потом `patch()` и `ENABLED = True`, `quick`.

## Риски
- dtype выхода и остаточный поток: если вернуть fp16 вместо fp32, сдвинется loss (проверки 5-6).
- Нечётный hidden (тест с 1001) и маски блоков.
- Подмена `forward` не должна ломать `apply()` перед тестами Soup (guard на `is_cuda`).
- Шаг 2 (слияние с остатком) трогает `DecoderLayer.forward`: согласовать с integrator.

## Критерий готовности
`test_rmsnorm` зелёный на T4; `quick`: `triton_launches` rmsnorm-кернелов около 57 + 57 за шаг, `own_kernel_ms` > 0 и меньше 62.1 мс/шаг
(цель около 10 мс/шаг, оценка, уточнить по первому замеру); wall не выше 77.4 с (максимум замеров решения, FINDINGS §1), пик не выше 8385 MiB,
eval loss не выше 2.0599 (0.5% над базой 2.04966).

## Статус (ветка feat/check4, коммит 522bada) : ШАГ 1 ГОТОВ
`solution/rmsnorm.py`: `rmsnorm_fwd_kernel` и `rmsnorm_bwd_kernel`, fp32 вход и вес, выход fp32 как у базы; вес заморожен, поэтому dw не считается (запасной путь в PyTorch, если `requires_grad`).
`test_rmsnorm.py`: 7 passed на T4 (допуск rtol = atol = 1e-4). В обучении 3648 + 3584 запуска, 4.6-4.7 + 6.3 мс/шаг, вместе 0.8-1.4% шага; speedup шага: медианный шаг 0.689 -> 0.639 с (Q1 -> Q2).
Осталось (шаг 2 задачи, не делали): слияние с остаточным сложением (`DecoderLayer` исключая вложенные: 7.5 мс/шаг), вариант fp16-выхода норм (не пробовали, меняет числа LoRA-ветки).
