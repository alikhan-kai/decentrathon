# Командная работа: Soup Kernel Challenge

Репозиторий команды: https://github.com/alikhan-kai/decentrathon (он сдаётся жюри, должен быть публичным).
Дедлайн: 31 октября 23:59 (Астана, UTC+5). Цель: ускорить `soup train` на Kaggle T4 или снизить пиковую
память своими Triton-кернелами из `solution/`, не меняя задачу.

## Где мы сейчас (источник: `kaggle/out_full_profile_20261030_151022/`, full_profile на T4, коммит fe282f8)

| | base | solution |
|---|---|---|
| wall, с (медиана 2 запусков) | 84.7 | 73.38 |
| пик памяти nvidia-smi, MiB | 14345 | 8385 |
| eval loss | 2.04966 | 2.04969 (+0.001%) |

speedup 1.154x, memory_ratio 1.711x. Проверки 1-3 и 5-10 проходят. **Падает только проверка 4**: наши кернелы
(без примера SwiGLU) занимают 0.4% времени шага, нужно >= 5%. Сейчас у нас один кернел, `head_loss_kernel`:
199 мс на 62 шага, то есть 3.2 мс на шаг, а шаг (с шага 3) длится около 705-715 мс, 5% это 35.3-35.8 мс на шаг.
Чтобы пройти проверку, нужно добавить ещё около 32 мс на шаг времени НАШИХ кернелов.

Профиль решения, мс на шаг (16 шагов, с накладными профайлера; `solution_profile/profile.json`):
matmul 310.7, elementwise/cast 285.0, attention 52.8, NF4 dequant 39.8, прочее 21.5, softmax/CE 3.0.
База: elementwise/cast 361.9, matmul 283.1, softmax/CE 55.6, attention 46.5, NF4 dequant 35.8.

## Старт (у каждого)

```bash
git clone https://github.com/alikhan-kai/decentrathon.git
cd decentrathon
git checkout feat/<своя-ветка>      # feat/head-loss, feat/rmsnorm, feat/nf4 или feat/lora
```

Работаем только в клоне `decentrathon`. На Mac без CUDA можно писать код и запускать
`python3 -m py_compile solution/*.py`; всё с GPU (pytest кернела, `harness/`) идёт только в Kaggle на T4.

## Роли

| # | Участник | Ветка | Файл | Цель |
|---|---|---|---|---|
| 1 | integrator | `feat/head-loss` | `solution/head_loss.py`, `kaggle/`, `notes/`, `tools/` | держать `head_loss` рабочим, проверки 5-7, собирать `apply()`, финальный full |
| 2 | rmsnorm | `feat/rmsnorm` | `solution/rmsnorm.py` | fused RMSNorm forward+backward, затем слияние с остатком |
| 3 | nf4 | `feat/nf4` | `solution/nf4.py` | Triton-распаковка NF4 вместо bitsandbytes, не медленнее |
| 4 | lora | `feat/lora` | `solution/lora.py` | слияние масштаба, приведений типов и сложения LoRA на q_proj, v_proj |

Куда патчить (прочитано в transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2; подробности в docstring заглушек):
- rmsnorm: `transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward`; слияние с остатком: `Qwen2DecoderLayer.forward`.
- nf4: `bitsandbytes.functional.dequantize_4bit`, `bitsandbytes.autograd._functions.MatMul4Bit` (backward зовёт
  `F.dequantize_4bit`), cuda-реализация в `bitsandbytes/backends/cuda/ops.py` (`_dequantize_4bit_impl`).
  Учесть двойное квантование (nested absmax).
- lora: `peft.tuners.lora.bnb.Linear4bit.forward`. Dropout не трогать.

### Цели и первый шаг (из профиля; оценки помечены)

Нужно суммарно около 35 мс/шаг нашего кернел-времени (сейчас 3.2).
- **nf4**: в профиле решения NF4 dequant (bnb) это 39.8 мс/шаг, 6224 запуска за 16 шагов (389 на шаг). Кернел не
  медленнее bnb даёт примерно столько же своего времени, а это уже около 5.6% и почти закрывает проверку 4.
  Первый шаг: Triton-кернел распаковки NF4 + тест против `bnb.functional.dequantize_4bit` на реальном `quant_state`.
  Профиль замерен с накладными профайлера, поэтому запас мал: проверять `own_kernel_ms` в `quick`.
- **rmsnorm**: в профиле решения самые крупные elementwise-кернелы это fp32 Mul (441+198 мс на 16 шагов), fp32 Add
  (526), fp32 copy/cast (942+857+148). Какие именно операции их порождают, профиль не говорит: гипотеза, что
  это RMSNorm (fp32-апкаст, mul на rsqrt, mul на вес) и остаточные сложения; проверить. Цель: убрать часть из 285
  мс/шаг elementwise и получить свой кернел-время (оценка: единицы мс/шаг, «не знаю» точнее).
  Первый шаг: кернел forward+backward для одной нормы, тест через `solution/kcheck.py`, проверить dtype выхода.
- **lora**: те же fp32/fp16 cast и add кернелы частично относятся к LoRA-ветке q_proj и v_proj (896 запусков
  fused_dropout, 80 мс на 16 шагов, остаются как есть). Доля LoRA в elementwise из профиля не видна: «не знаю».
  Первый шаг: профиль конкретного `Linear4bit.forward` (torch.profiler на одном слое в Kaggle), затем кернел.
- **integrator**: `head_loss_kernel` 3.2 мс/шаг. Первый шаг: после слияния веток запустить `full_profile` и
  сверить проверки 5-7 (entropy, accuracy, row loss в пределах 3%); следить, чтобы пик памяти не вырос
  (сейчас 8385 MiB, база 14345).

## Как запускать (на СВОЁМ аккаунте Kaggle)

1. Каждый создаёт свой токен: kaggle.com, Settings, API, Create New Token. Нужен подтверждённый телефон,
   иначе нет GPU и интернета. Токен никому не отправлять и не коммитить. Положить в `~/.kaggle/access_token`
   (`chmod 600`) или в переменную `KAGGLE_API_TOKEN`.
2. `python3 -m venv .venv-kaggle && source .venv-kaggle/bin/activate && pip install kaggle`
3. Закоммитить изменения (скрипт требует чистое дерево) и:
   ```bash
   tools/kaggle_run.sh quick                  # pytest + один прогон решения, 5-10 минут
   tools/kaggle_run.sh full_profile feat/nf4  # база + решение x2 + проверки + профили, около 15 минут
   ```
   Скрипт пушит коммит в origin, строит ноутбук под этот коммит, запускает его на T4 на твоём аккаунте,
   ждёт (до 50 минут), скачивает результат в `kaggle/out_<MODE>_<время>/` и печатает таблицы
   (`tools/summarize.py <папка>`). Квота около 30 GPU-часов в неделю на аккаунт: не гоняй `full_profile` без нужды.
4. Запиши строку в `notes/experiments.md`.

## Правила

- Нет `@triton.autotune` и CUDA graphs в `solution/` (кэш компиляции пуст в каждом запуске).
- Не менять `harness/`, `tests/`, `task/`, `templates/kaggle_notebook.ipynb`. Не менять задачу (модель, данные,
  батчи, шаги, LoRA вместе с dropout, оптимизатор, режимы Soup).
- Нельзя unsloth, liger-kernel, xformers, flash-attn, torchao и подобное. Кернелы, чей результат обучение не
  использует («холостые»), запрещены.
- В `main` только после зелёного юнит-теста на T4 и зелёного `quick`. Один кернел за раз.
- У каждого модуля `ENABLED` и идемпотентный `patch()`; `apply()` пробрасывает исключения. Заглушка выключена
  (`ENABLED = False`), включай её только когда кернел готов.
- Хелпер сравнения с PyTorch: `solution/kcheck.py` (пример: `solution/test_kcheck.py`). Тесты без CUDA
  пропускаются (`pytest.mark.skipif`).

## Как вливать

Пиши в свою ветку, открывай Pull Request в `main`, в описании: что сделано, вывод `quick` (pytest solution/tests,
`own_kernel_ms`, пик MiB). Мерджит integrator после проверки. Ветку обновляй из `main` (`git fetch origin &&
git rebase origin/main`), `git push --force` не использовать.

## График (Астана, UTC+5)

- Сейчас до заморозки: кернелы в своих ветках, PR в `main` по одному.
- Заморозка кода: за 8-10 часов до дедлайна, то есть 31 октября около 14:00-16:00. После неё в `main` только
  правки, которые чинят проверки.
- Финальный `full_profile` (или `full`) на коммите, который сдаётся; этот коммит записать в README, SUBMISSION.md
  и в публичный ноутбук Kaggle (`COMMIT`).
- Дедлайн: 31 октября 23:59.

## Чек-лист перед сдачей

- [ ] все проверки 1-10 проходят на финальном коммите (`kaggle/out_full*/…/checks.json`)
- [ ] README переписан: что сделано, как запустить, результаты, ограничения, ссылка на публичный ноутбук Kaggle
- [ ] `SUBMISSION.md` (из `templates/SUBMISSION.md`) в корне: коммит, ссылка на ноутбук, видео, презентация
- [ ] публичный ноутбук Kaggle: сохранённая версия с `COMMIT` = сдаваемый коммит; ссылка и в README, и в SUBMISSION.md
- [ ] видео до 3 минут
- [ ] презентация
- [ ] репозиторий `decentrathon` публичный
- [ ] в `solution/` нет запрещённых библиотек, autotune и CUDA graphs
