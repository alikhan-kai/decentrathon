# Командная работа: Soup Kernel Challenge

Репозиторий команды: https://github.com/alikhan-kai/decentrathon (сдаётся жюри, публичный). Дедлайн: 31 октября 23:59 (Астана, UTC+5).
Цель: ускорить `soup train` на Kaggle T4 или снизить пик памяти своими Triton-кернелами из `solution/`, не меняя задачу.
Прочитай по порядку: `docs/RULES.md` (правила), `docs/FINDINGS.md` (что мерили), `docs/INTERFACES.md` (контракты),
`docs/tasks/<твоя>.md` (твоя задача).

## Где мы (источники в `docs/FINDINGS.md`)
base 84.7 с и 14345 MiB, solution 73.38 с и 8385 MiB (speedup 1.154, memory 1.711), eval loss без потерь. Проверки 1-3 и 5-10 проходят.
**Проверка 4 падает**: наши кернелы 0.46% времени, нужно 5% (34.8 мс/шаг, есть 3.2). Главный путь: nf4 (около 39.8 мс/шаг потолок),
плюс rmsnorm (62.1 мс/шаг сегодня), lora (9.7), запасной rotary (32.0).

## Старт (у каждого)
```bash
git clone https://github.com/alikhan-kai/decentrathon.git
cd decentrathon
git checkout feat/<своя-ветка>      # feat/head-loss, feat/rmsnorm, feat/nf4 или feat/lora
```
Работаем только в клоне `decentrathon`. На Mac без CUDA: писать код и запускать `tools/check_local.sh`. Всё с GPU идёт в Kaggle на T4.

## Роли
| # | Участник | Ветка | Файл | Задача |
|---|---|---|---|---|
| 1 | integrator | `feat/head-loss` | `head_loss.py`, `kaggle/`, `notes/`, `tools/`, `__init__.py` | `docs/tasks/integrator.md` |
| 2 | rmsnorm | `feat/rmsnorm` | `rmsnorm.py`, `test_rmsnorm.py` | `docs/tasks/rmsnorm.md` |
| 3 | nf4 | `feat/nf4` | `nf4.py`, `test_nf4.py` | `docs/tasks/nf4.md` |
| 4 | lora | `feat/lora` | `lora.py`, `test_lora.py`, README, видео, презентация | `docs/tasks/lora.md` |
Правим только свой файл: тогда нет конфликтов слияния. `MODULES` в `solution/__init__.py` меняет только integrator.

## Kaggle-аккаунт, токен, запуск (на СВОЁМ аккаунте)
1. Подтверди телефон в Kaggle (без этого нет GPU и интернета). Токен: kaggle.com, Settings, API, Create New Token.
   Токен никому не отправлять и не коммитить; положить в `~/.kaggle/access_token` (`chmod 600`) или в `KAGGLE_API_TOKEN`.
2. `python3 -m venv .venv-kaggle && source .venv-kaggle/bin/activate && pip install kaggle`
3. Закоммить изменения (дерево должно быть чистым) и запусти, указав свой логин Kaggle:
   ```bash
   export KAGGLE_USERNAME=<твой логин>
   tools/kaggle_run.sh unit feat/nf4 solution/test_nf4.py   # только юнит-тест, 1-2 минуты
   tools/kaggle_run.sh quick                                # pytest + прогон решения, около 5-8 минут
   tools/kaggle_run.sh check_and_ops                        # quick + профиль по модулям
   ```
   Скрипт пушит текущую ветку в origin (в `main` напрямую не пушит), собирает ноутбук под коммит этой ветки, запускает на твоём
   аккаунте на T4, ждёт (до 50 минут), скачивает в `kaggle/out_<MODE>_<время>/` и печатает таблицы (`tools/summarize.py`).
   Квота около 30 GPU-часов в неделю на аккаунт: полный `full_profile` (около 15 минут) только у integrator.
4. Запиши строку в `notes/experiments.md`.

## Правила работы
- **В `main` не пушить напрямую.** Только Pull Request из своей ветки (шаблон PR содержит чек-лист).
- Перед пушем: `tools/check_local.sh` (синтаксис, `tools/lint_rules.py`, `harness/tests/task` не изменены). Тот же скрипт гоняет GitHub Actions (`lint`).
- Нет `@triton.autotune`, CUDA graphs, `os.environ`, холостых кернелов; не менять `harness/`, `tests/`, `task/`, задачу и LoRA с dropout.
- Один кернел за раз. `ENABLED = True` только после зелёного юнит-теста и `quick` на T4.

## Как вливать
1. PR в `main` из `feat/<ветка>`. В описании: ссылка на юнит-тест на T4 (ноутбук или лог) и результаты `quick`: wall, пик MiB,
   `own_kernel_ms` модуля, доля.
2. Зелёный `lint`, зелёный unit на T4, зелёный quick. Мерджит integrator; ветку держи актуальной: `git fetch origin && git rebase origin/main`
   (без `--force` на общих ветках).

## График (Астана, UTC+5)
- Сейчас: кернелы в своих ветках, PR по одному. Приоритет: nf4, потом rmsnorm, lora.
- **Заморозка кода за 8-10 часов до дедлайна**: 31 октября около 14:00-16:00. После неё в `main` только правки, чинящие проверки.
- Финальный `full_profile` (или `full`) на коммите, который сдаётся; его хеш в README, `SUBMISSION.md` и в публичном ноутбуке Kaggle (`COMMIT`).
- Дедлайн: 31 октября 23:59.

## Чек-лист сдачи
- [ ] все проверки 1-10 PASS на финальном коммите (`checks.json`)
- [ ] README переписан: что сделано, как запустить, результаты, ограничения, ссылка на публичный ноутбук Kaggle
- [ ] `SUBMISSION.md` (из `templates/SUBMISSION.md`) в корне: ссылки на коммит, публичный ноутбук, видео, презентацию
- [ ] публичный ноутбук Kaggle: сохранённая версия с `COMMIT` = сдаваемый коммит; ссылка и в README, и в `SUBMISSION.md`
- [ ] видео до 3 минут
- [ ] презентация
- [ ] репозиторий `decentrathon` публичный
- [ ] в `solution/` нет запрещённых библиотек, autotune, CUDA graphs, `os.environ`
