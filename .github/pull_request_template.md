## Что сделано
<!-- какой кернел / какой файл solution/, что патчится -->

## Чек-лист (без пунктов PR не вливаем)
- [ ] Юнит-тест `solution/test_<модуль>.py` зелёный на T4 (ссылка на ноутбук Kaggle или лог): 
- [ ] `tools/kaggle_run.sh quick` зелёный, результаты:
  - wall, с:
  - пик памяти, MiB:
  - own_kernel_ms этого модуля (мс, мс/шаг) и доля от timed_loop_s, %:
- [ ] Нет `@triton.autotune`, CUDA graphs, `os.environ`/`os.getenv` в `solution/`; `tools/check_local.sh` зелёный
- [ ] Eval loss, entropy и accuracy в допуске (проверки 5-7 из `harness/check_rules.py`, если запускался full)
- [ ] `ENABLED = True` только если кернел реально используется в обучении (есть в `triton_launches`)
- [ ] Менял только свой файл `solution/` и его тест; не трогал `harness/`, `tests/`, `task/`, список `MODULES`
