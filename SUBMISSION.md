# Submission / Сдача: TODO(TEAM_NAME)

## English

### Links

- Repository: [alikhan-kai/decentrathon](https://github.com/alikhan-kai/decentrathon).
- Final submission commit: [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).
- Submission Kaggle notebook, configured with the submission SHA: [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final).
- Demo video, up to 3 minutes: TODO(DEMO_VIDEO_URL).
- Presentation: TODO(PRESENTATION_URL).

The notebook URL and final SHA match [README](README.md). F is the recorded run of that SHA; its [result](docs/measurements/submission_632851f/result.json), [checks](docs/measurements/submission_632851f/checks.json) and [notebook log](docs/measurements/submission_632851f/decentrathon-soup-kernel-final.log) are tracked. Public notebook visibility and a version-specific URL remain unverified; local [metadata](submission/kernel-metadata.json) says `is_private: true`.

### Description

Custom Triton kernels accelerate NF4 unpacking, RMSNorm, rotary position encoding and chunked head loss/metrics. The solution also removes padding inside existing batches, preserves independent attention per example, reduces MLP saved activations and startup overhead. It preserves the fixed Soup/Qwen LoRA task and saves a standard PEFT adapter.

### Team

| Required team information | Status |
|---|---|
| Names or handles, roles, GitHub and Kaggle accounts, captain contact | TODO(TEAM_ROSTER_AND_CAPTAIN_CONTACT) |

The organizer allows up to 4 participants. Git contribution credits appear in [README](README.md); they are not a confirmed competition roster. No private contact information is inferred from commit metadata.

### Our kernels

| Kernel (`@triton.jit` name) | File | What it replaces | Share of timed training in F, two repeats |
|---|---|---|---:|
| `head_loss_kernel` | [head_loss.py](solution/head_loss.py) | Separate vocabulary loss, entropy, accuracy and gradient elementwise work | 0.5981–0.6015% |
| `nf4_dequant_kernel` | [nf4.py](solution/nf4.py) | bitsandbytes NF4 unpacking | 7.1514–7.2678% |
| `rmsnorm_fwd_kernel` | [rmsnorm.py](solution/rmsnorm.py) | Qwen2 RMSNorm forward operations | 0.6593–0.6723% |
| `rmsnorm_bwd_kernel` | [rmsnorm.py](solution/rmsnorm.py) | RMSNorm input-gradient operations | 0.6985–0.7076% |
| `ember_rope_joint` | [rotary.py](solution/rotary.py) | Q/K rotary forward/backward elementwise chain | 0.9202–0.9463% |

Per-kernel percentages are derived separately for each F solution run as `100 × own_kernel_ms / (1000 × timed_loop_s)`, then shown as ranges. Range endpoints need not come from the same repeat. The combined eligible share is **10.0% in the lowest repeat**, as reported by check 4. These are runtime shares, not speedup contributions. Source: F `runs.solution[].metrics` and checks, defined in [README results](README.md#results).

The starter `example_swiglu_*` kernels are **not our own** and are excluded from eligible time. `_rope_rotate_block` is a helper, not a separately launched kernel. Residual/norm kernels are disabled in the measured candidate.

Ideas and reference code:

- Head chunking: Makazhan Alpamys and Soup contributors, [stream_loss_chunk](https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/utils/stream_loss_chunk.py). Our head Triton kernel is original.
- NF4 format: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer, [QLoRA](https://arxiv.org/abs/2305.14314); [bitsandbytes reference](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py).
- RMSNorm/RoPE semantics: Hugging Face Transformers contributors, [Qwen2](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py).
- SwiGLU: organizer [starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge); no original-kernel credit claimed.
- Packed decoder and native/hybrid saves: team work documented in [candidate provenance](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). Startup changes: Alikhan-kai, [merge record](docs/MERGE_OVERHEAD.md).

### Results on Kaggle T4

Each dataset was measured in one **base → solution → solution → base** invocation. F is the submission-source run at `632851f`; R is the earlier additional-row/seed-7 run at `952f20a`, with identical `solution/`. Time is the per-side median; memory is the maximum peak. Sources and publication limits are in [README](README.md#results) and [experiment notes](notes/experiments.md).

| F: public rows | Base Soup | Ours | Comparison |
|---|---:|---:|---|
| Whole `soup train`, s | 83.25 | 59.66 | Speedup 1.3954× |
| Peak GPU memory, MiB | 14345 | 4421 | Base/ours 3.2447× |
| Eval loss | 2.04966 | 2.04942 | −0.012% vs base |

| R: other rows, seed 7 | Base Soup | Ours | Comparison |
|---|---:|---:|---|
| Whole `soup train`, s | 81.68 | 61.91 | Speedup 1.3193× |
| Peak GPU memory, MiB | 14351 | 4405 | Base/ours 3.2579× |
| Eval loss | 2.11513 | 2.11550 | +0.017% vs base |

- [x] All 10 automatic checks passed for recorded F and R.
- [x] README rewritten with implementation, reproduction, results and limitations in English and Russian.
- [x] Final submission SHA and notebook URL supplied; automatic checks at that SHA recorded in F.
- [ ] Public notebook visibility, saved-version URL and downloadable outputs verified.
- [ ] Team details, video and presentation supplied.

F result/check JSON and training logs are tracked in `docs/measurements/submission_632851f/`. Historical P/R raw evidence remains locally ignored. Downloading outputs through the saved Kaggle notebook remains unverified. The 79 official Soup regressions passed; the separate F solution suite had 298 passed, 126 skipped and 5 PEFT/torchao environment failures. See [README](README.md#correctness-validation).

### Presentation covers

The requested presentation should explain the fixed task; memory pressure from logits and saved activations; the head, NF4, RMSNorm and RoPE kernels; padding/MLP integration; paired results and quality checks; attribution, limitations and next experiments. The video must fit the organizer's 3-minute limit. No completed media is claimed until its links are supplied.

### Limitations

The candidate targets the recorded T4 FP16/library combination. MLP optimization is restricted by source hashes and first-order eager guards. Attention falls back for 256/7168 calls per F run (3.5714%, call share). R contains a 7.04% row-loss gap on a 7-answer-token row while its mean gaps, 0.225%/0.221%, pass check 5. Packing changes dropout RNG assignment. Cross-session times do not isolate gains. LoRA epilogue work is unfinished; residual fusion is disabled. Individual final-module gains and full notebook duration are unknown. Full evidence and proposed measurements are in [README](README.md#limitations-and-unsuccessful-work).

---

## Русский

### Ссылки

- Репозиторий: [alikhan-kai/decentrathon](https://github.com/alikhan-kai/decentrathon).
- Финальный коммит сдачи: [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).
- Kaggle-ноутбук сдачи, настроенный на финальный SHA: [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final).
- Видео до 3 минут: TODO(DEMO_VIDEO_URL).
- Презентация: TODO(PRESENTATION_URL).

URL ноутбука и финальный SHA совпадают с [README](README.md). F — записанный запуск этого SHA; его [результат](docs/measurements/submission_632851f/result.json), [проверки](docs/measurements/submission_632851f/checks.json) и [лог ноутбука](docs/measurements/submission_632851f/decentrathon-soup-kernel-final.log) отслеживаются Git. Публичность и ссылка на конкретную сохранённую версию пока не проверены; локальные [метаданные](submission/kernel-metadata.json) содержат `is_private: true`.

### Описание

Свои Triton-кернелы ускоряют распаковку NF4, RMSNorm, RoPE и порционное вычисление ошибки и метрик головы. Дополнительно удаляется padding внутри исходных батчей, сохраняется независимое внимание каждого примера, уменьшается память активаций MLP и время подготовки. Фиксированная задача Soup/Qwen LoRA сохраняется; на выходе получается обычный PEFT-адаптер.

### Команда

| Сведения о команде | Статус |
|---|---|
| Имена/ники, роли, GitHub и Kaggle, контакт капитана | TODO(TEAM_ROSTER_AND_CAPTAIN_CONTACT) |

Организатор разрешает до 4 участников. Авторство изменений указано в [README](README.md), но не заменяет подтверждённый состав. Личные контакты из метаданных коммитов не подставляются.

### Наши кернелы

| Имя `@triton.jit` | Файл | Что заменяет | Доля измеряемого времени F, два повтора |
|---|---|---|---:|
| `head_loss_kernel` | [head_loss.py](solution/head_loss.py) | Отдельные вычисления loss, entropy, accuracy и градиентов по словарю | 0.5981–0.6015% |
| `nf4_dequant_kernel` | [nf4.py](solution/nf4.py) | Распаковку NF4 в bitsandbytes | 7.1514–7.2678% |
| `rmsnorm_fwd_kernel` | [rmsnorm.py](solution/rmsnorm.py) | Операции прямого прохода Qwen2 RMSNorm | 0.6593–0.6723% |
| `rmsnorm_bwd_kernel` | [rmsnorm.py](solution/rmsnorm.py) | Вычисление градиента входа RMSNorm | 0.6985–0.7076% |
| `ember_rope_joint` | [rotary.py](solution/rotary.py) | Поэлементную цепочку RoPE для Q/K в обоих направлениях | 0.9202–0.9463% |

Для каждого повтора доля считается как `100 × own_kernel_ms / (1000 × timed_loop_s)` из F. В таблице — диапазон между повторами, поэтому границы разных строк могут относиться к разным прогонам. Общая зачётная доля в худшем повторе — **10.0%**, как в проверке 4. Это доля времени, а не вклад в ускорение. Источник: `runs.solution[].metrics` и проверки F, описанные в [README](README.md#результаты).

Кернелы `example_swiglu_*` принадлежат стартовому примеру и **не считаются нашими**. `_rope_rotate_block` — внутренняя функция, не отдельный запуск. Residual/norm-кернелы в измеренном кандидате выключены.

Идеи и эталонный код:

- Порционная голова: Makazhan Alpamys и участники Soup, [stream_loss_chunk](https://github.com/MakazhanAlpamys/Soup/blob/main/src/soup_cli/utils/stream_loss_chunk.py). Triton-кернел головы написан командой.
- NF4: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer, [QLoRA](https://arxiv.org/abs/2305.14314); [эталон bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py).
- Семантика RMSNorm/RoPE: участники Hugging Face Transformers, [Qwen2](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py).
- SwiGLU: [стартовый набор организатора](https://github.com/MakazhanAlpamys/soup-kernel-challenge), без заявления собственного авторства кернелов.
- Packed decoder и native/hybrid saves: [история кандидата](https://github.com/alikhan-kai/decentrathon/blob/62cd831c582298d53345b5d80feba3ea9796914a/docs/overnight/CANDIDATE.md). Подготовка запуска: Alikhan-kai, [запись интеграции](docs/MERGE_OVERHEAD.md).

### Результаты на Kaggle T4

Для каждой выборки выполнен один запуск **база → решение → решение → база**. F — замер сдаваемого кода на `632851f`; R — ранняя проверка других строк и seed 7 на `952f20a`, с той же папкой `solution/`. Время — медиана стороны, память — максимальный пик. Источники и ограничения публикации: [README](README.md#результаты), [журнал](notes/experiments.md).

| F: публичные строки | База Soup | Решение | Сравнение |
|---|---:|---:|---|
| Вся команда `soup train`, с | 83.25 | 59.66 | Ускорение 1.3954× |
| Пик памяти GPU, MiB | 14345 | 4421 | База/решение 3.2447× |
| Eval loss | 2.04966 | 2.04942 | −0.012% |

| R: другие строки, seed 7 | База Soup | Решение | Сравнение |
|---|---:|---:|---|
| Вся команда `soup train`, с | 81.68 | 61.91 | Ускорение 1.3193× |
| Пик памяти GPU, MiB | 14351 | 4405 | База/решение 3.2579× |
| Eval loss | 2.11513 | 2.11550 | +0.017% |

- [x] Все 10 автоматических проверок прошли для сохранённых F и R.
- [x] README описывает реализацию, запуск, результаты и ограничения на английском и русском.
- [x] Получены финальный SHA и URL ноутбука; автоматические проверки этого SHA записаны в F.
- [ ] Проверены публичность, ссылка на сохранённую версию и скачивание результатов ноутбука.
- [ ] Получены состав команды, видео и презентация.

JSON результата/проверок и логи F отслеживаются в `docs/measurements/submission_632851f/`. Исторические исходные данные P/R по-прежнему локально игнорируются. Скачивание результатов через сохранённую версию Kaggle пока не проверено. Отдельные 79 регрессионных тестов Soup прошли; в наборе решения F — 298 passed, 126 skipped и 5 ошибок окружения PEFT/torchao. См. [проверку корректности](README.md#как-проверяли-корректность).

### Содержание презентации

В презентации следует показать фиксированную задачу; расход памяти на logits и активации; кернелы головы, NF4, RMSNorm, RoPE; интеграцию padding/MLP; парные результаты и проверку качества; авторство, ограничения и следующие эксперименты. Видео должно укладываться в ограничение организатора — 3 минуты. Готовность медиаматериалов не заявляется без ссылок.

### Ограничения

Кандидат рассчитан на указанное окружение T4 FP16. MLP ограничен хэшами исходника и обучением первого порядка в eager-режиме. Внимание использует fallback для 256/7168 вызовов F (3.5714% вызовов). В R есть отклонение row loss 7.04% на строке с 7 токенами ответа при проходящих средних 0.225%/0.221%. Packing меняет назначение dropout RNG. Межсессионные замеры не выделяют вклад оптимизаций. LoRA-эпилог не завершён, residual fusion выключена. Отдельные финальные выигрыши модулей и полная длительность ноутбука неизвестны. Подробности и нужные замеры — в [README](README.md#ограничения-и-что-не-получилось).
