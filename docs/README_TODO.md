# README / submission follow-up — задачи перед сдачей

README.md and SUBMISSION.md contain English and Russian versions. The user asked to leave all unavailable submission details as TODO. / README.md и SUBMISSION.md подготовлены на двух языках; пользователь попросил отметить отсутствующие сведения как TODO.

## Supplied values / Полученные значения

- Notebook / ноутбук: [decentrathon-soup-kernel-final](https://www.kaggle.com/code/iencis/decentrathon-soup-kernel-final). URL supplied by the user; no saved-version ID is included. / Ссылка получена от пользователя; ID сохранённой версии не указан.
- Final SHA / финальный SHA: [632851f0f2c3d202ae898d07b9b7b050eed61699](https://github.com/alikhan-kai/decentrathon/commit/632851f0f2c3d202ae898d07b9b7b050eed61699).
- Final-source run F / замер финального кода F: [result](measurements/submission_632851f/result.json), [checks](measurements/submission_632851f/checks.json), [notebook log](measurements/submission_632851f/decentrathon-soup-kernel-final.log). All automatic checks passed; the full solution suite still has five failures. / Все автоматические проверки прошли; в полном наборе решения остаются пять ошибок.

## Missing values / Недостающие значения

- TODO(TEAM_NAME): official competition team name / официальное название команды.
- TODO(TEAM_ROSTER_AND_CAPTAIN_CONTACT): names or handles, GitHub logins, roles, Kaggle accounts and captain's publishable contact / имена или ники, GitHub, роли, Kaggle и разрешённый к публикации контакт капитана.
- TODO(DEMO_VIDEO_URL): public demo video link, within the organizer's duration limit / публичная ссылка на видео в пределах длительности по условиям.
- TODO(PRESENTATION_URL): public presentation link / публичная ссылка на презентацию.

## Publication and verification work / Публикация и проверки

- Publish the existing P/R raw result/check JSON and logs through the saved notebook output or a separately authorized evidence commit. Local files under docs/measurements/full_profile_final_e0eed88/runs/ and docs/measurements/robust_final_952f20a/runs_robust/ are ignored by Git. Only their stage status files are tracked there. / Опубликовать исходные результаты и логи: сейчас указанные каталоги игнорируются Git, в этих двух каталогах замеров отслеживаются только status.txt.
- The failed full-suite output is now tracked in the F notebook log. Preserve it with the submission evidence. / Вывод полного набора с ошибками теперь отслеживается в логе F; сохранить его с доказательствами сдачи.
- The configured notebook is present at submission/kaggle_notebook.ipynb, with the supplied SHA and --repeats 2. Verify the saved Kaggle version matches this file. / Настроенный ноутбук уже есть в submission/kaggle_notebook.ipynb с полученным SHA и двумя повторами; сверить сохранённую версию Kaggle с этим файлом.
- Verify public visibility and provide a saved-version URL. The web tool could not open the supplied Kaggle URL; submission/kernel-metadata.json still has is_private: true. Do not infer live visibility from that local file alone. / Проверить публичность и получить ссылку на сохранённую версию. Веб-инструмент не открыл Kaggle, локальные метаданные содержат is_private: true; это само по себе не доказывает текущую видимость страницы.
- The official checks at the supplied final SHA are recorded as F and passed. Rerun if the submitted runtime code changes. / Проверки полученного финального SHA записаны в F и прошли; повтор нужен при изменении сдаваемого исполняемого кода.
- Resolve the PEFT/torchao test-environment incompatibility and rerun the complete solution suite. Do not mark the recorded 5-failure suite as green. / Устранить несовместимость тестового окружения и повторить весь набор solution; старые пять ошибок не объявлять исправленными.
- Replace matching TODO identifiers consistently in both languages and both root files. Update checklist items only after the corresponding work is verified. / Заменить одинаковые TODO в обеих языковых версиях и обоих файлах; отмечать чек-лист только после проверки.

This documentation task only edits README.md, SUBMISSION.md and docs/README_TODO.md. Existing F evidence and the configured notebook were supplied through other commits. No push or merge is performed here. / Эта задача меняет только три документа; доказательства F и настроенный ноутбук уже добавлены другими коммитами. Здесь push и merge не выполняются.

## Explicit unknowns / Где написано «не знаю»

- README, “What changed” / «Что сделано»: isolated final-candidate gain for each module is unknown; needs a one-module on/off ABBA experiment in one T4 session.
- README, “Reproduction” / «Как запустить»: total full-profile notebook duration is unknown; needs timing from clean setup through tests and evaluation.
- README, “Next steps” / «Что дальше», NF4 row: future speed gain is unknown; needs a correct candidate and full ABBA including compilation.
- README, “Next steps” / «Что дальше», attention row: time saved by removing layer-0 fallbacks is unknown; needs a per-layer profile and gradient validation.
- README, “Next steps” / «Что дальше», MLP row: benefit of changing the fixed recomputation threshold is unknown; needs fixed-policy comparisons on both datasets.

The LoRA row cites an older profile only and does not predict a current gain. SUBMISSION.md summarizes the module-gain and notebook-duration unknowns. / Строка LoRA ссылается только на старый профиль и не предсказывает текущий выигрыш; SUBMISSION.md кратко повторяет неизвестность вкладов модулей и длительности ноутбука.
