# Merge of feat/overhead (Alikhan-kai) into integrate-overhead

Base: 0aa53cc (safe-v1). No blind merge; one commit per logical change, each revertable alone.

| Author change | Decision | Why |
|---|---|---|
| 4c51afb timing in `apply()` | take with edit | MODULES order kept (head_loss before packed), no "overhead" first. Prints to stderr on every run, nothing else changes. |
| 44d1a47 part 1: `device_map` "auto" -> `{"": 0}` | take with edit | Fails loudly if soup_cli is missing (the original returned silently). Harness sets CUDA_VISIBLE_DEVICES=0, so placement is the same. Separate flag `DEVICE_MAP`. |
| 44d1a47 part 2: in-memory dataset instead of `Dataset.map` | take with edit, behind a test | Equivalence with Soup's `_map_text_sft_rows` (columns, features, rows) is not verified locally: soup-cli 0.77.0 could not be downloaded. `test_fast_map_matches_soup` checks it where soup_cli is installed; separate flag `FAST_MAP`. If it fails, set it to False or revert the commit. |
| 1cfb49d notes/experiments.md | do not take | Numbers come from another session and are not comparable; results will be re-measured in one session (ab_overhead). |
| `test_overhead.py` | take with edit | Skips without soup_cli; extra tests for both patches. |

Open: the argument values of `resolve_device_map` are unknown; the condition is `!= "cpu"` as the author's.
