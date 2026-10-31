# Coordination handoff for the memory-task Codex

**GPU is released.** Our Kaggle scratch session was stopped and verified off at 2026-10-31 12:40:05 UTC. Evidence was downloaded and checksum-verified first. No GPU job remains active or queued from this thread. The user assigned the next slot to the memory-task Codex. Do not run competing jobs on the account. We could not identify that independent agent in the available T3 threads, so direct delivery is not claimed.

## Exact current measurement

Decentrathon commit `25855208695fd113fa1897385abccc73f71a19cd`, unchanged:

- Full 64-step solution repeats: 66.80 and 68.50 seconds; median **67.65 seconds**.
- Official external peak: **5327 MiB** in both repeats. PyTorch allocated peak: 4917.8 MiB; reserved peak: 5184 MiB. These are different metrics.
- All ten official checks passed, including 79 Soup regressions. Minimum eligible original-kernel share: 9.5232122800%, excluding all starter SwiGLU specializations.
- Evaluation loss: 2.04947 versus base 2.04959. Exact conditions, data hashes and selected raw results are in this directory's README and JSON files.

Raw ZIP is local at `/home/homelab/.t3/experiments/decentrathon-time-2026-10-31/evidence/latest-2585520/raw.zip`, SHA256 `2c2cbc68c285e4d6684049fcd0fcf196fa59ea82a70dbc2875a96f31f864b939`.

## Relevant source boundaries

This commit uses `ENABLE_SAVE=True`, `LAYER_BUDGET=True`, `BUDGET_ROWS=1800` and head `CHUNK=768`. The latest LoRA/NF4 commit did not change the memory policy; the preceding `662557c` raised its budget. The retained tests asserting the old 1024-row estimate fail 147/147 cases. That is a stale promise, not an observed OOM; the new 1800-row estimate is looser and feasible under its own model. Actual peak still needs the complete official run.

Qualified PR10 at `a4ea260` used a different MLP implementation and the 1024-row policy, measuring 4421 MiB in a different session. Its five MLP runtime files must be kept together if composed with newer main, with fresh validation. Do not subtract historical times across sessions.

The new LoRA forward introduces a BF16 cast before dropout despite FP32 activations and FP16 autocast. The CPU reproduction shows changed adapter gradients, including the first B update. See LORA_REVIEW.md and its reproduction. Passing the official tolerances does not prove gradient equivalence. Preserve the correct scaled loss head.

The NF4 conditional-contiguous change preserves ordinary tensor behavior: the prior `.contiguous()` already returned the same tensor for contiguous inputs.

## Ownership

This thread will stay on CPU/source work until GPU ownership is coordinated again. It will not modify the memory agent's session, working tree, or shared branches. Its current report branch is `docs/benchmark-2585520`; only this documentation directory changes. The isolated `perf/compile-overlap` checkout remains at its original qualified source.

Two unmeasured source experiments are parked locally: `/tmp/decentrathon-time-precompile-r1` (qualified all-half signature coverage and honest compilation diagnostics) and `/tmp/decentrathon-time-frozen-k-r1` (default-off first-layer attention admission). Neither was run on a GPU or enabled in a checkout. Current main already contains the older precompile implementation and frozen-K admission, so these are not independent new speedups to add blindly.
