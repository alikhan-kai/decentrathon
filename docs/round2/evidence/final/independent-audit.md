Final evidence audit r2 — 2026-10-31 UTC

**Verdict: the exact candidate's recorded full admission is internally consistent: all ten checks pass, with an exact minimum eligible share of 3384997/367840 percent = 9.202362440191387559808612440…%. The 54.8-second target is not met. The separate raw official-test transcript/export remains pending.**

Read-only audit begun at 08:08:44 UTC and completed within five minutes. No repository edits, GPU use, browser, network, installs, child agents, or test/benchmark execution. This report is the only intentional write. Verification used file hashing, local Git objects, JSON/AST inspection, and the protected checker's pure checks 1–9; it did not invoke the checker main function or its pytest subprocess.

Candidate: `/home/homelab/.t3/overnight/decentrathon-2026-10-31-round2/evidence/final/raw`. Checkout: `/home/homelab/.t3/worktrees/decentrathon/round2-performance`.

**Concrete findings and limits**

1. **P2 — Raw transcript of the original 79 official regression tests is absent.** `checks.log`, `runs/checks.json`, and the retained notebook consistently record `79 passed, 14 warnings in 0.60s`; the original protected checker captures pytest output and retains only its summary. That proves a consistent recorded official pass, not preservation of the underlying transcript. The requested separately logged post-run invocation and bounded export were pending in the supplied brief and are not part of this audited ZIP. Do not claim this deliverable is complete until its separate artifact is received and bound to this source. The solution-suite transcript is present: **302 passed, 126 skipped, 42 warnings in 38.82s**.

2. **Prior P1/P2 runner findings remain prospective defects, not an observed interruption of this result.** The retained `builder.py` hashes to `2c52db5dd54e0d571a3dee280e9e25a781df1b4f3dc5300614b1992469edad19`, exactly the generator reviewed in `/tmp/decentrathon-round2-final-runner-review-r1.md`. The executed notebook still kills only the outer subprocess group on an exception; measured children start separate sessions. Its hardware query and original ZIP export also remain unbounded as described in r1. However, the notebook records all three stages exiting zero, no error output, and completion at **08:06:45.483046 UTC**, before 08:07. All four measured runs exited zero; their logs contain no timeout/interrupt/traceback markers, and no `error.txt` exists in the 159-file archive. No timeout or interrupt occurred in this recorded execution. A later helper cannot retroactively repair or be attributed to this runner.

3. **Performance target missed; no paired incremental claim against historical PR #7.** The official summary is **66.64 s / 4421 MiB** versus the same-session organizer base **83.28 s / 14345 MiB**: 1.2497× reported speedup, about 19.98% less command time and 69.18% less peak memory. Final evaluation loss is 2.04942 versus 2.04962, approximately −0.009758% (checker −0.010%). The target gap is **+11.84 s**, or 21.61% above 54.8 s. Memory is 272 MiB below the 4693-MiB target. Historical PR #7's 59.52 s belongs to another session/source admission; its numerical difference is not evidence of a paired incremental regression or gain from this provenance repair. Opponent timing is user-reported and unmatched.

**Archive, source, and hash bindings**

- `raw.zip`: **146330249 bytes**, SHA-256 `395d26c3ce5e94836ed95010a080feca2d98af8021b4b3427d3b449d6bfa92e8`, matching the requested candidate and notebook's export line. All **159** members match extracted bytes; no duplicate names, missing files, mismatches, or extra extracted files.
- All **47 source-manifest entries** match their bytes. The notebook's embedded manifest is identical and its embedded 47-file source/data ZIP matches the extracted candidate. Executed notebook source matches `input.ipynb`. `source-manifest.json` SHA-256: `3e4126a5e89e74b8653d1c868012e7b27bfdd188c71c7e276fb9763761e32b1e`.
- All **15 protected files** match the starter-integrity manifest and local starter commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`. Both data files contain 256 rows and match the established initial probe hashes: training `93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd`; evaluation `b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e`.
- `runs/result.json` SHA-256 **`b902c4044d75526c13a0e7782578332d380ad8882d443ea0554e2471030b382d`** equals `checks.json.result_sha256`. Recomputed official aggregate solution hash **`b63f5edc4caab3b80163c6a510a75bc153535d30e10fbaebb5d4290b015d794c`** equals `checks.json.solution_sha256`.
- `runs/checks.json` SHA-256: `786aab537874f8bb65983adc3114b08dd8920dc134982ca1b3f3e3701cd5c804`. Its entire parsed object matches notebook `FINAL_CHECKS`; every check name/detail agrees with `checks.log`. The result/check files are archive-bound; they are not entries in the source-only manifest.
- Retained `completed.ipynb` SHA-256: `4b4dd151fae0edd20a7ce34b22eb5c092df06b79651121eb1b65bed164ba6185`.

All 45 repository-sourced manifest files match the checkout and the local object for the supplied published commit **`0718eadb07ba36a55e6d569135a1040e71863f53`**. The two data files are supplied separately and absent from that checkout. All 28 solution files also match local HEAD **`d1fd86944388da74ac6e5794bed3869e80f93ce4`**. Remote publication was not queried. Observed checkout dirtiness is confined to `docs/round2/EXPERIMENTS.md` and `docs/round2/evidence/audit/`; it does not change these source bindings.

Against historical runtime `59755b01e594b65ccb5830b97637417b687e50ea`, the only changed existing solution files are `native_half.py` and `swiglu_pack.py`; additions are `bnb_provenance.py` and its test. The changes replace installed-package source inspection with loaded-code provenance validation. Native arithmetic classes, runtime-input guards, starter verification, hook guards, module guards, and installed forward ASTs remain unchanged. Attention and hybrid-MLP source also remain unchanged. Thus **none of the new round-two repeat-once, startup-map, tied-storage, or rematerialization experiments is promoted**. Existing baseline behavior remains active: `ENABLE_SAVE=True`, 1344 packed calls and 448 existing hybrid/rematerialization calls per solution run. Those old counters and the old `repeat_sdpa` primitive must not be mistaken for newly enabled round-two experiments.

**Full run and all-ten-check consistency**

The executed commands are the complete solution pytest suite, `harness/run_task.py --repeats 2 --timeout 240`, and the unmodified `harness/check_rules.py` without `--skip-tests`. Task seed is 1, `only=null`, `profile_run=null`, and the order is base_1 / solution_1 / solution_2 / base_2.

- Base runs: **80.40 / 86.17 s**, each **14345 MiB**.
- Solution runs: **66.47 / 66.82 s**, each **4421 MiB**.
- All four have exit code zero, **64 steps**, checkpoint global step 64, and 64 logged loss values. Each raw metrics object agrees with the result after removing `step_s`, exactly as the official harness does. Reconstructed trainer outputs, rewind records, adapter configuration and safetensors headers agree with the result. Summary recomputation and both raw evaluation JSONs agree. Evaluation uses the first successful adapter from each side, 256 rows, zero skipped rows and 38109 tokens each.
- All four final adapters and checkpoint-64 adapters are retained; each adapter tensor file is **8731128 bytes**. Training logs, per-step metrics, trainer states, rewind JSONL, configs, warm-up log, both evaluation logs, harness log, checker log, and solution-test log are present. Compiler caches are intentionally excluded by the original exporter.

Pure checks **1–9 reproduce the stored booleans and detail strings exactly**: complete runs; no banned imports/loaded libraries or package changes; eligible kernels launched in both runs; minimum share above 5%; task/output parity; loss-curve tolerance; evaluation tolerance; time/memory bounds; and T4/idle/one-GPU/allocator/peak consistency. Check **10** is consistently recorded passing by JSON, log and notebook, with the raw-transcript limitation above. This is an evidence audit, not a rerun of the 79 tests.

The eligible intersection is exactly `ember_rope_joint`, `head_loss_kernel`, `nf4_dequant_kernel`, `rmsnorm_bwd_kernel`, and `rmsnorm_fwd_kernel`. The protected source classifier rejects both `example_swiglu_fwd_kernel` and `example_swiglu_bwd_kernel`; they contribute **zero** to these calculations. Other own kernels not launched in every run also contribute zero.

- solution_1: **3374.727 ms / (1000 × 36.607 s) × 100 = 9.218802414838692053432403639…%**.
- solution_2: **3384.997 ms / (1000 × 36.784 s) × 100 = 9.202362440191387559808612440…%**, the minimum.

The denominator is the official timed training loop, **62 timed steps (steps 3–64)**, not whole-command duration or summed GPU profiler time. The exact fraction above is exact arithmetic from recorded rounded timings, not a claim of additional measurement precision. It agrees with the official rounded **9.2%** and exceeds the 5% threshold by about **4.20236 percentage points**.
