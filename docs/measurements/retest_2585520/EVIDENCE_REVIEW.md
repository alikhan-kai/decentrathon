# Latest-commit evidence review, round 1

**Verdict: evidence supports the requested unchanged-commit benchmark result. No blocking evidence inconsistency found.** Scope: local, read-only audit of commit `25855208695fd113fa1897385abccc73f71a19cd`; no experiment rerun, GPU query/use, browser, network, installs, children, repository edits, or kernel-cache changes. Only this report was written.

## Verified provenance

- `evidence/latest-2585520/raw.zip` under `/home/homelab/.t3/experiments/decentrathon-time-2026-10-31/`: **146352845 bytes**, SHA-256 **`2c2cbc68c285e4d6684049fcd0fcf196fa59ea82a70dbc2875a96f31f864b939`**. All 171 ZIP members pass CRC verification.
- Independently hashed **52/52 manifest entries** against archive bytes. The 50 staged baseline files match `/tmp/decentrathon-latest-2585520/source-identity.json`: **48 Git-tracked files match the exact requested commit**, plus two datasets not present in that Git tree. Thus “50 Git source files” would be inaccurate. Both runner files match the local prepared runner bytes.
- Dataset SHA-256: train `93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd`; eval `b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e`. Both match requested identity hashes and archive bytes.
- All **15 protected files** match the starter-integrity hashes and starter commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
- Manifest SHA `9e0bcad3fe19da63e09da69931b6221aa8fb708ff573b560601025d830916b96` matches `source-verification.json`. Checks bind the exact result bytes: SHA `234b25a9d9f8630be3ebdfbd20df3dbf9a1377237fbf98c1b713a46dbad7bad9`. Independently recomputed the harness's filename-plus-content solution digest: `3bf7c36c11946c0354e02b3e9cd1bda966dc3bcbeb4b8b3db1d54ecb66487660`, matching `checks.json`. Extracted result/checks/manifest/status copies equal their archived counterparts.
- Completed notebook cell source equals `/tmp/decentrathon-latest-2585520/official-latest.ipynb` exactly after joining the valid JSON string/list source representations; source SHA `0bc6809d540d5722a05e42d707131383de7b78ef0e240e5b06ba90de8781152d`. Its embedded ZIP SHA is `0b21836dcc8b718c60fe2e106ed00b5e8449fb6c0dfc8f88829f7efdbbdfeeb2`; all 54 embedded members equal exported counterparts.

## Verified result

- Solution **66.80 / 68.50 s**, median **67.65 s**, externally sampled peak **5327 MiB in both runs**. Base **86.94 / 92.66 s**, median **89.80 s**, peak **14345 MiB in both**. Medians independently recomputed; speedup 1.3274x.
- **10/10 official checks pass**, including the reported **79 Soup regression tests** (14 warnings). Recomputed eligible kernel shares: **10.08430022642433% / 9.523212280044929%**, excluding both example SwiGLU kernels; minimum exceeds 5%. Denominator is timed training steps 3 onward, not whole-command wall time.
- Every run has metric steps, checkpoint global step, and 64 logged losses agreeing at **64**, with **52890 training tokens**. Per-run metric files match result metrics after the documented removal of `step_s`.
- Trainer configuration is identical across all four runs: RewindSFTTrainer; FP16 enabled/BF16 disabled at trainer level; batch 4, accumulation 1, epoch 1, max steps -1, LR 0.0002, seed 1, AdamW/cosine, max length 512, no packing/checkpointing/torch compile. Task is Qwen2.5-1.5B-Instruct, 4-bit; saved adapters have rank 16, alpha 32, dropout 0.05, q/v targets, no DoRA. Trainer precision flags alone do not prove every intermediate or parameter dtype.
- Eval **2.04947 solution vs 2.04959 base**, 256 rows, zero skipped, 38109 tokens. Archived eval JSON matches result eval entries after the harness adds exit code 0.
- Solution PyTorch peaks **4917.8 MiB allocated / 5184 MiB reserved** are supporting, distinct metrics. The scored **5327 MiB** is external all-GPU memory sampled every 100 ms; do not substitute allocator figures or describe this as a just-over-4-GiB external peak.

## Completion and limits

The archived status has complete/cleanup true but export/source false because it was written before export. Runner source confirms that ordering. The **completed notebook's final JOB_STATUS has all four flags true**, and its ARTIFACT receipt matches the local archive size/hash. This is the final recorded receipt; current GPU/session shutdown was not queried or independently certified.

These are two timing repetitions on the fixed public task; evaluation covers the first successful adapter per side, not both repeats. Cold compiler caches are supported by the unchanged harness's per-run directory reset and cache settings. The archive is internally consistent evidence, not an independent rerun or signed hardware attestation; source hashes bind saved bytes, not a continuous proof against transient runtime modifications. Remote “latest” status was not checked; the exact requested commit was verified.

Official checks establish their stated tolerances, **not bitwise, gradient, or optimizer-update equivalence**. The independent `decentrathon-time-new-lora-review-r1.md` reports a CPU mixed-precision LoRA pre-dropout cast/gradient mismatch; this audit read that report but did not reproduce it. Passing the official benchmark does not clear that finding, and the CPU finding does not invalidate the measured passing result. The 79 tests are the official Soup regression suite, not certification of every custom kernel or LoRA equivalence.
