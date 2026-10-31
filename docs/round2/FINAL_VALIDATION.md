# Round-two exact-candidate qualification

**All ten official checks passed. Latest result: 66.64 seconds / 4421 MiB. The 54.8-second target was not met.** The memory target of 4693 MiB was met. The competitor result was user-reported and was not measured in this session.

| Side | Repeat 1 | Repeat 2 | Official median | Maximum peak | Evaluation loss |
|---|---:|---:|---:|---:|---:|
| Organizer base | 80.40 s | 86.17 s | 83.28 s | 14345 MiB | 2.04962 |
| Corrected packed/native solution | 66.47 s | 66.82 s | 66.64 s | 4421 MiB | 2.04942 |

Order was base / solution / solution / base, using the unchanged official harness, full 64-step task, fresh per-run compiler caches, one visible T4, and unchanged dropout/optimizer/data. Official speedup is 1.2497× and memory ratio 3.2447× against this matched organizer base. These measure the complete inherited candidate plus the provenance repair, not the incremental benefit of that repair. No incremental speed claim is made. Historical PR #7's 59.52 seconds came from a different session and must not replace this result.

Validation: 302 solution tests passed, 126 skipped; all 79 official Soup regressions passed; all ten admission checks passed. Original eligible kernels account for an independently recomputed minimum 9.202362440191388% of warmed training time. All starter SwiGLU time is excluded. Exact-source and protected-file manifests are preserved alongside the raw result and checker output. All 15 protected files match the starter.

## Selected source

The only round-two runtime change replaces installed BNB source-file reads with fingerprints of already-loaded code, defaults and class structure. It preserves the existing computation and guard contract. It is pinned to CPython 3.13 and BNB 0.50.2; unsupported provenance declines admission, and the stage installer may reject an unsupported environment. This is a compliance repair, not a speed optimization.

The native-half, packed decoder, efficient attention, scaling-correct supervised head and rotary paths remain as in PR #7. Repeat-once attention, fast dataset mapping, tied-weight storage and custom MLP rematerialization were not promoted. See [EXPERIMENTS.md](EXPERIMENTS.md) for failed and inconclusive results and donor attribution.

Draft [PR #9](https://github.com/alikhan-kai/decentrathon/pull/9) stacks on [PR #7](https://github.com/alikhan-kai/decentrathon/pull/7). No automatic merge was performed. The measured source manifest, rather than documentation-only commit IDs, identifies the exact runtime.

## Evidence

- [Official result](evidence/final/result.json), SHA256 `b902c4044d75526c13a0e7782578332d380ad8882d443ea0554e2471030b382d`.
- [All ten checks](evidence/final/checks.json), binding that result and solution digest `b63f5edc4caab3b80163c6a510a75bc153535d30e10fbaebb5d4290b015d794c`.
- [Source manifest](evidence/final/source-manifest.json): 47 source/data/license entries, all verified after export and against the checkout.
- [Full solution-test log](evidence/final/solution-tests.log), [harness log](evidence/final/harness.log), [checker log](evidence/final/checks.log).
- [Executed notebook](evidence/final/completed.ipynb): frozen historical record with embedded exact source/data; its dated deadline deliberately prevents later execution.
- Full raw ZIP: **146330249 bytes**, SHA256 `395d26c3ce5e94836ed95010a080feca2d98af8021b4b3427d3b449d6bfa92e8`. It contains all four runs' logs, adapters, checkpoints, metrics and Rewind data, plus source and checks. Retained locally at `/home/homelab/.t3/overnight/decentrathon-2026-10-31-round2/evidence/final/raw.zip`; it is not a remotely hosted release asset.

Runtime: T4, Python 3.13.15, torch 2.11.0+cu128, Triton 3.6.0, Soup 0.77.0, Transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2, Accelerate 1.14.0, datasets 4.8.5. The full version inventory is in result.json. No packages changed during measurement.

## Reproduction

Use the pinned environment and complete solution directory with the clean organizer kit. Prepare the public dataset before measuring:

```bash
export CUDA_VISIBLE_DEVICES=0
python task/prepare_data.py
python -B -m pytest solution -q -p no:cacheprovider --tb=short
python -B harness/run_task.py --repeats 2 --timeout 240
python -B harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q -p no:cacheprovider
```

Do not use the frozen notebook as a reusable job launcher. Review identified that its timeout handler kills only the direct process group, while harness children create separate sessions; export hashing also lacked a deadline. This run completed normally before 08:07 UTC, with no timeout or interruption. The separately tested post-run supervisor captures the raw regression transcript and verifies cleanup/export boundaries. It does not retroactively change the supervision used for the benchmark.


## Post-run capture and shutdown

A separate bounded post-run invocation repeated the identical patched regression command and retained its complete [stdout](evidence/final/raw79.stdout.log) and [stderr](evidence/final/raw79.stderr.log): **79 passed**. [Cleanup evidence](evidence/final/cleanup-before-export.json) verifies all helper-owned descendants exited before exporting; [integrity evidence](evidence/final/integrity-report.json) rechecks source and protected files. The [completed supplemental notebook](evidence/final/postrun-completed.ipynb) records successful export, final digest and GPU idle readings of 3 / 0 MiB. The underlying benchmark was not rerun or altered.

Supplemental ZIP: **146344419 bytes**, SHA256 `2c3fa0329616598a82903b10570cacd20acfdc3980fc14749a2fe67730e9bd47`; retained at the local ledger's `evidence/postrun/raw.zip`. This includes the original full run plus raw regression and cleanup evidence. [Independent final audit](evidence/final/independent-audit.md) verified all four 64-step runs, hashes and exact eligible share; its then-pending supplemental transcript was completed afterward as recorded here.

The owned Kaggle scratch session was stopped at approximately **08:13:54 UTC**, after verified exports. Conservative elapsed GPU-session usage was **5453 seconds (90m53s)**, below the two-hour limit. Both round-two scheduled tasks were disabled. No new experiments, automatic merges or GPU restart are authorized by this completed window. The earlier overnight window remains completed.
