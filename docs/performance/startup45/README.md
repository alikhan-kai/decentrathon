# Startup experiments for the current time-reduction pass

These experiments are disabled and have no T4 performance result. They target
startup inside the measured training process and must preserve the chosen
memory configuration. The independent memory task owns Kaggle; this pass used
CPU/source evidence only.

## Keep the baselines separate

| Source | Evidence | Wall time | External peak |
| --- | --- | ---: | ---: |
| `25855208695fd113fa1897385abccc73f71a19cd` | Exact-source Kaggle, two solution runs, all ten checks | 67.65 s median | 5327 MiB |
| Memory checkpoint branch `1470393` | Archived exploratory Colab record for its first-layer-checkpoint predecessor | 61.25 s | 3979 MiB |

The user additionally reports a Kaggle reproduction near 64 seconds / 3900 MB.
Its separate raw receipt has not been bound to this pass. The first row does
not refute that result: the tested configurations differ. Source `2585520`
uses head chunk 768 and activation budget 1800; the memory branch uses chunk
128, budget 768 and a checkpoint of the first packed decoder layer. Do not
replace that branch's policy when trying a startup helper.

The exact `2585520` measured command splits into 37.98 seconds training,
14.075 seconds in `solution.apply()`, and 15.595 seconds elsewhere. These
are disjoint intervals; background compilation overlaps them. There is no
current complete operator timeline that attributes all remaining time.

## CPU experiment evidence

**Final source, four fresh processes per arm**, importing the real Qwen/TRL/
Trainer/SFTTrainer/Soup dependencies through Soup CLI:

| Independent experiment | Control import span | Candidate import span | Difference |
| --- | ---: | ---: | ---: |
| Scoped metadata reuse | 2.820308 s | 2.645431 s | 0.174877 s |
| Lazy BNB compiler construction | 2.854159 s | 2.144528 s | 0.709631 s |

All 16 processes restored the original functions, returned equal package
mappings, and left CUDA uninitialized. Lazy processes admitted exactly four
helpers with zero first calls and did not load Inductor's compiler entry point.
Each row is a separate comparison; there is no combined measurement. Raw
commands, logs, versions and exact helper hashes are in
[evidence/results.json](evidence/results.json). These measurements supersede
the earlier prototypes below for the delivered source. They do not establish
whole-command T4 savings, GPU memory reduction, or the target result.

The first metadata prototype reused Transformers' already-computed package
mapping during imports. Six fresh processes per arm measured median import
spans of **2.676050 → 2.480989 seconds** and process times of
**5.493977 → 5.271244 seconds**. Every measured returned mapping matched a
fresh standard-library scan. This is CPU import evidence, not training or T4
evidence, and does not establish safe reuse of stale or customized mappings.
Review found lifecycle and composition defects in that first prototype. The
repaired helper is accepted only for the documented fresh static environment:
installed distributions, search providers and Transformers' mapping must have
remained unchanged since that mapping was constructed. It rejects ordinary
pre-existing hooks/custom resolvers, restores its owned wrapper, and lets nested
calls retain the outer search-state decision. It is not a general metadata cache.

A separate preliminary probe deferred construction of four BNB default-CPU
compiler wrappers until first use: **2.6212 → 1.8720 seconds** for CPU imports.
The complete implementation must retain the real compiler and CPU fallback
semantics. Those preliminary timings do not qualify a later implementation.
Do not add the two savings: a combined configuration has not been measured.

The final lazy helper also preserves a later replacement of `torch.compile`
on both early restoration and scope cleanup. It authenticates the four exact
BNB 0.50.2 source functions and defers only wrapper construction. The original
compiler, function, options and BNB per-call fallback remain responsible for
execution. Unknown or already-imported BNB retains the ordinary path.

## Reproduce and stage

No existing runtime file is changed in this contribution. Both helpers require
an explicit opt-in; importing the solution leaves them inactive. Use a fresh
process for each test group:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -B -m pytest -q -p no:cacheprovider solution/test_startup_metadata.py
PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest solution.test_lazy_bnb_compile -v
python -B docs/performance/startup45/measure_imports.py \
  --python /absolute/path/to/cpu/python --output /tmp/startup-import-evidence
```

The portable metadata subset passed **22 tests**. The lazy helper passed
**21 tests**, with **one additional test skipped** because the local C++
compiler is absent. Real BNB tensor tests exercised its genuine runtime
fallback; compiled tensor success and CUDA behavior remain unverified.
The fuller scratch metadata suite passed 29 tests, including the actual old
cache and three-base composition checks. Six public-stager combinations
preserved all 15 protected files and every unrelated runtime file.

Build a new control/candidate pair from the exact desired base, rather than
replacing the memory task's solution with this branch:

```sh
python -B docs/performance/startup45/stage.py \
  --repo /absolute/path/to/decentrathon \
  --commit 1470393c64c09a6be6e48bd643f1a5de07574d19 \
  --variant lazy-bnb --output /tmp/startup-lazy-memory-pair \
  --train-file /absolute/path/to/data/train.jsonl \
  --eval-file /absolute/path/to/data/eval_public.jsonl
```

Use `--variant metadata` for a separate experiment. On `8f50dcb`, that variant
also disables the old `PACKAGES_CACHE` flag because its shallow copies leak
list mutations. The control retains the flag, and the manifest discloses this
additional change. Other overhead flags and the memory policy are preserved.
The builder refuses moving refs and existing output directories; it never
starts a GPU job. Its manifest binds both source trees and dataset bytes.

The GPU owner can run the **unexecuted diagnostic** below in a separate fresh
process after arranging exclusive GPU ownership. Its manifest, four-helper
admission, restoration and genuine NF4-autograd checks must all pass. It is
not a benchmark warmup or full-task qualification.

```sh
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=5s 120s \
  python -B docs/performance/startup45/lazy_cuda_preflight.py \
  --stage /tmp/startup-lazy-memory-pair/candidate --output /tmp/lazy-preflight.json
```

Keep the existing owner-managed bounded runner for full official jobs and
descendant cleanup. No GPU test or training run was performed in this pass.

## Gate before enabling an experiment

1. Preserve the exact memory-owner source in a separate control directory.
   Add only the reviewed startup helper and its explicit opt-in to a copied
   candidate; record every file hash and the delta. Keep all task, harness,
   dataset, MLP, head, optimizer and diagnostic settings unchanged.
2. In a fresh process, verify restoration of temporarily wrapped functions,
   the intended dispatch counts, real BNB operation/fallback behavior and
   canonical provenance. A skipped optimization is not a successful candidate.
3. Only in a coordinated GPU window, compare unchanged complete cold-cache
   commands in alternating order. Record wall time, phase timing, memory,
   clocks and source identities. Avoid concurrent GPU work.
4. Require all official checks, the regression suite, correct metrics/Rewind,
   and useful original-kernel share with starter SwiGLU excluded. Reject a
   candidate whose complete-command improvement is within repeat variation.

No tokenizer fingerprint shortcut, tokenizer-hash cache, fake package,
compiler-cache reuse across measured runs, altered allocator setting, early
process exit, or unmeasured enabled default is introduced by this work.
