# Independent lazy-BNB review, round 2

Reviewed 2026-10-31, starting 13:21:26 UTC; completed within the six-minute window. No children, GPU operations, browser/network access, installs, repository edits, or package-file edits. All review writes are under `/tmp/decentrathon-speed45-lazy-bnb-review-r2*`.

**Result: two reproducible issues to address.** The normal four-wrapper deferral and fallback implementation otherwise passed this bounded review. No GPU correctness or performance conclusion follows.

## Findings

### P2 — New staged candidates cannot run the supplied CUDA preflight

Locations: `/home/homelab/.t3/worktrees/decentrathon/speed48/docs/performance/startup45/stage.py:70-80`; `/tmp/decentrathon-speed45-lazy-bnb-r1/cuda_preflight.py:33-34`.

The new staging script enables lazy BNB by appending an outer `apply()` wrapper to `solution/__init__.py`. It copies the pinned `stage_options.py` unchanged. All three documented base revisions predate `ENABLE_LAZY_BNB_COMPILE`. The released preflight unconditionally imports that name and requires it to be true before calling `solution.apply()`.

Reproduced by staging exact `8f50dcbb5c7edacc4d6738a58b7d254274b85004` into this review's scratch directory with the released helper and scratch data fixtures. The candidate's `stage_options.py` contains only `ENABLE_SAVE = True`. Executing just the preflight's CPU-safe import statement raises:

```
cannot import name 'ENABLE_LAZY_BNB_COMPILE' from 'solution.stage_options'
```

Thus the new candidate is enabled but fails the documented preflight before its four-wrapper and genuine NF4 checks. This does not affect the original `candidate.patch`/`experimental` integration, which adds the option. Align the new staging and preflight opt-in protocols while retaining the admission/count/restoration checks. Do not interpret this failure as a BNB/GPU failure or simply remove the validation gate.

### P2 — Cleanup overwrites compiler patches installed after this scope has restored itself

Location: `/tmp/decentrathon-speed45-lazy-bnb-r1/candidate/solution/lazy_bnb_compile.py:184-195`.

Both restoration assignments write the captured `original` unconditionally. Particularly, after the fourth admitted decorator has already restored `torch.compile`, a later same-thread patch during the rest of `solution.apply()` is silently discarded by the final assignment. This is deterministic and does not require two threads racing:

```
with lazy.install(enabled=True):
    import bitsandbytes.backends.default.ops
    assert torch.compile is original  # early restoration succeeded
    torch.compile = replacement
assert torch.compile is original     # replacement was incorrectly lost
```

The probe reproduced this against the real installed BNB import, with four admitted functions and zero constructions/calls. It also reproduced ordinary scope cleanup losing a replacement. A concurrent replacement has the same problem; `_SCOPE_LOCK` coordinates only this helper's own install scopes.

Restore only while this scope still owns the installed callable, including at early restoration; add a regression where a replacement is installed after the fourth decorator, plus exception cleanup. Ownership checks address the demonstrated clobber; they do not promise atomic coordination with arbitrary third-party writers. The implementation report explicitly disclaims independently competing patches, and no current registry module assigning `torch.compile` was found, so this is a conditional integration bug rather than evidence of a failure in the present benchmark path.

## Reviewed behavior that holds

- Admission pins installed distribution version 0.50.2, package/source-loader paths, and both `__init__.py` and default `ops.py` SHA256 values. Both installed files matched the declared hashes in independent read-only checks.
- Admission additionally requires the importing main thread, genuine initializing module globals/spec, expected function/name/qualname, expected decorator/caller code and its exact `fn` local, exact options, and one interception per name. Real import admitted exactly `_dequantize_blockwise_compute` and `_dequantize_4bit_compute` with `dynamic=False`, and `_optimizer_precondition_32bit` and `_optimizer_update_32bit` with no options. Compiler restoration happened before the import returned; all four call/construction counters remained zero.
- The saved compiler receives the original function and unchanged positional values/options. The admitted production calls have no additional positional arguments. Unrelated calls, keyword-form calls, calls with other options, and other-thread calls take the saved compiler path; no `disable`, backend, or graph option is injected.
- Constructor `Exception` is remembered permanently. The private signal makes BNB's original wrapper perform exactly one eager invocation, including when that eager body raises. Runtime compiler failures instead fall through the original BNB per-call fallback, and later calls retry the compiled callable. `BaseException` propagates. First-call construction is locked; the focused eight-caller tests passed for success and constructor failure.
- Normal/error scope exit restores the compiler and releases the scope lock. Disabled, nested, already-imported and unsupported-source cases are ordinary/no-op paths. These checks do not resolve the competing-patch issue above.
- The released integration leaves the default false and imports the helper only on the enabled path. Its delta is exactly the helper, its test, `stage_adapter.py`, and `stage_options.py`; protected files remain byte-identical.

## Code identity test limitation

`fn.__code__ == expected` is structural code equality, not object identity or equality of every code attribute. In this Python 3.13 environment, changing only `co_filename` still compares equal. The existing test at `test_lazy_bnb_compile.py:256-257` creates a different function object, so its rejection also follows from `caller.f_locals['fn'] is fn`; it does not prove filename rejection.

A targeted probe temporarily changed `co_filename` on the *same* incoming function, preserving caller-local identity, and all four were still admitted. Original code was restored immediately. This is a provenance/test-assurance limitation, not a demonstrated numerical or arbitrary-function admission bug: source hashes, module origin, structural executable-code checks and caller checks still hold. If exact filename identity is part of the intended contract, check it explicitly and test it without changing function identity. The preflight's MatMul4Bit comparison likewise checks structural code equality, with a separate on-disk source-equality check.

## New staging review

The staged control was independently compared with `git cat-file` for every selected entry: **48/48 exact Git blob matches** for the pinned 8f50dcb revision. Both control and candidate retained identical `harness/`, `task/`, and `tests/` bytes. Lazy candidate delta was exactly:

- `solution/__init__.py`: outer explicit opt-in around the already-adapted `apply()`.
- `solution/lazy_bnb_compile.py`: released helper bytes.

The original registry/adapters and memory/kernel settings remain in place. The broader outer scope does not reorder them. The source-copy path reads commit objects rather than dirty working-tree runtime files; the helper intentionally comes from the checkout and is recorded by its resulting hash. The full-commit check, new-destination check, checkout-ancestor exclusion, and regular-file/path checks are suitable for these actual source trees. ROOT resolves correctly to the speed48 checkout. Data is copied once from the supplied inputs into both arms; my staging probe used scratch fixtures, not benchmark data.

The metadata variant was inspected statically only: it additionally changes the existing `PACKAGES_CACHE=True` switch to false when present and records this in `changed_files`/`old_cache_disabled`. That arm is a replacement of the old metadata cache, so its comparison must disclose the additional delta. No metadata-helper correctness conclusion is made here.

Parent removed the unused `shutil` import during review. I re-read the updated script; behavior and the finding above are unchanged. Executed staging snapshot SHA256: `e6572454773c4ab44cfa694380a57ad6e16bf602ea4e2506b71c877e4af8ee88`. Final re-read/saved snapshot: `cce4d39c359f9997d8d5dd486b71ea54085e148fb91275153cea9fbb7d95fc03` (`stage-reviewed.py` in review scratch).

## Evidence and limits

Independently executed **9 focused contract tests, all passed**, plus the targeted real-package import/ownership/filename probe and staging checks. No broad suite or numerical compilation rerun. Commands/scripts and raw results:

- `/tmp/decentrathon-speed45-lazy-bnb-review-r2/focused_tests.py`
- `/tmp/decentrathon-speed45-lazy-bnb-review-r2/focused-tests.log`
- `/tmp/decentrathon-speed45-lazy-bnb-review-r2/probe.py`
- `/tmp/decentrathon-speed45-lazy-bnb-review-r2/probe.log`
- `/tmp/decentrathon-speed45-lazy-bnb-review-r2/stage/manifest.json`

The released log reports 21 tests, 20 passed and one skipped. Its four real BNB tensor helpers all recorded one runtime `InductorError` after real compiler-wrapper construction; the logged cause was a missing C++ compiler. The numerical equality therefore establishes real BNB eager-fallback parity for the selected CPU inputs, **not successful compiled tensor execution**, CUDA dispatch/autograd parity, full `solution.apply()` correctness, or whole-command/memory improvements. The scalar successful compiler test does not fill that tensor-codegen gap. The report accurately discloses these limits. The CUDA preflight remains unexecuted in this review and belongs to the GPU owner.

Released helper SHA256 independently verified: `b0416ca4ef145e3d8c521269bf3971a5090618394d977f987b30a142b0f4c860`.
Released test SHA256 independently verified: `d92360b2edb30cd703d8c9841db527ab06117301246a0c05ad0639cccd1c14d7`.
