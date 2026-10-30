# Native-hybrid candidate r1

**Status: single full run + component passes; formal admission pending.**
The measured runtime completed one solution-only, seed-1, 64-step run on one
visible Tesla T4: **58.51 s, 4421 MiB, eval loss 2.04943**. The accompanying
79 official Soup regressions passed. These observations establish neither formal
admission nor a speedup. The parent restarted the full T4 solution suite and
ABBA after migrating the fallback fixture; those final results remain pending.
The previous suite's 271 passed / 126 skipped / 27 fixture setup errors are
historical evidence, not a successful full-suite result.

Source: `/tmp/decentrathon-native-hybrid-stage-r1`, variant
`packed_gqa_native_hybrid`, copied byte-for-byte onto parent
`50e84ed7b9fa9fbe7ddb8bb83de2e26c3cbcd73b` (the supplied `origin/feat/check4`
content snapshot). All 17 runtime files match the measured native-hybrid export.
Only 11 differing runtime files and `solution/test_head_fallback.py` were copied;
unchanged team NF4, RMSNorm, LoRA, rotary, residual norm and existing tests remain
intact. All 15 protected `harness/`, `task/`, `tests/` hashes are unchanged.
The expanded overnight integration scope was explicitly authorized; no merge is
authorized. Intended integration base: [head guard PR #5](https://github.com/alikhan-kai/decentrathon/pull/5)
(final guard `701d9a80ca0afdaf48606c065ce65e5be6308188`). This contribution does
not edit that PR. The parent handles remote upload and integration.

The corrected final eligibility and head/packing bridge scope packing to an
explicit head request. The stage adapter activates rotary, packed-row attention
and guarded MLP saves through `solution.apply()`. For up to 1024 rows the MLP
uses native FP16 saves; larger rows use RNG-preserving branch recomputation.
This trades recomputation for lower saved-activation memory. The native-half
parent observed 57.25 s / 5261 MiB / 2.04938 eval; the prior hybrid observed
58.96 s / 4521 MiB / 2.04943. These are separate single solution runs, not paired
ABBA comparisons. Packing changes dropout RNG assignment; the guarded MLP path
requires first-order eager training and pinned library provenance. The separate
unmeasured batched-attention prototype is excluded.

Credits: packed decoder donor **Ansar / iencislennon**, commit
`3e48f6a685fe6604d9d432e3753e35b7273999b8`, adapted with scoped admission and the
head bridge. Preserve team credit for the head-loss, NF4 and RMSNorm kernels
(iencislennon), corrected head gradients (stritphatographir), and joint Q/K rotary
work (team/Codex). The head-chunking idea credits Soup's `stream_loss_chunk`.
**All starter SwiGLU kernel credit is EXCLUDED**, including packed/native saves,
recomputation and both `do_not_specialize=["n_elements"]` decorators; these do
not make the starter kernels original. Existing LoRA remains the disabled team
stub. LICENSE and NOTICE are unchanged.

Jury compatibility: copy the complete `solution/` directory into the clean
organizer kit and explicitly call `import solution; solution.apply()` before
model/trainer construction (the unchanged harness already does this). Importing
`solution` wraps `apply` but does not install model patches. All candidate runtime
dependencies are inside `solution/`; no stage scripts, docs, `/tmp` files or
modified protected harness are needed. The target environment must supply the
pinned packages; the candidate performs no installation.

Local validation: the migrated original head fallback suite passed **28 tests**
on CPU in 5.47 s (`python -m pytest solution/test_head_fallback.py -q`, existing
venv, CUDA hidden and Hugging Face offline). Its fixture uses a fresh `_REQUEST`
ContextVar and checks cleanup plus clearing `STEP_ATTR`; runtime and numeric
tolerances were not changed. Python AST parsing passed for all 26 solution files;
static rule lint passed (10 kernels, zero problems); `git diff --check` passed.
No GPU test or full kernel suite was run in this contribution worktree.

Evidence root (local, read-only):
`/home/homelab/.t3/overnight/decentrathon-2026-10-30/evidence`.
The run is `native-hybrid/runs/result.json`, SHA256
`47f30cb1fb834fbf0d41b5fca7419d6c651fb6f6935dd65d46338aac65e4f516`;
regressions are `native-hybrid/regression.log`. Component passes are recorded in
`native/component.log`, `hybrid/component/cuda-results.json` and the deterministic
native composition gate `native/combined/evidence/native-gate/result.json`.
Component evidence does not substitute for full-task admission. Historical stage
metadata/docstrings retain their original bytes, including earlier pending labels.
The concurrently written `/tmp/decentrathon-overnight-handoff-r1` draft is not copied.

## Exact source SHA256 manifest

This is the frozen stage manifest (including the final migrated CPU fixture),
whose original JSON SHA256 is `27f230b800c2bd99ff109dd2ed3c7c0afbf3fbed4680d17a71ad6ce1e6e783dc`.
The entries also pin unchanged team sources, protected files and license notices.
Paths are relative to the repository root; documentation is outside this manifest.

```text
cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30  LICENSE
2f0089a88e9623e5f6eba497796f279c2fdb3a15be1df0973e4d0055d35876e4  NOTICE
45a62642dceab32cf3efcda504de87a9c25ea28b119b1806f56d14ff1146a89d  harness/_train.py
879ab1861674f6e714eadd00aaf044235d98526dd75fab1035deec3401121ef5  harness/check_rules.py
c83066f24f266dab62ebcc9c2d44f5a65902760b53da912e65ebfe511579c2db  harness/common.py
47cddbb8e4579c69f9535b099ff2371b1c3af08bfbd93d14bf79aadd74d38482  harness/eval_adapter.py
6a11055f4a1822e53971c228c6e3e3aa6b88edd9eb6db81a388e26a70a030ceb  harness/example_reference.py
d935a8959f348977313f2eef36870ff2699b9ba013a7978918ff35e148968a79  harness/run_task.py
731c9b644dcc4069b0b069616f438064d1f7f376a202feb15201fd910ec688c5  harness/score.py
66db347091a0789aa93fa87f4b2ebd0c719cb98d2c701dd346d36ce7c55e3db0  solution/__init__.py
1add9019d245aba7872f79bee155a86d8bbaab8ba47f548c956f23d621827e14  solution/attention_adapter.py
f4d2cdafe0ece75c4a2510ce44acfec14a6f819040e16e0dca3b282b55ab3dd1  solution/attention_gqa.py
fd66b7eef815487e5a80d201eab8dfc5706d1a1d0e83813ea90dfb7121a087b7  solution/head_loss.py
89421a52d5025bc1b35e53b0723ded7216551db4309ca6ef0440098e3ab3f199  solution/hybrid_mlp.py
f6b29c27738e727d2c10811dd47d51ac004b0a9f7f910f01c9dd16eb99970daa  solution/kcheck.py
e923126c5fad64c23b08eefc42edc5cdf0092fb09230e14ee14d79a3c2e7a806  solution/lora.py
b7e23022f8eeac3b46490a36582ac59cb6adc02960eb9cd61be5401bff06c30a  solution/native_half.py
94a4cd004cd83a09ec51794b9a05fd6d5da9d6e073f144b20e3e8a8cf9c93740  solution/nf4.py
263a8eb3bb7654aa2d152b78299308abbaed1b268c502b23ec56f88e48d595e2  solution/packed.py
6b453981e8570a9e448accde1422d4fbfc766d330d432ec6082a6f72d6de473c  solution/residual_norm.py
60bb02df363aab45cb9a32639850888d7203d784f83295bc713d4a93acd5b2c5  solution/rmsnorm.py
488522f8332e5402a85b5df74a9caaa080004c8ca9ab588f82c0f0ca292b2677  solution/rotary.py
f05b89a5c6b1d753bd5436c310c1ca32d27e13ce4a7a0e8478fa7131dc3c6194  solution/stage_adapter.py
3aff930958f2f91c94ae26a3fa21e07e5714671a6a9d96450c0094c38dcf2afd  solution/stage_options.py
e1dc639838c4c46ea0ed0aec3f0d904d360c92a91c601b943c9cfef85ca21c64  solution/swiglu.py
b87e20c4ca006a6674846cd72ae2a44e234f899d5e22a770a8932c8af8702d54  solution/swiglu_pack.py
ef034f42c7e110e4dab6b3ac6f5c0db0ea16d5187ec9034698b93bfb5a703ec3  solution/test_head_fallback.py
b8beecd53659f788312523f13930ed34ac5f7ce533095fb3049e7195233711f1  solution/test_head_loss.py
9d0e88d503f523366d7e5a0b02734a82efec9e043b7a5796ade03aaf7106b405  solution/test_kcheck.py
a45b842a4f22c1aefe404baeff1ac26511253452ecb0f633c985fbc9df8b252a  solution/test_kernels.py
356a6f480c252f6f124a44f1da5ae3fd226a72388b92f7c61ffba757499b85fc  solution/test_lora.py
7363fd34c10a9b76703e2f47eaa84b5e02175c4d1b59085782de07e7f9377e66  solution/test_nf4.py
f42ed5da71aff5c6d3b740ef3a20407902c999933257b5b372eb38e7653f5793  solution/test_residual_norm.py
639f8dca507c1ac54fd65f5ffd190d9af24653ee2b6c1732a92874e512739b74  solution/test_rmsnorm.py
c180e3abb37df60c71755987db94f94e410adada269b6c68d94c7d1aedc85d9d  solution/test_rotary.py
a8543f11a55aff80dbdc714a0e62391244a5296616eea2e34dc2eb3ff3ec18f7  task/prepare_data.py
dd70c739fcd143e06aef3bb1efcc9386f682bf1de26f30e08409d86a97ff47ce  task/task.yaml
a82eabab468ea0ef1e003cccd7c3827ce2f4d00d09c32948a34fb48221f7337a  tests/conftest.py
1919303a3670c0a33abbfce9b33eab0913bb64a3a21c3312d86531c32b5bf490  tests/test_callback.py
db808c44470d144b7cf18b7329492f41f76e37cfc24306e7cba0488bca9d787c  tests/test_config.py
5d656ed50c59f794e934473d459ea85635f892ba028e192414474e9b5d65919e  tests/test_formats.py
3dad1bea2d33aab97e52b7cd44888d768bc9519af44fbe6cf3e96c260026632f  tests/test_loader.py
b7608d62a40983bf7c9e5b270e0e527b641a837e462d6df2909fec88f607e821  tests/test_trainer_init.py
```
