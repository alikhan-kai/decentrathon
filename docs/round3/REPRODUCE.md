# Reproduce the selected runtime

Use the exact published branch/commit recorded in README and a single T4. The runtime requires the pinned task environment: Python 3.13, PyTorch 2.11.0+cu128, Triton 3.6.0, Soup 0.77.0, Transformers 5.16.1, PEFT 0.20.0, TRL 0.29.1, bitsandbytes 0.50.2, Accelerate 1.14.0 and datasets 4.8.5. Provenance guards deliberately fall back if the loaded BNB executable differs.

Use the repository's ordinary data preparation and unchanged task configuration. Our source-identity receipt records SHA-256 for the exact 256-row train and public-eval files; comparisons with different rows or seeds are separate experiments. Existing model/data downloads may be reused; every measured compiler cache is recreated by the official harness. Do not copy a warmed compiler cache into a run.

With that environment and data prepared, from the checkout:

```bash
export CUDA_VISIBLE_DEVICES=0
export TRITON_INTERPRET=0
python -B -m pytest solution -q -ra -p no:cacheprovider --tb=short
python -B harness/run_task.py --repeats 2 --timeout 240
python -B harness/check_rules.py
SOUP_APPLY_SOLUTION=1 python -B -m pytest tests -q -p no:cacheprovider
```

The official harness measures base, solution, solution, base and evaluates both adapters. Preserve its full result, individual metrics and training logs, checker output and test transcripts. Exclude every starter `example_swiglu_*` specialization from useful original-kernel share. The selected runtime has no benchmark-dependent branch selection.

The additional portable tests can also run in a fresh CPU interpreter process:

```bash
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 \
python -B -m pytest solution/test_native_half.py solution/test_frozen_remat.py \
  -q -ra -p no:cacheprovider --tb=short
```

The experiment's bounded supervisor is measurement infrastructure, not part of `solution/`. It verifies source hashes, cleans only its descendants, exports raw evidence and emits a separate final receipt. The ZIP's earlier job-status snapshot predates export; final success requires the separately saved all-true receipt bound to the ZIP digest. Source/result manifests and selected raw outputs are retained in `evidence/`; full archives also contain adapters and remain local to the experiment ledger.

Do not apply the later main layer-budget/precompile changes and reuse these results. That combined source requires new validation.
