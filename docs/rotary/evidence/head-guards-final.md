# Head dispatch guard validation

Exact production files from reviewed local commit `701d9a80ca0afdaf48606c065ce65e5be6308188`, published as `5b4dd1f` on `fix/head-dispatch-guards`. Only eligibility changes; all remaining head AST matches corrected R8. The measured stage enables rotary explicitly, as in the prior R8 evidence.

Kaggle Tesla T4, GPU 0, pinned starter stack, unchanged official harness with two repeats. Base: 75.19 / 79.06 s, 14345 MiB. Solution: 69.02 / 69.77 s, 8367 MiB. Median solution 69.39 s, base 77.12 s; evaluation loss 2.04963 versus 2.04971. All ten checks pass; minimum eligible kernel share approximately 7.9%, excluding starter SwiGLU. These timings do not establish that eligibility checks improved performance.

The prepared NF4/q-v LoRA test uses Soup's actual FP32 trainable-parameter alignment, Accelerate FP16 autocast and GradScaler. It confirms bitwise equality against exact R8 for both tested upstream scales, including gradients, optimizer updates, metrics, Rewind row counts and RNG state. At scale 65536 the dense-reference gradient and storage-aware optimizer-update gates pass. **Scale 1 retains inherited R8 dense-gradient differences near FP16 underflow; this is recorded, not a numerical-parity pass.** The script and full log accompany this report. The test uses the Rewind row helper; full official runs exercise the recorder.

28 CPU trainer dispatch tests pass, including four global-hook regressions. Official regression check: 79 passed. Source hashes:

```json
{
  "head_loss.py": "bfb91304e15fde7dfe4a937a93d09b9dedba22dd1995a7342f216eb61180aa3f",
  "test_head_fallback.py": "0a425e3a7aff0ec909a8bdbb298f6a2de3ae95ec8a10ee69b439b4ba4eca0263"
}
```

Raw official result and matching checks are adjacent. The notebook is a private development run, not the final public submission notebook. No automatic merge or winner claim.
