# Residual alignment experiment

This isolated candidate specializes hidden width, leaving sequence lengths,
input gradient strides and missing branches dynamic. It also removes the second
identical eligibility check inside the decoder's already-validated branch.
The production residual module and its disabled default remain unchanged.

The compiler derives alignment from the width; no unproved alignment hints are
added. Odd widths and unaligned storage offsets remain supported. Independent
review requested a real CUDA offset-pointer check; `test_alignment.py` includes
that case at widths 17 and 1536 without cloning away the offset.

On the pinned single-T4 environment, from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest test_residual_norm.py test_alignment.py -q
CUDA_VISIBLE_DEVICES=0 python benchmark.py --iterations 50 --warmup 10 --output benchmark.json
```

The benchmark compares complete allocating forward/backward calls against the
current team add-plus-RMSNorm and existing residual fusion. It checks dense
oracles, scale 65536, cancellation and cache reuse before collecting seven
interleaved timing blocks. Event timing includes host launch gaps. Component
results alone do not establish a full-training speedup.
