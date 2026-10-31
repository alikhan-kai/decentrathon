# Rotary contribution and validated head correction

The rotary work and the separately reviewed loss-gradient correction are merged
into `feat/check4` at
[`075c38bd0ac769196656da079719ea53b820c819`](https://github.com/alikhan-kai/decentrathon/commit/075c38bd0ac769196656da079719ea53b820c819).
This note makes the integration handoff discoverable from the default branch;
it does not enable or change any kernel on `main`.

Contribution authored through [@stritphatographir](https://github.com/stritphatographir):

- [PR #2](https://github.com/alikhan-kai/decentrathon/pull/2): joint Q/K rotary
  forward/backward kernels, tracing fallbacks, tests, research and raw T4 evidence.
- [PR #3](https://github.com/alikhan-kai/decentrathon/pull/3): apply the upstream
  gradient scale before fp16 rounding and recompute only supervised rows in backward.

The measured configuration passed all ten official checks over two T4 repeats:
78.07 s median command time, 8367 MiB peak memory, 7.0106% minimum eligible kernel
share, and reported evaluation loss 2.04966. These are aggregate team results,
not an isolated rotary speedup or a speedup over the previous incorrect head.

See the pinned [integration instructions](https://github.com/alikhan-kai/decentrathon/blob/075c38bd0ac769196656da079719ea53b820c819/docs/rotary/README.md),
[source manifest](https://github.com/alikhan-kai/decentrathon/blob/075c38bd0ac769196656da079719ea53b820c819/docs/rotary/evidence/r8-final-manifest.json),
and [official checks](https://github.com/alikhan-kai/decentrathon/blob/075c38bd0ac769196656da079719ea53b820c819/docs/rotary/evidence/r8-final-checks.json).
The measured snapshot selects rotary explicitly; the shared registry still needs
that selection during final integration. Residual fusion remains disabled.
Any final configuration combining other team changes needs its own validation.
