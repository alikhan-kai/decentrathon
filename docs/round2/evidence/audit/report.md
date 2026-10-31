Review handoff — completed round-two evidence only, 2026-10-31 UTC.

**PR #9 contains the loaded-code BNB provenance repair; current full official admission remains pending the parent's final results.** Local Git objects confirm supplied published commit `0718eadb07ba36a55e6d569135a1040e71863f53` and local `a6fa63b39015957c0ea4d108f42ba8e567788af7` have identical tree `6de497cffc5afeade8e5ed5853c60c976efaded3`. No live remote query was made. Map, repeat-once and remat r1/r2 remain rejected for promotion. Pending r3 evidence was neither read nor awaited.

| Evidence / variant | Command seconds | Peak MiB | Eligible original share | Interpretation |
|---|---:|---:|---:|---|
| Historical PR #7 base | 75.09 / 78.19; median 76.64 | 14345 | — | Notebook-recorded organizer base |
| Historical PR #7 solution | 59.40 / 59.65; harness median 59.52 | 4421 | Minimum printed 10.2%; exact unavailable | Ten checks recorded pass; final raw result/zip unavailable |
| Opponent | 54.8 | 4693 | Unknown | User report; unmatched and unverified |
| Fresh round-two controls | 73.45 / 68.39 | 4421 | 10.2856% / 10.0069% | Separate solution-only controls; do not pool with historical score |
| Map ABBA controls | 63.11 / 66.16; median 64.635 | 4421 | Minimum 9.1769% | All 18 runtime files match current PR #9 local source |
| Map ABBA candidate | 63.76 / 63.96; median 63.860 | 4421 | Minimum 9.2059% | Reject: 0.775 s median gap versus 3.05 s control range |
| Repeat ABBA controls | 68.49 / 68.19; median 68.34 | 4421 | Minimum 9.1864% | Predates provenance repair |
| Repeat ABBA candidate | 69.13 / 66.67; median 67.90 | **4419** | Minimum 9.1886% | Reject: 0.44 s median gap versus 2.46 s candidate range |

ABBA order is control/candidate/candidate/control. These are solution-control comparisons, not full organizer base/solution admissions. Both candidates increased the median timed training loop: map 34.9515 → 36.3515 s; repeat 35.7345 → 36.9870 s. Two samples per variant establish no repeatable command-time benefit. Historical PR #7 is numerically 4.72 s slower and 272 MiB smaller than the opponent report; this is not a matched comparison or a winning claim.

Recomputed share uses `100 × eligible_kernel_ms / (1000 × timed_loop_s)` for harness steps 3+, counting only NF4, RMSNorm forward/backward, head loss and joint Q/K rotary. Both starter `example_swiglu_*` kernels are excluded completely. Counts and source paths confirm the five eligible kernels launched in each archived run. Every recomputed share exceeds 5%; that arithmetic does **not** certify all ten checks. Per-run numerators, denominators and hashes are in [comparison.json](comparison.json).

Concrete evidence findings:

- All **203 map** and **201 repeat** archive members match their extracted bytes; all **55/53 source-manifest entries** match. All ten copied round-two result JSONs match their raw counterparts. Map controls match all 18 current runtime files; repeat controls differ in `native_half.py` and `swiglu_pack.py` and lack `bnb_provenance.py`.
- Remat r1/r2 raw GPU reports both say `NO_GO_COMPONENT`, with bitwise output/input-gradient/RNG cases passing and the memory gate failing. R1 component peaks increase by 33.8101 / 38.7920 / 48.25 MiB at 1025/1427/2048 rows. R2 still increases by **4 MiB at 2048 rows** (379584512 → 383778816 bytes). Their `assessment.json` “GPU untested” labels are stale; component gains cannot support full-task claims.
- Startup r5's original comparator failed on an absent placement map; its raw `compare-map.json` is a traceback. The corrected diagnostic reports text-map time 2.7216 → 0.5519 s and completed 64-step runs, but explicitly limits hook evidence to metadata. Later raw map ABBA supplies command measurements and still rejects promotion.
- The archived provenance gate records `PASS`, canonical BNB 0.50.2, Python 3.13.15 and Torch 2.11.0+cu128; the archived CPU report records six passing tests. These component results do not replace pending final admission.
- Historical notebook SHA256 independently matches `e290a582d7ca83749dd518fb643fe4aa970f4990d14818904796b8fe91abfd70`; its output contains the two runs and ten passing checks. The printed raw-result digest `5379fe6a…` is **not independently recomputed** because that file is missing. Exact historical share, final per-step raw logs and final adapters remain unavailable. Historical results must not be relabeled as PR #9 qualification.

Source credit retained: Ansar / iencislennon packed decoder (`3e48f6a…`) and repeat-once idea (`375a0a4…`); iencislennon head-loss/NF4/RMSNorm; stritphatographir head-gradient correction; team/Codex joint rotary; Soup `stream_loss_chunk` head-chunking idea. Starter SwiGLU, wrappers, saves, recomputation and decorators earn no new original-kernel credit. Existing LICENSE/NOTICE credit Soup and its contributors. Sources: repository `docs/overnight/CANDIDATE.md`, `docs/overnight/FINAL_VALIDATION.md`, `docs/round2/EXPERIMENTS.md`, and the cited raw artifacts.

The [raw hash inventory](/tmp/decentrathon-round2-handoff-r1/raw-hash-inventory.json) records SHA256 and byte size for **1,484 files**, including all 11 completed export archives; [SHA256SUMS.raw](/tmp/decentrathon-round2-handoff-r1/SHA256SUMS.raw) is directly usable for verification. The key archive hashes are:

| Raw archive | SHA256 |
|---|---|
| map-abba/raw.zip | `6a67d5493829a9c94f38402228b7bf89683dcf2e731b3bb87940d89e2edf3da8` |
| repeat-abba/raw.zip | `e9256d844ed8d9f84f882294619d3f87a3554759ce3b43df02b4f710c4f488af` |
| remat-gate/raw.zip | `550ff11a8bd03552682ed099448b9a56e97eb093b5c4a1d190186b03f8155852` |
| remat-r2/raw.zip | `57506e12ddf821645a24ea443b6c2de6968469968fd3d538c28005f79af5d3be` |

Audit writes are confined to this handoff directory. No repository edits, GPU work, browser, network, installation or child agents were used. Parent supplies final qualification and pending r3 results.
