# R5 narrow review: r2 → r3 GradientEdge delta

Reviewed 2026-10-31, approximately 07:45–07:49 UTC, within the four-minute budget. This is independent of the active r2 review and covers only the change in `candidate/solution/frozen_remat.py`. No repository edits, GPU work, network, browser, installs, or delegated children. The only intentional file write is this report. Existing CPU environment: `/home/homelab/.t3/projects/decentrathon/.venv/bin/python`, PyTorch **2.11.0+cpu**, bitsandbytes **0.50.2**. Probes ran with `CUDA_VISIBLE_DEVICES=''` and bytecode writes disabled; the full-function probes mocked CUDA RNG calls as the existing CPU diagnostics do.

## Verdict

**The delta passes the inspected CPU lifetime, storage, topology, arithmetic, and mutation checks.** Independent actual-nested-NF4 probes demonstrate that r3 releases both gate/up output storages before either nested backward, while keeping the weights and graph needed for backward alive. No unconditional correctness defect was found in the inspected delta.

**Conditional compatibility concern at r3 lines 86–88:** the target PyTorch must preserve ownership of Python autograd nodes when constructing `GradientEdge`. The installed CPU implementation explicitly provides an ownership token; a deliberately tokenless control fails after deleting the output. This is a demonstrated ownership requirement, not evidence that the T4 runtime is affected. Its implementation/version was not established in this narrow review. The queued actual T4 component/prepared gates must still pass; this report does not clear the N=2048 peak-nonincreasing gate or claim GPU memory/latency results.

The supplied r2 T4 correctness/timing result and r3 36-case CPU fanout result are background evidence from the parent, not reruns performed here.

## Exact scope and source identity

- r2: `/tmp/decentrathon-round2-remat-r2/candidate/solution/frozen_remat.py`
  - SHA-256: `8a6e8e7469ae8e260eff53b96ce6fc153c2f63799269c686c75ad5510586f861`
- r3: `/tmp/decentrathon-round2-remat-r3/candidate/solution/frozen_remat.py`
  - SHA-256: `13167bdbc49ae33dab83bc1a97bea3baad6f1d60066cebe693612a33ad51a7a9`

The diff consists of capturing two gradient edges, deleting `gate`/`up` before the first nested grad, passing edges instead of tensors to the same grad calls, and deleting edges at the corresponding previous tensor-deletion sites. Supporting inspection was limited to the CPU probe helpers and installed PyTorch/bitsandbytes implementations needed to evaluate this change.

## Lifetime and storage proof

Installed `torch/autograd/graph.py:214–239` obtains the tensor's grad_fn/output number. For Python `_FunctionBase` nodes it also retains `tensor.view_as(tensor).grad_fn` as `ownership_token`, because the Python grad_fn wrapper alone does not own its C++ graph. The view contributes graph ownership without requiring retention of the output data storage. This was verified by storage-lifetime observations, not inferred merely from Python tensor collection.

Installed `bitsandbytes/autograd/_functions.py:300–384` stores the quantization state and packed weight view (`ctx.state`, `ctx.tensors = (None, B)`) for nonempty MatMul4Bit inputs. It does not save its output. Backward dequantizes B and multiplies the supplied cotangent by it. Thus the real graph needs frozen weights/quantization state but does not need its projection output storage.

Independent probe 1 imported **each exact r2/r3 helper** and executed its forward/backward with actual `Linear4bit`, NF4, nested quantization, frozen weights, FP16 compute, dimensions H=64/I=128, and input shape `[1,33,64]`. Only the starter kernel was replaced by the existing CPU arithmetic shim. At the SwiGLU backward call, the probe recorded `torch.multiprocessing.reductions.StorageWeakRef(t.untyped_storage())` for gate/up, then observed expiration immediately before each real nested `torch.autograd.grad` call. It did not retain those outputs or their strong storage references.

Observed storage expiration, in `[gate, up]` order:

| Version | Before up backward | Before gate backward | Nested roots |
|---|---|---|---|
| r2 | `[false, false]` | `[false, true]` | Tensor, Tensor |
| r3 | `[true, true]` | `[true, true]` | GradientEdge, GradientEdge |

The two inputs to the nested calls were distinct FP32 leaf objects; both cotangents were FP16. r2/r3 output and final input gradient were **bitwise identical**.

Independent probe 2 constructed a real nested-NF4 projection, obtained its reference gradient with `retain_graph=True`, captured its gradient edge, and then deleted **both module and output**, followed by garbage collection. Results:

- Root node: `MatMul4BitBackward`; ownership token: `ViewBackward0`.
- Output tensor weak reference dead; output **storage weak reference expired** before grad.
- Weight **storage weak reference still live**, despite deleting the module.
- `autograd.grad(edge, input, cotangent)` succeeded and matched the reference bitwise.
- Weight storage expired after backward and deletion of the edge.

Negative control: deleting a sigmoid output killed its Python tensor wrapper but **did not** expire its storage while the edge retained the saved result. That storage expired after backward/edge deletion. This establishes that the probe distinguishes Python wrapper lifetime from real saved-storage retention.

CPU storage expiration proves that the corresponding allocations can be released; it does not establish CUDA allocator peak behavior, stream-reuse timing, process RSS reduction, or the final whole-component peak.

## Separate topology and arithmetic

- r3 lines 68–71 still create independent `hidden_gate` and `hidden_up` leaves and separate FP16 cast nodes. Shared underlying input storage does not make them one autograd leaf.
- Lines 90–96 still perform up backward first, gate backward second, and return `(dhidden_up, dhidden_gate, None, None)` to the unchanged duplicate outer input edges. There is no new branch accumulation inside nested autograd.
- The edge identifies the same root node and output number as the original output. Its ownership token is a lifetime device; the grad call still targets the original node.
- Kernels, projection order, dtype conversions, FP32-to-FP16 cotangent rounding, gradient arguments, `create_graph=False`, and `allow_unused=False` are unchanged. The new calls/deletions introduce no numerical operation or RNG consumption. The metadata view used by the installed API does not perform floating-point arithmetic.
- The unchanged first-order guard rejects higher-order work before this delta is reached. No broader graph-scheduling guarantee is inferred from the reduced CPU probe.

## Static mutation semantics

The delta introduces no parameter snapshot, clone, detach, parameter write, quantization-state mutation, or change to RNG/autocast replay. The module is still retained on ctx and its current weights are still used during rematerialization; the original input remains saved through `save_for_backward`.

Independent actual-NF4 full-helper checks, seeded identically for r2/r3:

1. Unmutated execution: input gradients bitwise equal.
2. Original input modified in place between forward/backward: both versions raise the saved-input version error (`modified by an inplace operation`).
3. Packed gate weight modified in place between forward/backward: both versions complete and produce bitwise-equal input gradients to each other.

Case 3 confirms **unchanged behavior**, not correctness or authorization of dynamic weight mutation: neither version gains a weight snapshot or mutation guard. The existing static/frozen-weight contract remains necessary. General concurrent mutations and arbitrary tensor hooks were not tested or brought into scope.

## Target-runtime ownership check

A control constructed `GradientEdge(full.node, full.output_nr)` without the installed API's ownership token, deleted `full` and the output, then invoked grad. It failed on the installed runtime with:

> RuntimeError: Attribute '_input_metadata' is invalid for this instance of _C._FunctionBase.

This makes a node-only edge insufficient for the demonstrated Python `MatMul4BitBackward` path. It does **not** establish the behavior of every older PyTorch release. Before accepting r3 on T4, confirm that its real `get_gradient_edge` has equivalent graph ownership or observe successful real-NF4 execution after the output deletions. If the target exposes node-only behavior and fails, preserving an explicit C++ graph owner is required before rerunning the gates; retaining just the Python grad_fn is not a proven remedy.

## Remaining acceptance evidence

Run the already queued T4 component/prepared gates on the exact r3 candidate. Require unchanged bitwise/topology/RNG results, actual peak nonincrease at N=2048 (and all required sizes), and the existing latency criteria. Releasing these two storages addresses the intended overlap, but a different point in backward can still determine the global peak. No additional GPU jobs or changes were initiated by this review.
