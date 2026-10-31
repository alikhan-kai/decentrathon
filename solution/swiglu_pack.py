"""Experimental saved-activation storage only. Disabled; never add to registry.

Reuse the unchanged R8 starter forward/backward, including its kernel names and
casts. example_swiglu_* time remains INELIGIBLE for originality/kernel share.
Only install explicitly for fixed-task, first-order, CUDA fp16 training.
"""
import ast
import hashlib
import importlib
import inspect
from pathlib import Path

import torch

ENABLED = False
STARTER_SHA256 = "77bd4f5aca08a5cd601724f18abc5a46372d851014ddc909edfd2cbd8176cf2e"
CANDIDATE_STARTER_SHA256 = "e1dc639838c4c46ea0ed0aec3f0d904d360c92a91c601b943c9cfef85ca21c64"


def _verified_starter(starter):
    """Only TWO exact known sources. AST normalization does not bypass hashes."""
    source = Path(starter.__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest not in (STARTER_SHA256, CANDIDATE_STARTER_SHA256):
        return False
    tree = ast.parse(source)
    expected = ast.parse('@triton.jit(do_not_specialize=["n_elements"])\ndef f(): pass').body[0].decorator_list[0]
    plain = ast.parse('@triton.jit\ndef f(): pass').body[0].decorator_list[0]
    changes = 0
    for node in tree.body:
        if getattr(node, "name", "") in ("example_swiglu_fwd_kernel", "example_swiglu_bwd_kernel"):
            if len(node.decorator_list) == 1 and ast.dump(node.decorator_list[0]) == ast.dump(expected):
                node.decorator_list = [plain]
                changes += 1
    if changes != (2 if digest == CANDIDATE_STARTER_SHA256 else 0):
        return False
    # Parse both trees with this interpreter: AST field sets differ by Python minor.
    canonical = source.replace(b'@triton.jit(do_not_specialize=["n_elements"])', b'@triton.jit')
    if hashlib.sha256(canonical).hexdigest() != STARTER_SHA256:
        return False
    return ast.dump(tree) == ast.dump(ast.parse(canonical))


# Above 1024 rows, recompute the gate/up branch in only as many layers as the memory budget needs,
# instead of in all of them. Saved bytes per token and layer (fp32 residual stream, Qwen2.5-1.5B),
# from the single runs in docs/overnight/CANDIDATE.md (native-all 5261 MiB vs 1024-hybrid 4521 MiB,
# longest step 1427 rows): ~68.8 KB with native fp16 gate/up saves, ~39 KB with the branch recomputed
# (it keeps only the fp32 input). The budget is BUDGET_ROWS fully native rows, below the 1024 rows the
# all-native path already allows, so a step above 1024 rows never saves more than one at 1024 did.
# Fixed constants: no step, seed or dataset lookup, nothing measured at runtime.
LAYER_BUDGET = True
NATIVE_KB, REMAT_KB, BUDGET_ROWS = 68.8, 36.0, 780


def native_layers(n, layers):
    """Number of layers (lowest indices first) that keep native saves for an n-row input above 1024."""
    k = int(layers * (BUDGET_ROWS * NATIVE_KB - n * REMAT_KB) / (n * (NATIVE_KB - REMAT_KB)))
    return max(0, min(layers, k))


def _task_input_shape(x):
    """Metadata only: original B4/T512 or packed B1/N2048, both H1536."""
    return (x.layout == torch.strided and x.ndim == 3 and x.shape[2] == 1536
            and ((x.shape[0] == 4 and 1 <= x.shape[1] <= 512)
                 or (x.shape[0] == 1 and 1 <= x.shape[1] <= 2048)))


_active = None


class _PackSaves:
    def __init__(self, ctx):
        self.ctx = ctx

    def save_for_backward(self, gate, up):
        # Private, reached only after provenance checks in installed forward.
        # No tensors are attached to ctx except these two packed saves.
        assert gate.dtype == up.dtype == torch.float32
        self.ctx.save_for_backward(gate.to(torch.float16), up.to(torch.float16))


class _PackedSwiGLU(torch.autograd.Function):
    """Private primitive; NOT an API for packing arbitrary fp32 inputs."""

    @staticmethod
    def forward(ctx, gate, up, starter):
        ctx.starter = starter
        # Execute exactly the existing forward, intercepting only its saves.
        return starter.SwiGLU.forward(_PackSaves(ctx), gate, up)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            # create_graph is only known at backward, too late to redo forward.
            raise RuntimeError("SwiGLU packing is first-order only; remove installer before forward for higher-order work")
        gate16, up16 = ctx.saved_tensors
        # The unchanged starter kernel already promotes every load to FP32.
        # Feed its packed saves directly, preserving FP32 incoming/output grads.
        grad = grad.to(torch.float32).contiguous()
        dgate = torch.empty(gate16.shape, device=gate16.device, dtype=torch.float32)
        dup = torch.empty(up16.shape, device=up16.device, dtype=torch.float32)
        n = gate16.numel()
        starter = ctx.starter
        starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n, starter.BLOCK),)](
            grad, gate16, up16, dgate, dup, n, BLOCK=starter.BLOCK)
        return dgate, dup, None


class _FusedQwenMLP(torch.autograd.Function):
    """Fused Qwen2MLP for validated frozen nested-NF4 layers (native or remat)."""

    @staticmethod
    def forward(ctx, x, meta, remat):
        (w_gate_q, qs_gate, off_gate,
         w_up_q, qs_up, off_up,
         w_down_q, qs_down, off_down,
         starter, nf4_dequant) = meta
        ctx.set_materialize_grads(False)
        ctx.meta = meta
        ctx.remat = remat
        ctx.x_shape = x.shape
        m = x.numel() // 1536
        ctx.m = m

        x16 = x.detach().to(torch.float16).reshape(m, 1536)
        w_buf = torch.empty((8960, 1536), dtype=torch.float16, device=x.device)

        s2_g = qs_gate.state2
        nf4_dequant(w_gate_q, qs_gate.absmax, s2_g.code, s2_g.absmax, off_gate, w_buf)
        gate = torch.mm(x16, w_buf.t())

        s2_u = qs_up.state2
        nf4_dequant(w_up_q, qs_up.absmax, s2_u.code, s2_u.absmax, off_up, w_buf)
        up = torch.mm(x16, w_buf.t())

        swiglu_out = torch.empty((m, 8960), dtype=torch.float16, device=x.device)
        n_elem = m * 8960
        starter.example_swiglu_fwd_kernel[(starter.triton.cdiv(n_elem, starter.BLOCK),)](
            gate, up, swiglu_out, n_elem, BLOCK=starter.BLOCK
        )

        if remat:
            del gate, up
            ctx.save_for_backward(x16)
        else:
            ctx.save_for_backward(gate, up)
        del x16

        w_down = w_buf.view(1536, 8960)
        s2_d = qs_down.state2
        nf4_dequant(w_down_q, qs_down.absmax, s2_d.code, s2_d.absmax, off_down, w_down)
        out16 = torch.mm(swiglu_out, w_down.t())
        del swiglu_out, w_down, w_buf
        return out16.to(torch.float32).view(ctx.x_shape)

    @staticmethod
    def backward(ctx, grad_out):
        if torch.is_grad_enabled():
            raise RuntimeError("SwiGLU packing is first-order only; remove installer before forward for higher-order work")
        if grad_out is None or not ctx.needs_input_grad[0]:
            return None, None, None
        (w_gate_q, qs_gate, off_gate,
         w_up_q, qs_up, off_up,
         w_down_q, qs_down, off_down,
         starter, nf4_dequant) = ctx.meta
        m = ctx.m
        n_elem = m * 8960
        g16 = grad_out.reshape(m, 1536).to(dtype=torch.float16, memory_format=torch.contiguous_format)
        w_buf = torch.empty((8960, 1536), dtype=torch.float16, device=g16.device)

        w_down = w_buf.view(1536, 8960)
        s2_d = qs_down.state2
        nf4_dequant(w_down_q, qs_down.absmax, s2_d.code, s2_d.absmax, off_down, w_down)
        d_swiglu = torch.mm(g16, w_down)
        del g16, w_down

        dgate16 = torch.empty((m, 8960), dtype=torch.float16, device=d_swiglu.device)
        dup16 = torch.empty((m, 8960), dtype=torch.float16, device=d_swiglu.device)

        if ctx.remat:
            (x16,) = ctx.saved_tensors
            s2_g = qs_gate.state2
            nf4_dequant(w_gate_q, qs_gate.absmax, s2_g.code, s2_g.absmax, off_gate, w_buf)
            gate = torch.mm(x16, w_buf.t())

            s2_u = qs_up.state2
            nf4_dequant(w_up_q, qs_up.absmax, s2_u.code, s2_u.absmax, off_up, w_buf)
            up = torch.mm(x16, w_buf.t())

            starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n_elem, starter.BLOCK),)](
                d_swiglu, gate, up, dgate16, dup16, n_elem, BLOCK=starter.BLOCK
            )
            del d_swiglu, gate, up

            dx = torch.mm(dup16, w_buf).to(torch.float32)
            del dup16

            nf4_dequant(w_gate_q, qs_gate.absmax, s2_g.code, s2_g.absmax, off_gate, w_buf)
            dx.add_(torch.mm(dgate16, w_buf))
            del dgate16, w_buf
        else:
            gate, up = ctx.saved_tensors
            starter.example_swiglu_bwd_kernel[(starter.triton.cdiv(n_elem, starter.BLOCK),)](
                d_swiglu, gate, up, dgate16, dup16, n_elem, BLOCK=starter.BLOCK
            )
            del d_swiglu, gate, up

            s2_u = qs_up.state2
            nf4_dequant(w_up_q, qs_up.absmax, s2_u.code, s2_u.absmax, off_up, w_buf)
            dx = torch.mm(dup16, w_buf).to(torch.float32)
            del dup16

            s2_g = qs_gate.state2
            nf4_dequant(w_gate_q, qs_gate.absmax, s2_g.code, s2_g.absmax, off_gate, w_buf)
            dx.add_(torch.mm(dgate16, w_buf))
            del dgate16, w_buf

        return dx.view(ctx.x_shape), None, None

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError("SwiGLU packing does not support forward AD")


def _no_mod_hooks(mod):
    return not (mod._forward_hooks or mod._forward_pre_hooks
                or mod._backward_hooks or mod._backward_pre_hooks)


def _build_fused_meta(mlp, dev, starter):
    from . import nf4
    if not getattr(nf4, "_patched", False):
        return None
    items = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(mlp, name)
        w = mod.weight
        qs = w.quant_state
        if not ((w.ndim == 1 or (w.ndim == 2 and (w.shape[0] == 1 or w.shape[1] == 1)))
                and w.is_contiguous() and not w.requires_grad
                and nf4._eligible_nested(w, qs, None, require_fp16_state=False)):
            return None
        off = nf4._get_offset_cuda(qs, dev)
        if off is None:
            return None
        items.extend((w, qs, off))
    return (*items, starter, nf4.dequantize_nf4_nested)


def _hooked(module):
    return any(value for key, value in vars(module).items()
               if key.endswith("hooks") or key.endswith("hooks_with_kwargs")
               or key.endswith("hooks_always_called"))


def _global_hooks():
    import torch.nn.modules.module as mm
    return any(value for key, value in vars(mm).items()
               if key.startswith("_global_") and "hook" in key)


class SavedActivationPatch:
    def __init__(self, enabled=False, reason="disabled"):
        self.enabled, self.reason = enabled, reason
        self.packed_calls = 0
        self.remat_calls = 0
        self.fallback_calls = {}
        self.layer_ids = {}

    def remove(self):
        global _active
        if self.enabled:
            if self.mlp_cls.forward is not self.forward:
                raise RuntimeError("MLP forward changed after installation; refusing to overwrite it")
            self.mlp_cls.forward = self.baseline
            self.enabled = False
            _active = None

    def module_reason(self, mlp):
        """Static provenance rejection reason, also usable for diagnostics."""
        if type(mlp) is not self.mlp_cls:
            return "mlp-subclass-or-adapter"
        if "forward" in vars(mlp) or self.mlp_cls.forward is not self.forward:
            return "custom-mlp-forward"
        if _global_hooks():
            return "global-hooks"
        if set(mlp._modules) != {"gate_proj", "up_proj", "down_proj", "act_fn"}:
            return "extra-mlp-children"
        if _hooked(mlp) or _hooked(mlp.act_fn):
            return "mlp-hooks"
        if mlp._parameters or mlp._buffers:
            return "extra-mlp-state"
        if not mlp.training or mlp.config.hidden_act != "silu" or type(mlp.act_fn) is not self.act_cls:
            return "nontraining-or-activation"
        if mlp.hidden_size != 1536 or mlp.intermediate_size != 8960:
            return "non-task-dimensions"
        # Check every projection, so no MLP adapter, trainable projection, custom
        # forward, bias, or hook can participate in the packed branch.
        for name, shape in (("gate_proj", (1536, 8960)), ("up_proj", (1536, 8960)),
                            ("down_proj", (8960, 1536))):
            mod = getattr(mlp, name)
            if type(mod) is not self.linear_cls:
                return name + ":noncanonical-linear"
            if "forward" in vars(mod) or type(mod).forward is not self.linear_forward:
                return name + ":custom-forward"
            if getattr(mod, "_compiled_call_impl", None) is not None or _hooked(mod):
                return name + ":hooks-or-compiled"
            if mod._modules or mod._buffers or set(mod._parameters) != {"weight", "bias"}:
                return name + ":extra-state"
            if any(k.startswith(("lora_", "adapter", "ia3_")) for k in vars(mod)):
                return name + ":adapter-state"
            if mod.bias is not None or mod.weight.requires_grad or not mod.training:
                return name + ":bias-trainable-or-eval"
            if type(mod.weight) is not self.params_cls or (mod.in_features, mod.out_features) != shape:
                return name + ":weight-or-shape"
            # compute_type_is_set=False would let bnb choose fp32 from fp32 x.
            if mod.compute_dtype != torch.float16 or not mod.compute_type_is_set:
                return name + ":compute-dtype"
        return None

    def reason_for(self, mlp, x):
        reason = self.module_reason(mlp)
        if reason:
            return reason
        if type(x) is not torch.Tensor or x.device.type != "cuda" or x.dtype != torch.float32:
            return "requires-plain-cuda-fp32-input"
        if (not torch.is_grad_enabled() or not x.requires_grad or torch.is_inference_mode_enabled()
                or torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active()):
            return "requires-eager-first-order-training"
        if getattr(mlp, "_compiled_call_impl", None) is not None:
            return "compiled-mlp"
        if not _task_input_shape(x):
            return "non-task-input-shape"
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            return "requires-fp16-autocast"
        # Canonical bnb outputs are computed at fp16 for both fused GEMM and
        # dequant+F.linear paths, then expanded to fp32 by Linear4bit.forward.
        for name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(mlp, name)
            qs = mod.weight.quant_state
            if (mod.weight.device != x.device or mod.weight.dtype != torch.uint8
                    or not mod.weight.bnb_quantized or qs is None
                    or qs.quant_type != "nf4" or qs.blocksize != 64
                    or tuple(qs.shape) != (mod.out_features, mod.in_features)):
                return name + ":non-task-quant-state"
        return None


def install_saved_activation_packing(*, enable=False, first_order_training=False, starter_module=None):
    """Call AFTER solution.apply(); supports installation before model creation.

    Defaults do nothing. Explicit first_order_training is a caller contract:
    ordinary GradScaler backward only, no create_graph/JVP/vmap/compile.
    Unsupported individual MLP calls use the captured current-team forward.
    No hooks, custom linear forwards, registry edits, or tensor-content scans.
    """
    global _active
    if not enable:
        return SavedActivationPatch(reason="disabled")
    if not first_order_training:
        return SavedActivationPatch(reason="first-order-training-not-declared")
    if _active is not None:
        return _active
    import bitsandbytes as bnb
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from transformers.activations import SiLUActivation

    if bnb.__version__ != "0.50.2" or transformers.__version__ != "5.16.1":
        return SavedActivationPatch(reason="unpinned-library-version")
    starter = starter_module or importlib.import_module("solution.swiglu")
    if not _verified_starter(starter):
        return SavedActivationPatch(reason="not-exact-r8-starter")
    baseline = Qwen2MLP.forward
    code = next(c for c in starter.patch_qwen2_mlp.__code__.co_consts
                if inspect.iscode(c) and c.co_name == "forward")
    if getattr(baseline, "__code__", None) is not code:
        return SavedActivationPatch(reason="current-team-starter-not-installed-or-custom-forward")

    from . import native_half
    bad = native_half._validate_versions()
    if bad:
        return SavedActivationPatch(reason=bad)

    import torch.nn.modules.module as mm
    from . import nf4 as nf4_mod

    handle = SavedActivationPatch(enabled=True, reason="explicit-install")
    handle.mlp_cls, handle.linear_cls, handle.params_cls = Qwen2MLP, bnb.nn.Linear4bit, bnb.nn.Params4bit
    handle.act_cls = SiLUActivation
    handle.linear_forward, handle.baseline = bnb.nn.Linear4bit.forward, baseline
    mlp_cache = {}

    def _fast_ready(mlp, x, entry):
        (mlp_ref, gate, up, down, act, w_g, w_u, w_d) = entry[:8]
        return (mlp is mlp_ref
                and type(x) is torch.Tensor and x.is_cuda and x.dtype == torch.float32
                and x.requires_grad and torch.is_grad_enabled()
                and not torch.is_inference_mode_enabled()
                and _task_input_shape(x)
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.float16
                and not torch.compiler.is_compiling()
                and not torch._C._are_functorch_transforms_active()
                and mlp.training and gate.training and up.training and down.training
                and handle.mlp_cls.forward is handle.forward
                and handle.linear_cls.forward is handle.linear_forward
                and "forward" not in mlp.__dict__
                and "forward" not in gate.__dict__
                and "forward" not in up.__dict__
                and "forward" not in down.__dict__
                and getattr(mlp, "_compiled_call_impl", None) is None
                and getattr(gate, "_compiled_call_impl", None) is None
                and getattr(up, "_compiled_call_impl", None) is None
                and getattr(down, "_compiled_call_impl", None) is None
                and mlp.gate_proj is gate and mlp.up_proj is up
                and mlp.down_proj is down and mlp.act_fn is act
                and mlp.config.hidden_act == "silu"
                and not (mm._global_forward_pre_hooks or mm._global_forward_hooks
                         or mm._global_backward_pre_hooks or mm._global_backward_hooks)
                and _no_mod_hooks(mlp) and _no_mod_hooks(act)
                and _no_mod_hooks(gate) and _no_mod_hooks(up) and _no_mod_hooks(down)
                and gate.weight is w_g and up.weight is w_u and down.weight is w_d
                and gate.bias is None and up.bias is None and down.bias is None
                and gate.compute_dtype == torch.float16
                and up.compute_dtype == torch.float16
                and down.compute_dtype == torch.float16
                and not (w_g.requires_grad or w_u.requires_grad or w_d.requires_grad)
                and w_g.device == x.device and w_u.device == x.device and w_d.device == x.device
                and getattr(nf4_mod, "_patched", False))

    def forward(self, x):
        entry = mlp_cache.get(id(self))
        if entry is None or not _fast_ready(self, x, entry):
            reason = handle.reason_for(self, x)
            if reason is not None:
                handle.fallback_calls[reason] = handle.fallback_calls.get(reason, 0) + 1
                return baseline(self, x)
            layers = self.config.num_hidden_layers
            layer = handle.layer_ids.setdefault(id(self), len(handle.layer_ids) % layers)
            meta = _build_fused_meta(self, x.device, starter)
            if meta is not None:
                entry = (self, self.gate_proj, self.up_proj, self.down_proj, self.act_fn,
                         self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight,
                         meta, layer, layers)
                mlp_cache[id(self)] = entry
        else:
            meta, layer, layers = entry[8], entry[9], entry[10]

        n = x.numel() // 1536
        remat = n > BUDGET_ROWS and not (LAYER_BUDGET and layer < native_layers(n, layers))
        if remat:
            handle.remat_calls += 1
        else:
            handle.packed_calls += 1
        if meta is not None:
            return _FusedQwenMLP.apply(x, meta, remat)
        if remat:
            from .hybrid_mlp import hybrid_mlp_forward
            return hybrid_mlp_forward(self, x, starter, threshold=BUDGET_ROWS)
        return native_half._validated_entry(self, x, starter)

    handle.forward = forward
    Qwen2MLP.forward = forward
    _active = handle
    return handle
