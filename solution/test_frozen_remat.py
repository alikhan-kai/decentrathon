"""Frozen projection rematerialization contracts, without installing the solution.

Shared helpers live in test_native_half to keep the portable contribution at two
modules. Private CPU entry probes bypass admission only here. CUDA execution is
not part of these tests; real starter arithmetic requires the CPU interpreter.
"""
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from . import frozen_remat, hybrid_mlp, native_half
from .test_native_half import (
    GRAPHS, _FrozenProjection, assert_exact, cpu_starter, cpu_state, graph_output,
    interpreter_starter, make_mlp, reference_mlp,
)


def remat_mlp(module, hidden, starter):
    return module.down_proj(frozen_remat.frozen_branch_remat(module, hidden, starter)).float()


@pytest.mark.parametrize('graph', GRAPHS)
@pytest.mark.parametrize('seed', (0, 1))
@pytest.mark.parametrize('kind', ('torch', 'nf4'))
def test_remat_final_output_and_fanout_order(cpu_state, cpu_starter, graph, seed, kind):
    torch.manual_seed(seed)
    module, width = make_mlp(kind)
    source, cotangent = torch.randn(1, 7, width), torch.randn(1, 7, width)
    observations = []
    for forward in (reference_mlp, remat_mlp):
        hidden = source.clone().requires_grad_(True)
        out = graph_output(lambda x: forward(module, x, cpu_starter), hidden, graph)
        grad, = torch.autograd.grad(out, hidden, cotangent)
        assert out.dtype == grad.dtype == torch.float32
        observations.append((out.detach(), grad))
    for actual, expected in zip(observations[1], observations[0]):
        assert_exact(actual, expected)


@pytest.mark.parametrize('rows', (1024, 1025))
def test_hybrid_threshold_outputs_gradients_and_replay(cpu_state, cpu_starter, rows):
    torch.manual_seed(83)
    module, width = make_mlp()
    source, cotangent = torch.randn(1, rows, width), torch.randn(1, rows, width)
    observations = []
    for forward in (reference_mlp, hybrid_mlp.hybrid_mlp_forward):
        for projection in (module.gate_proj, module.up_proj):
            projection.autocast_calls.clear()
        hidden = source.clone().requires_grad_(True)
        out = graph_output(lambda x: forward(module, x, cpu_starter), hidden, 'weighted_residual')
        grad, = torch.autograd.grad(out, hidden, cotangent)
        assert out.dtype == grad.dtype == torch.float32
        observations.append((out.detach(), grad))
    assert len(module.gate_proj.autocast_calls) == (4 if rows > 1024 else 2)
    for actual, expected in zip(observations[1], observations[0]):
        assert_exact(actual, expected)


@pytest.mark.parametrize('forward', (native_half._validated_entry, remat_mlp), ids=('native', 'remat'))
def test_separate_half_cast_boundaries_produce_2049(cpu_state, cpu_starter, forward):
    """Actual MLP backward: gate contributes 2048, up contributes 1 to FP32 x."""
    class Affine(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float16))
            self.register_buffer('bias', torch.tensor(bias, dtype=torch.float16))

        def forward(self, x):
            return torch.nn.functional.linear(x.half(), self.weight, self.bias).to(x.dtype)

    module = SimpleNamespace(
        gate_proj=Affine([[1.0], [0.0]], [0.0, 16.0]),
        up_proj=Affine([[0.0], [1.0 / 16.0]], [4096.0, 0.0]),
        down_proj=lambda x: x.sum(-1, keepdim=True).float(), intermediate_size=2)
    hidden = torch.zeros(1, 1, 1, requires_grad=True)
    out = forward(module, hidden, cpu_starter)
    grad, = torch.autograd.grad(out.sum(), hidden)
    assert out.dtype == grad.dtype == torch.float32
    assert grad.item() == 2049
    # Negative control: sharing the half input rounds 2048 + 1 back to 2048.
    shared = hidden.half()
    wrong_out = module.down_proj(cpu_starter.swiglu(
        module.gate_proj(shared).float(), module.up_proj(shared).float()))
    wrong, = torch.autograd.grad(wrong_out.sum(), hidden)
    assert wrong.item() == 2048


@pytest.mark.parametrize('mode', ('repeat', 'accumulate'))
@pytest.mark.parametrize('amp', (False, True))
def test_rng_autocast_and_repeated_backward(cpu_state, cpu_starter, mode, amp):
    torch.manual_seed(123)
    # Exercise FP16 autocast, matching the admitted production precision.
    # CPU BF16 adds a different rounding boundary and is not that contract.
    module, width = make_mlp('nf4')
    module.gate_proj = _FrozenProjection(width, module.intermediate_size, dropout=0.25)
    module.up_proj = _FrozenProjection(width, module.intermediate_size, dropout=0.25)
    source, cotangent = torch.randn(1, 7, width), torch.randn(1, 7, width)
    observations = []
    for forward in (reference_mlp, remat_mlp):
        torch.manual_seed(912)
        module.gate_proj.autocast_calls.clear()
        hidden = source.clone().requires_grad_(True)
        def run():
            with torch.autocast('cpu', enabled=amp, dtype=torch.float16):
                return graph_output(lambda x: forward(module, x, cpu_starter), hidden, 'weighted_residual')
        out = run()
        torch.rand(17)  # Replay must preserve intervening caller RNG draws.
        before_backward = torch.get_rng_state().clone()
        out.backward(cotangent, retain_graph=mode == 'repeat')
        assert torch.equal(torch.get_rng_state(), before_backward)
        first_grad = hidden.grad.clone()
        second = out if mode == 'repeat' else run()
        before_backward = torch.get_rng_state().clone()
        second.backward(cotangent)
        assert torch.equal(torch.get_rng_state(), before_backward)
        if mode == 'repeat':
            assert_exact(hidden.grad, first_grad + first_grad)
        assert all(enabled is amp and dtype == torch.float16
                   for enabled, dtype in module.gate_proj.autocast_calls)
        assert out.dtype == hidden.grad.dtype == torch.float32
        observations.append((out.detach(), hidden.grad.clone(), torch.get_rng_state()))
    for actual, expected in zip(observations[1], observations[0]):
        assert_exact(actual, expected)


def test_only_hidden_saved_then_released(cpu_state, cpu_starter):
    module, width = make_mlp()
    hidden = torch.randn(1, 7, width, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        out = frozen_remat.frozen_branch_remat(module, hidden, cpu_starter)
    assert len(saved) == 1 and saved[0] is hidden
    saved.clear()
    node = out.grad_fn
    out.sum().backward(retain_graph=True)
    assert node.saved_tensors[0].data_ptr() == hidden.data_ptr()
    out.sum().backward()
    with pytest.raises(RuntimeError, match='backward through the graph a second time|saved tensors'):
        _ = node.saved_tensors


@pytest.mark.parametrize('kind', ('torch', 'nf4'))
def test_gradient_edges_do_not_own_projection_outputs(cpu_state, cpu_starter, monkeypatch, kind):
    """Observe real nested autograd; instrumentation does not replace gradients."""
    module, width = make_mlp(kind)
    hidden = torch.randn(1, 7, width, requires_grad=True)
    refs, seen = {}, []
    kernel = cpu_starter.example_swiglu_bwd_kernel
    original_grad = torch.autograd.grad

    class ObserveKernel:
        def __getitem__(self, grid):
            def launch(grad, gate, up, dg, du, n, **kwargs):
                refs.update(gate=weakref.ref(gate), up=weakref.ref(up),
                            dg=weakref.ref(dg), du=weakref.ref(du))
                return kernel[grid](grad, gate, up, dg, du, n, **kwargs)
            return launch

    def observe_grad(edge, leaf, cotangent, **kwargs):
        if not isinstance(edge, torch.autograd.graph.GradientEdge):
            return original_grad(edge, leaf, cotangent, **kwargs)
        assert leaf.is_leaf and leaf.dtype == torch.float32
        assert cotangent.dtype == torch.float16
        assert refs['gate']() is None and refs['up']() is None
        if seen:
            assert refs['du']() is None and refs['dg']() is cotangent
        else:
            assert refs['du']() is cotangent
        result = original_grad(edge, leaf, cotangent, **kwargs)
        assert result[0].dtype == torch.float32
        seen.append(True)
        return result

    out = frozen_remat.frozen_branch_remat(module, hidden, cpu_starter)
    with monkeypatch.context() as context:
        context.setattr(cpu_starter, 'example_swiglu_bwd_kernel', ObserveKernel())
        context.setattr(torch.autograd, 'grad', observe_grad)
        out.sum().backward()
    gc.collect()
    assert len(seen) == 2
    assert all(ref() is None for ref in refs.values())


def test_remat_saved_version_and_first_order_guards(cpu_state, cpu_starter):
    module, width = make_mlp()
    for higher_order in (False, True):
        hidden = torch.randn(1, 7, width, requires_grad=True)
        out = frozen_remat.frozen_branch_remat(module, hidden, cpu_starter)
        if not higher_order:
            with torch.no_grad():
                hidden.add_(1)
        with pytest.raises(RuntimeError, match='first-order' if higher_order else 'modified by an inplace'):
            torch.autograd.grad(out.sum(), hidden, create_graph=higher_order)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(hidden, torch.ones_like(hidden))
        with pytest.raises(RuntimeError, match='forward AD'):
            frozen_remat.frozen_branch_remat(module, dual, cpu_starter)
    with torch.no_grad():
        assert frozen_remat._FrozenBranchRemat.backward(
            SimpleNamespace(needs_input_grad=(True, True)), None) == (None, None, None, None)


@pytest.mark.parametrize('scale', (1.0, 65536.0, 1.0 / 65536.0))
@pytest.mark.filterwarnings('ignore:.*:RuntimeWarning:triton.runtime.interpreter')
def test_interpreter_nf4_noncontiguous_cotangent(cpu_state, interpreter_starter, scale):
    torch.manual_seed(33)
    module, width = make_mlp('nf4')
    source = torch.randn(1, 7, width)
    cotangent = torch.randn(width, 7).t().unsqueeze(0) * scale
    assert not cotangent.is_contiguous()
    observations = []
    for forward in (reference_mlp, native_half._validated_entry, remat_mlp):
        hidden = source.clone().requires_grad_(True)
        out = graph_output(lambda x: forward(module, x, interpreter_starter), hidden, 'weighted_residual')
        grad, = torch.autograd.grad(out, hidden, cotangent)
        assert out.dtype == grad.dtype == torch.float32
        observations.append((out.detach(), grad))
    for observation in observations[1:]:
        for actual, expected in zip(observation, observations[0]):
            assert_exact(actual, expected)


def test_reversed_fanout_negative_control(cpu_state, cpu_starter, monkeypatch):
    """Swapping the two returned FP32 edges must change a real fanout gradient."""
    torch.manual_seed(1)
    module, width = make_mlp('nf4')
    source, cotangent = torch.randn(1, 7, width), torch.randn(1, 7, width)

    def gradient(forward):
        hidden = source.clone().requires_grad_(True)
        out = graph_output(lambda x: forward(module, x, cpu_starter), hidden, 'weighted_residual')
        return torch.autograd.grad(out, hidden, cotangent)[0]

    expected = gradient(reference_mlp)
    assert_exact(gradient(remat_mlp), expected)
    original = frozen_remat._FrozenBranchRemat.backward

    def reversed_edges(ctx, grad):
        up, gate, module_grad, starter_grad = original(ctx, grad)
        return gate, up, module_grad, starter_grad

    with monkeypatch.context() as context:
        context.setattr(frozen_remat._FrozenBranchRemat, 'backward', staticmethod(reversed_edges))
        wrong = gradient(remat_mlp)
    with pytest.raises(AssertionError):
        assert_exact(wrong, expected)
