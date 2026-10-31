"""CPU regressions for the private, already-validated frozen MLP entry.

No installer or global solution patch is invoked. The CPU starter adapter uses
PyTorch autograd as an independent derivative oracle; the optional interpreter
cases execute the repository's unchanged Triton starter. Neither admits CPU
inputs through production guards nor constitutes CUDA validation.
"""
import builtins
import inspect
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from . import bnb_provenance, native_half


def assert_exact(actual, expected):
    """Finite bytes (including signed zero); NaN classes and infinity signs."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    for classify in (torch.isfinite, torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(classify(actual), classify(expected))
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].contiguous().view(torch.uint8),
                       expected[finite].contiguous().view(torch.uint8))


def _expression(gate, up):
    return gate * torch.sigmoid(gate) * up


class _CPUKernel:
    """Only the starter launch protocol is adapted; derivatives use autograd."""
    def __init__(self, backward=False):
        self.backward = backward

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self.backward:
                grad, gate, up, dgate, dup, _ = args
                with torch.enable_grad():
                    g = gate.detach().float().requires_grad_(True)
                    u = up.detach().float().requires_grad_(True)
                    dg, du = torch.autograd.grad(_expression(g, u), (g, u), grad.float())
                dgate.copy_(dg)
                dup.copy_(du)
            else:
                gate, up, out, _ = args
                out.copy_(_expression(gate.float(), up.float()))
        return launch


class _CPUForward:
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return _expression(gate, up)


class _CPUStarter:
    SwiGLU = _CPUForward
    BLOCK = 1024
    triton = SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)
    example_swiglu_fwd_kernel = _CPUKernel()
    example_swiglu_bwd_kernel = _CPUKernel(backward=True)

    @staticmethod
    def swiglu(gate, up):
        return _expression(gate, up)


class _FrozenProjection(torch.nn.Module):
    def __init__(self, inputs, outputs, dropout=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(outputs, inputs).half() * 0.1,
                                         requires_grad=False)
        self.dropout = dropout
        self.autocast_calls = []

    def forward(self, x):
        self.autocast_calls.append((torch.is_autocast_enabled('cpu'),
                                   torch.get_autocast_dtype('cpu')))
        x16 = F.dropout(x.half(), self.dropout, training=True)
        return F.linear(x16, self.weight).to(x.dtype)


def make_mlp(kind='torch', dropout=0.0):
    """Small real linear projections, optionally real nested NF4 on CPU."""
    hidden, intermediate = (8, 16) if kind == 'torch' else (64, 128)

    def projection(inputs, outputs):
        if kind == 'torch':
            return _FrozenProjection(inputs, outputs, dropout)
        bnb = pytest.importorskip('bitsandbytes')
        layer = bnb.nn.Linear4bit(inputs, outputs, bias=False,
                                 compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type='nf4')
        layer.weight = bnb.nn.Params4bit(
            torch.randn(outputs, inputs).half() * 0.1, requires_grad=False,
            compress_statistics=True, quant_type='nf4', blocksize=64,
            module=layer).to('cpu')
        assert layer.weight.quant_state.nested
        # Production admission requires initialized, frozen FP16 compute.
        # Otherwise the first FP32 reference call silently selects FP32 compute.
        layer.compute_type_is_set = True
        assert layer.compute_dtype == torch.float16
        return layer

    return SimpleNamespace(gate_proj=projection(hidden, intermediate),
                           up_proj=projection(hidden, intermediate),
                           down_proj=projection(intermediate, hidden),
                           intermediate_size=intermediate), hidden


def reference_mlp(module, hidden, starter):
    """Explicit original FP32 starter graph, with independent half boundaries."""
    gate = module.gate_proj(hidden)
    up = module.up_proj(hidden)
    return module.down_proj(starter.swiglu(gate.float(), up.float()))


def graph_output(mlp, hidden, graph):
    if graph == 'isolated':
        return mlp(hidden)
    if graph == 'residual':
        return mlp(hidden) + hidden
    if graph == 'shared':
        return mlp(hidden) + mlp(hidden)
    if graph == 'weighted':
        return mlp(hidden) + 0.3 * mlp(hidden)
    if graph == 'weighted_residual':
        return hidden + mlp(hidden) + 0.3 * mlp(hidden)
    assert graph == 'norm_residual'
    normalized = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    return hidden + mlp(normalized)


@pytest.fixture
def cpu_state(monkeypatch):
    """Isolate RNG/thread state; replace only CUDA RNG plumbing for CPU replay."""
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    initialized = torch.cuda.is_initialized()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *a, **k: torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda *a, **k: None)
    try:
        yield
    finally:
        torch.set_rng_state(state)
        torch.set_num_threads(threads)
        assert torch.cuda.is_initialized() == initialized


@pytest.fixture
def cpu_starter():
    return _CPUStarter()


@pytest.fixture
def interpreter_starter():
    if os.environ.get('TRITON_INTERPRET') != '1':
        pytest.skip('CPU interpreter only: start a fresh process with TRITON_INTERPRET=1; no CUDA launch')
    pytest.importorskip('triton')
    from . import swiglu
    # An earlier import in a non-interpreter process must never launch on CUDA.
    if type(swiglu.example_swiglu_fwd_kernel).__name__ != 'InterpretedFunction':
        pytest.skip('starter was imported before interpreter mode was enabled')
    return swiglu


GRAPHS = ('isolated', 'residual', 'shared', 'weighted', 'weighted_residual', 'norm_residual')


@pytest.mark.parametrize('graph', GRAPHS)
@pytest.mark.parametrize('seed', (0, 1))
@pytest.mark.parametrize('kind', ('torch', 'nf4'))
def test_native_final_output_and_fanout_gradients(cpu_state, cpu_starter, graph, seed, kind):
    torch.manual_seed(seed)
    module, width = make_mlp(kind)
    source, cotangent = torch.randn(1, 7, width), torch.randn(1, 7, width)
    observations = []
    for forward in (reference_mlp, native_half._validated_entry):
        hidden = source.clone().requires_grad_(True)
        out = graph_output(lambda x: forward(module, x, cpu_starter), hidden, graph)
        grad, = torch.autograd.grad(out, hidden, cotangent)
        assert out.dtype == grad.dtype == torch.float32
        observations.append((out.detach(), grad))
    for actual, expected in zip(observations[1], observations[0]):
        assert_exact(actual, expected)


@pytest.mark.parametrize('operand', (0, 1))
def test_native_save_identity_release_and_mutation(cpu_state, cpu_starter, operand):
    values = [torch.randn(7, dtype=torch.float16, requires_grad=True) for _ in range(2)]
    out = native_half._NativeHalfSwiGLU.apply(*values, cpu_starter)
    node = out.grad_fn
    assert all(saved is original for saved, original in zip(node.saved_tensors, values))
    out.sum().backward(retain_graph=True)
    assert all(saved is original for saved, original in zip(node.saved_tensors, values))
    out.sum().backward()
    with pytest.raises(RuntimeError, match='backward through the graph a second time|saved tensors'):
        _ = node.saved_tensors
    out = native_half._NativeHalfSwiGLU.apply(*values, cpu_starter)
    with torch.no_grad():
        values[operand].add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace'):
        out.sum().backward()


def test_native_rejects_higher_order(cpu_state, cpu_starter):
    gate, up = [torch.randn(7, dtype=torch.float16, requires_grad=True) for _ in range(2)]
    out = native_half._NativeHalfSwiGLU.apply(gate, up, cpu_starter)
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(out.sum(), gate, create_graph=True)


@pytest.mark.parametrize('operand', (0, 1))
@pytest.mark.parametrize('scale', (1.0, 65536.0, 1.0 / 65536.0))
@pytest.mark.filterwarnings('ignore:.*:RuntimeWarning:triton.runtime.interpreter')
def test_interpreter_all_half_encodings(cpu_state, interpreter_starter, pinned_bnb, operand, scale):
    """Exhaust every FP16 encoding across the SAME canonical BNB down boundary."""
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    gate = encodings if operand == 0 else torch.linspace(-10, 10, 65536).half()
    up = encodings if operand == 1 else torch.linspace(-3, 3, 65536).half()
    torch.manual_seed(982)
    module, width = make_mlp('nf4')
    gate, up = gate.reshape(1, 512, 128), up.reshape(1, 512, 128)
    cotangent = torch.linspace(-1, 1, 512 * width).reshape(1, 512, width) * scale
    observations = []
    for candidate in (False, True):
        g, u = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
        out = (native_half._NativeHalfSwiGLU.apply(g, u, interpreter_starter) if candidate
               else interpreter_starter.swiglu(g.float(), u.float()))
        # Canonical down casts FP32 inputs to half; the final MLP widens its output.
        # A half private output is valid; arbitrary FP32 private cotangents are not.
        out = module.down_proj(out).float()
        grads = torch.autograd.grad(out, (g, u), cotangent)
        observations.append((out.detach(), *grads))
    for actual, expected in zip(observations[1], observations[0]):
        assert_exact(actual, expected)


@pytest.fixture
def pinned_bnb():
    bnb = pytest.importorskip('bitsandbytes')
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13) or bnb.__version__ != '0.50.2':
        pytest.skip('positive executable provenance is pinned to CPython 3.13 / bitsandbytes 0.50.2')
    from bitsandbytes.autograd._functions import MatMul4Bit, matmul_4bit
    assert bnb_provenance.canonical_bnb(), 'run with canonical BNB; do not call solution.apply()'
    return bnb, MatMul4Bit, matmul_4bit


def test_provenance_needs_no_installed_source(pinned_bnb, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected package file/source access')
    with monkeypatch.context() as context:
        context.setattr(inspect, 'getsource', forbidden)
        context.setattr(builtins, 'open', forbidden)
        context.setattr(Path, 'read_bytes', forbidden)
        assert bnb_provenance.canonical_bnb()


@pytest.mark.parametrize('mutation', ('defaults', 'body', 'location'))
def test_provenance_executable_identity(pinned_bnb, mutation):
    _, _, original = pinned_bnb
    code, defaults = original.__code__, original.__defaults__
    if mutation == 'defaults':
        defaults = (1, None)
    elif mutation == 'body':
        code = code.replace(co_consts=code.co_consts + (123,))
    else:
        code = code.replace(co_filename='/unread/source.py', co_firstlineno=1)
    changed = FunctionType(code, original.__globals__, original.__name__, defaults)
    changed.__qualname__ = original.__qualname__
    assert bnb_provenance.matches('matmul_4bit', changed) is (mutation == 'location')


@pytest.mark.parametrize('target', ('matmul_forward', 'setup_context', 'linear_forward'))
def test_provenance_rejects_replaced_bnb(pinned_bnb, monkeypatch, target):
    bnb, matmul, _ = pinned_bnb
    if target == 'linear_forward':
        monkeypatch.setattr(bnb.nn.Linear4bit, 'forward', lambda *a: None)
    else:
        name = 'forward' if target == 'matmul_forward' else 'setup_context'
        monkeypatch.setattr(matmul, name, staticmethod(lambda *a: None), raising=False)
    assert not bnb_provenance.canonical_bnb()


@pytest.mark.parametrize('version', ('0.50.1', '0.50.3', 'unknown'))
def test_provenance_rejects_wrong_version(pinned_bnb, monkeypatch, version):
    bnb, _, _ = pinned_bnb
    monkeypatch.setattr(bnb, '__version__', version)
    assert not bnb_provenance.canonical_bnb()


def test_source_authentication_without_installer(pinned_bnb, tmp_path):
    from . import swiglu, swiglu_pack
    assert swiglu_pack._verified_starter(swiglu)
    assert native_half._validate_versions() is None
    changed = tmp_path / 'changed_starter.py'
    changed.write_bytes(Path(swiglu.__file__).read_bytes() + b'\n# changed source\n')
    impostor = SimpleNamespace(__file__=str(changed))
    assert not swiglu_pack._verified_starter(impostor)


def test_validate_versions_rejects_mutated_canonical_bnb(pinned_bnb, monkeypatch):
    """Exercise the production reason API after a real loaded-BNB mutation."""
    _, matmul, _ = pinned_bnb
    assert native_half._validate_versions() is None
    with monkeypatch.context() as context:
        context.setattr(matmul, 'forward', staticmethod(lambda *a: None))
        assert not bnb_provenance.canonical_bnb()
        assert native_half._validate_versions() == 'noncanonical-bnb-executable'
    assert native_half._validate_versions() is None


@pytest.mark.parametrize('path', ('native', 'hybrid'))
def test_missing_final_widen_negative_control(cpu_state, cpu_starter, pinned_bnb, path):
    from . import hybrid_mlp
    torch.manual_seed(717)
    module, width = make_mlp('nf4')
    hidden = torch.randn(1, 7, width, requires_grad=True)
    expected = reference_mlp(module, hidden, cpu_starter)
    actual = (native_half._validated_entry(module, hidden, cpu_starter) if path == 'native'
              else hybrid_mlp.hybrid_mlp_forward(module, hidden, cpu_starter, threshold=1))
    assert actual.dtype == expected.dtype == torch.float32
    assert_exact(actual, expected)
    delta = 2.0 ** -24
    assert_exact(actual + delta, expected + delta)
    # Removing only the final widening moves the following addition into FP16.
    missing_widen = actual.half()
    with pytest.raises(AssertionError):
        assert_exact((missing_widen + delta).float(), expected + delta)
