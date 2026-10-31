import ast
import builtins
import importlib.util
import inspect
import pathlib
import types
from unittest.mock import patch

ROOT = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location('proof', ROOT/'bnb_provenance.py')
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import MatMul4Bit, matmul_4bit


def test_no_package_source_reads():
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected file/source access')
    with patch.object(inspect, 'getsource', forbidden), patch.object(builtins, 'open', forbidden), patch.object(pathlib.Path, 'read_bytes', forbidden):
        assert proof.canonical_bnb()


def test_body_and_default_mutations_rejected():
    original = matmul_4bit
    changed = types.FunctionType(original.__code__, original.__globals__, original.__name__, (1, None))
    changed.__qualname__ = original.__qualname__
    assert not proof.matches('matmul_4bit', changed)
    changed = types.FunctionType(original.__code__.replace(co_consts=original.__code__.co_consts + (123,)), original.__globals__, original.__name__, original.__defaults__)
    changed.__qualname__ = original.__qualname__
    assert not proof.matches('matmul_4bit', changed)


def test_file_location_is_irrelevant():
    original = matmul_4bit
    changed = types.FunctionType(original.__code__.replace(co_filename='/not/read.py', co_firstlineno=1), original.__globals__, original.__name__, original.__defaults__)
    changed.__qualname__ = original.__qualname__
    assert proof.matches('matmul_4bit', changed)


def test_replacement_body_class_rejected():
    with patch.object(MatMul4Bit, 'forward', staticmethod(lambda *a: None)):
        assert not proof.canonical_bnb()
    with patch.object(MatMul4Bit, 'setup_context', staticmethod(lambda *a: None), create=True):
        assert not proof.canonical_bnb()
    with patch.object(bnb.nn.Linear4bit, 'forward', lambda *a: None):
        assert not proof.canonical_bnb()

