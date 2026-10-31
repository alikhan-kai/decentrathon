"""Authenticate already-imported pinned Python functions without reading package files.

The executable fingerprint is deliberately CPython-minor-specific. A mismatch
declines the optimization; it never weakens the frozen-BNB admission contract.
"""
import hashlib
import json
import sys
import types

EXPECTED = {'Linear4bit.forward': '06a7f4fecf2d1056317768d15dc11fddef60578b3925a040efe063b5b101a75a', 'MatMul4Bit.forward': 'bc6537aeb7cfe99c993b57d5646a7ba172c4b50f244384b84b05f6f62a627945', 'MatMul4Bit.backward': '66fcee158d72e76e81448a2e83206a9212281991c2209e653d096946902ff439', 'matmul_4bit': '90f8f5bef4688542c5a31c167595991af954d88d4374e79fca8529609f18532f'}


def _encode(value):
    if isinstance(value, types.CodeType):
        fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount',
                  'co_nlocals', 'co_stacksize', 'co_flags', 'co_code', 'co_consts',
                  'co_names', 'co_varnames', 'co_freevars', 'co_cellvars',
                  'co_name', 'co_qualname', 'co_exceptiontable')
        return ['code', [[name, _encode(getattr(value, name))] for name in fields]]
    if type(value) in (str, int, bool) or value is None:
        return [type(value).__name__, value]
    if type(value) is float:
        return ['float', value.hex()]
    if type(value) is complex:
        return ['complex', value.real.hex(), value.imag.hex()]
    if type(value) is bytes:
        return ['bytes', value.hex()]
    if type(value) is tuple:
        return ['tuple', [_encode(x) for x in value]]
    if type(value) is frozenset:
        return ['frozenset', sorted((_encode(x) for x in value), key=repr)]
    if value is Ellipsis:
        return ['ellipsis']
    raise TypeError('unsupported executable constant')


def fingerprint(fn):
    if type(fn) is not types.FunctionType or fn.__closure__ or fn.__kwdefaults__:
        raise TypeError('expected ordinary pinned function')
    payload = [_encode(fn.__code__), _encode(fn.__defaults__), fn.__module__, fn.__qualname__]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def matches(name, fn):
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13):
        return False
    try:
        return name in EXPECTED and fingerprint(fn) == EXPECTED[name]
    except (TypeError, AttributeError, ValueError):
        return False


def canonical_bnb():
    import bitsandbytes as bnb
    import torch
    from bitsandbytes.autograd._functions import MatMul4Bit, matmul_4bit
    if bnb.__version__ != '0.50.2':
        return False
    if MatMul4Bit.__bases__ != (torch.autograd.Function,):
        return False
    allowed = {'__module__', '__firstlineno__', 'forward', 'backward',
               '__static_attributes__', '__doc__', '_backward_cls'}
    if set(vars(MatMul4Bit)) != allowed:
        return False
    if MatMul4Bit.__static_attributes__ != ():
        return False
    if any(type(vars(MatMul4Bit)[n]) is not staticmethod for n in ('forward', 'backward')):
        return False
    objects = {'Linear4bit.forward': bnb.nn.Linear4bit.forward,
               'MatMul4Bit.forward': MatMul4Bit.forward,
               'MatMul4Bit.backward': MatMul4Bit.backward,
               'matmul_4bit': matmul_4bit}
    return all(matches(name, fn) for name, fn in objects.items())
