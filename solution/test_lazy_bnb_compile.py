"""CPU-only deferral contract tests; uses installed BNB source and real Torch.

Run in a fresh process: python -B -m unittest solution.test_lazy_bnb_compile -v
No dummy dependency modules, CUDA tensors, or solution.apply() calls.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import functools
import importlib.metadata
import inspect
import shutil
import sys
import json
from pathlib import Path
import threading
import time
import types
import unittest
from unittest.mock import patch

import torch
from solution import lazy_bnb_compile as lazy


def original_decorator():
    path = Path(importlib.metadata.distribution('bitsandbytes').locate_file('bitsandbytes/backends/default/ops.py'))
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_try_torch_compile')
    namespace = {'torch': torch, 'wraps': functools.wraps}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_try_torch_compile']


def deferred_bnb(fn, compiler, **options):
    record = None
    def intercept(original, *args, **kwargs):
        nonlocal record
        wrapped, record = lazy._defer(compiler, original, args, kwargs)
        return wrapped
    with patch.object(torch, 'compile', intercept):
        wrapped = original_decorator()(fn, **options)
    return wrapped, record


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(n == "bitsandbytes" or n.startswith("bitsandbytes.") for n in sys.modules):
            raise unittest.SkipTest("Import admission tests require a fresh process; see module docstring")

    def test_no_early_construction_or_body_and_original_arguments(self):
        events = []
        def fn(x: int, *, offset=2) -> int:
            'original documentation'
            events.append('body')
            return x + offset
        fn.custom = object()
        def compiler(original, *args, **options):
            events.append(('construct', original, args, options))
            return lambda *a, **kw: original(*a, **kw)
        wrapped, record = deferred_bnb(fn, compiler, dynamic=False)
        self.assertEqual(events, [])
        self.assertEqual(wrapped(3, offset=4), 7)
        self.assertEqual(wrapped(8), 10)
        self.assertEqual(events[0], ('construct', fn, (), {'dynamic': False}))
        self.assertEqual(record.constructions, 1)
        self.assertEqual(events.count('body'), 2)
        for attr in ('__name__', '__qualname__', '__doc__', '__annotations__', '__module__', 'custom'):
            self.assertEqual(getattr(wrapped, attr), getattr(fn, attr))
        self.assertIs(wrapped.__wrapped__, fn)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(fn))

    def test_direct_lazy_preserves_metadata_and_all_compiler_arguments(self):
        def fn(x): return x
        args = ('opaque-positional',)
        opts = {'dynamic': False, 'options': {'opaque': object()}}
        seen = []
        def compiler(*a, **kw):
            seen.append((a, kw)); return fn
        wrapped, _ = lazy._defer(compiler, fn, args, opts)
        self.assertIs(wrapped._torchdynamo_orig_callable, fn)
        self.assertIs(wrapped.__wrapped__, fn)
        self.assertEqual(wrapped(4), 4)
        self.assertEqual(seen, [((fn, *args), opts)])

    def test_permanent_constructor_failure_and_single_eager_call(self):
        calls = []
        def fn(x): calls.append(x); return x + 1
        def compiler(*a, **kw): raise ValueError('constructor failure')
        wrapped, record = deferred_bnb(fn, compiler)
        self.assertEqual([wrapped(1), wrapped(2)], [2, 3])
        self.assertEqual(calls, [1, 2])
        self.assertEqual(record.constructions, 1)
        self.assertEqual(record.constructor_failure, 'ValueError')

    def test_constructor_failure_does_not_retry_failing_eager_body(self):
        calls = []
        error = ValueError('body failure')
        def fn(): calls.append(1); raise error
        def compiler(*a, **kw): raise RuntimeError('constructor')
        wrapped, record = deferred_bnb(fn, compiler)
        for _ in range(2):
            with self.assertRaises(ValueError) as caught: wrapped()
            self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(record.constructions, 1)

    def test_runtime_failure_retries_compiled_on_next_call(self):
        events = []
        def fn(x): events.append('eager'); return x + 1
        def compiled(x):
            events.append('compiled')
            if events.count('compiled') == 1: raise RuntimeError('runtime')
            return x + 1
        wrapped, record = deferred_bnb(fn, lambda *a, **kw: compiled)
        self.assertEqual([wrapped(1), wrapped(2)], [2, 3])
        self.assertEqual(events, ['compiled', 'eager', 'compiled'])
        self.assertEqual(record.constructions, 1)
        self.assertEqual(record.runtime_failures, 1)

    def test_runtime_failure_remains_per_call_not_permanent_eager(self):
        calls = []
        def compiler(fn, **kw):
            def fail(x): calls.append('compiled'); raise RuntimeError()
            return fail
        def fn(x): calls.append('eager'); return x
        wrapped, record = deferred_bnb(fn, compiler)
        self.assertEqual([wrapped(1), wrapped(2)], [1, 2])
        self.assertEqual(calls, ['compiled', 'eager'] * 2)
        self.assertEqual(record.runtime_failures, 2)

    def test_baseexception_is_not_swallowed(self):
        def compiler(*a, **kw): raise KeyboardInterrupt()
        wrapped, _ = deferred_bnb(lambda: 1, compiler)
        with self.assertRaises(KeyboardInterrupt): wrapped()

    def test_concurrent_first_calls_construct_once(self):
        for failure in (False, True):
            barrier = threading.Barrier(8)
            def fn(x): return x * 2
            def compiler(original, **kwargs):
                time.sleep(.025)
                if failure: raise RuntimeError()
                return original
            wrapped, record = deferred_bnb(fn, compiler)
            def run(i): barrier.wait(timeout=5); return wrapped(i)
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(list(pool.map(run, range(8))), list(range(0, 16, 2)))
            self.assertEqual(record.constructions, 1)
            self.assertEqual(record.calls, 8)

    def test_first_call_uses_real_original_compiler_success(self):
        # Scalar control flow needs no C++ toolchain; uses unmodified default
        # torch.compile, with no backend= or disable= override.
        def fn(value: int):
            return (value + 3) * 2
        compiler = torch.compile
        wrapped, record = deferred_bnb(fn, compiler, dynamic=False)
        self.assertEqual(record.constructions, 0)
        self.assertEqual(wrapped(7), 20)
        self.assertIs(record.compiler, compiler)
        self.assertIs(record.fn, fn)
        self.assertEqual(record.constructions, 1)
        self.assertEqual(record.runtime_failures, 0)
        self.assertIsNone(record.constructor_failure)

    def test_disabled_is_noop_without_admission(self):
        original = torch.compile
        with patch.object(lazy, '_admit', side_effect=AssertionError('must not admit')):
            with lazy.install(): self.assertIs(torch.compile, original)

    def test_real_admission(self):
        admission, reason = lazy._admit()
        self.assertIsNotNone(admission, reason)
        self.assertEqual(set(lazy._OPTIONS) & admission[1].keys(), set(lazy._OPTIONS))

    def test_unknown_source_is_noop(self):
        original = torch.compile
        with patch.object(lazy, '_OPS_SHA256', '0' * 64):
            with lazy.install(enabled=True): self.assertIs(torch.compile, original)
        self.assertIn('source digest mismatch', lazy.stats()['status'])

    def test_unknown_version_is_noop(self):
        original = torch.compile
        dist = importlib.metadata.distribution('bitsandbytes')
        with patch.object(type(dist), 'version', property(lambda self: '0.0.0')):
            with lazy.install(enabled=True): self.assertIs(torch.compile, original)

    def test_admission_exception_is_noop(self):
        original = torch.compile
        with patch.object(lazy, '_admit', side_effect=OSError('unreadable')):
            with lazy.install(enabled=True): self.assertIs(torch.compile, original)

    def test_background_scope_is_noop(self):
        original = torch.compile
        def run():
            with lazy.install(enabled=True): return torch.compile is original
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertTrue(pool.submit(run).result(timeout=5))

    def test_unrelated_dispatch_and_restoration_success_failure_nested(self):
        seen = []
        token = object()
        def original(*a, **kw): seen.append((a, kw)); return token
        def fn(x): return x
        for fail in (False, True):
            with patch.object(torch, 'compile', original):
                try:
                    with lazy.install(enabled=True):
                        scoped = torch.compile
                        self.assertIsNot(scoped, original)
                        with lazy.install(enabled=True): self.assertIs(torch.compile, scoped)
                        self.assertIs(torch.compile(), token)
                        self.assertIs(torch.compile(fn, 'arg', dynamic=True), token)
                        self.assertIs(torch.compile(model=fn, fullgraph=True), token)
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            self.assertIs(pool.submit(torch.compile, fn, dynamic=False).result(timeout=5), token)
                        if fail: raise ValueError('import failed')
                except ValueError: pass
                self.assertIs(torch.compile, original)
        expected = [((), {}), ((fn, 'arg'), {'dynamic': True}),
                    ((), {'model': fn, 'fullgraph': True}), ((fn,), {'dynamic': False})]
        self.assertEqual(seen, expected * 2)

    def test_name_spoof_is_rejected(self):
        seen = []
        def fn(x): return x
        fn.__name__ = fn.__qualname__ = '_dequantize_4bit_compute'
        fn.__module__ = lazy._MODULE
        with patch.object(torch, 'compile', lambda *a, **kw: seen.append((a, kw))):
            with lazy.install(enabled=True): torch.compile(fn, dynamic=False)
        self.assertEqual(seen, [((fn,), {'dynamic': False})])

    def test_cleanup_preserves_later_compiler_patch(self):
        original = torch.compile
        replacement = lambda *args, **kwargs: None
        for fail in (False, True):
            try:
                try:
                    with lazy.install(enabled=True):
                        self.assertIsNot(torch.compile, original)
                        torch.compile = replacement
                        if fail:
                            raise ValueError('later import failed')
                except ValueError:
                    pass
                self.assertIs(torch.compile, replacement)
            finally:
                torch.compile = original


class ZRealBNB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(n == 'bitsandbytes' or n.startswith('bitsandbytes.') for n in sys.modules):
            raise unittest.SkipTest('Real import test requires a fresh process; see module docstring')
        if torch.version.cuda is not None:
            raise unittest.SkipTest('These first-call tests require a CPU-only Torch installation')
        cls.original = torch.compile
        match = lazy._matches
        audited = []
        def audit(fn, args, opts, admission, caller, owner, seen):
            accepted = match(fn, args, opts, admission, caller, owner, seen)
            if accepted:
                audited.append(fn.__name__)
                assert not match(fn, (object(),), opts, admission, caller, owner, seen)
                assert not match(fn, args, dict(opts, disable=False), admission, caller, owner, seen)
                assert not match(fn, args, opts, admission, caller, owner + 1, seen)
                assert not match(fn, args, opts, admission, caller, owner, {fn.__name__})
                if opts:
                    assert not match(fn, args, {'dynamic': 0}, admission, caller, owner, seen)
                    assert not match(fn, args, {'dynamic': True}, admission, caller, owner, seen)
                altered = types.FunctionType(fn.__code__.replace(co_filename='other.py'), fn.__globals__)
                assert not match(altered, args, opts, admission, caller, owner, seen)
                original_code = fn.__code__
                try:
                    fn.__code__ = original_code.replace(co_filename='other.py')
                    assert not match(fn, args, opts, admission, caller, owner, seen)
                finally:
                    fn.__code__ = original_code
                wrong_caller = types.SimpleNamespace(f_globals=caller.f_globals,
                    f_code=audit.__code__, f_locals=caller.f_locals)
                assert not match(fn, args, opts, admission, wrong_caller, owner, seen)
            return accepted
        replacement = lambda *args, **kwargs: None
        try:
            with patch.object(lazy, '_matches', audit):
                with lazy.install(enabled=True):
                    import bitsandbytes.backends.default.ops as ops
                    cls.ops = ops
                    cls.before = lazy.stats()
                    assert torch.compile is cls.original
                    torch.compile = replacement
                assert torch.compile is replacement
        finally:
            torch.compile = cls.original
        assert set(audited) == set(lazy._OPTIONS), audited
        assert torch.compile is cls.original
        assert len(cls.before['functions']) == 4, cls.before
        assert all(r['calls'] == r['constructions'] == 0 for r in cls.before['functions']), cls.before

    def test_a_import_no_execution_metadata_canonical_code_and_preimport_noop(self):
        for name in lazy._OPTIONS:
            fn = getattr(self.ops, name)
            self.assertEqual(fn.__name__, name)
            self.assertEqual(fn.__module__, lazy._MODULE)
            self.assertEqual(inspect.signature(fn), inspect.signature(fn.__wrapped__))
        with lazy.install(enabled=True): self.assertIs(torch.compile, type(self).original)
        self.assertIn('already imported', lazy.stats()['status'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_b_actual_default_dequantization_real_compile_and_parity(self):
        code = torch.linspace(-1, 1, 256)
        indices = torch.tensor([0, 7, 15, 42, 99, 127, 254], dtype=torch.uint8)
        absmax = torch.tensor([.5, 1.5])
        fn = self.ops._dequantize_blockwise_compute
        args = (indices, absmax, code, 4, torch.float32)
        torch.testing.assert_close(fn(*args), fn.__wrapped__(*args))
        if lazy._RECORDS[0].runtime_failures:
            try:
                lazy._RECORDS[0].compiled(*args)
            except Exception as exc:
                print('CPU_CODEGEN_ERROR ' + str(exc), flush=True)
        fn = self.ops._dequantize_4bit_compute
        args = (torch.tensor([0x01, 0x27, 0xab, 0xf0], dtype=torch.uint8),
                absmax, torch.linspace(-1, 1, 16), 4, (7,), torch.float32)
        torch.testing.assert_close(fn(*args), fn.__wrapped__(*args))
        for row in lazy.stats()['functions'][:2]:
            self.assertEqual(row['constructions'], 1)
            self.assertIsNone(row['constructor_failure'])
            self.assertIn(row['runtime_failures'], (0, 1), row)

    def test_c_actual_default_optimizer_real_compile_and_mutation_parity(self):
        torch.manual_seed(7)
        g, p = torch.randn(8), torch.randn(8)
        s1, s2, unorm = torch.zeros(8), torch.ones(8), torch.zeros(1)
        cases = [('_optimizer_precondition_32bit',
                  (g, p, s1, s2, unorm, .9, .99, 1e-8, .01, 2, .001, 1., 3)),
                 ('_optimizer_update_32bit',
                  (g, p, s1, s2, unorm, 0., 1., .9, .99, .999, 0., 1e-8, .01, 2, .001, 1., 3))]
        for name, args in cases:
            fn = getattr(self.ops, name)
            actual = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
            expected = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
            self.assertEqual(fn(*actual), fn.__wrapped__(*expected))
            for a, e in zip(actual, expected):
                if isinstance(a, torch.Tensor): torch.testing.assert_close(a, e)
        for row in lazy.stats()['functions']:
            self.assertEqual(row['constructions'], 1)
            self.assertIsNone(row['constructor_failure'])
            self.assertIn(row['runtime_failures'], (0, 1), row)
        self.assertFalse(torch.cuda.is_initialized())
        print('LAZY_BNB_CPU_STATS ' + json.dumps(lazy.stats()), flush=True)

    def test_d_cpu_codegen_success_when_toolchain_available(self):
        if not shutil.which('g++') and not shutil.which('clang++'):
            self.skipTest('No installed C++ compiler; actual tensor parity above exercised BNB runtime fallback')
        for row in lazy.stats()['functions']:
            self.assertEqual(row['runtime_failures'], 0, row)


if __name__ == '__main__':
    unittest.main()
