import functools
import importlib.machinery
import importlib.metadata as md
import importlib.util
from pathlib import Path
import sys
import threading

import pytest
import transformers
from transformers.utils import import_utils as ti
from solution import startup_metadata as m
HERE = Path(__file__).resolve().parent

ORIGINAL = md.packages_distributions


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(m, 'ENABLED', True)
    assert md.packages_distributions is ORIGINAL
    yield
    assert md.packages_distributions is ORIGINAL
    assert not m._IN_CONTEXT


def test_fresh_copies_and_source_isolation(monkeypatch):
    source = ti.PACKAGE_DISTRIBUTION_MAPPING
    before = {k: list(v) for k, v in source.items()}
    with m.reuse_package_mapping():
        assert md.packages_distributions is not ORIGINAL
        first = md.packages_distributions()
        first['transformers'].append('caller')
        first['injected'] = []
        with monkeypatch.context() as patch:
            patch.setitem(source, 'transformers', ['source-mutation'])
            assert md.packages_distributions() == before
        assert all(first[k] is not source[k] for k in source)
    assert source == before


def test_exception_and_escaped_function():
    with pytest.raises(ValueError, match='expected'):
        with m.reuse_package_mapping():
            escaped = md.packages_distributions
            raise ValueError('expected')
    assert md.packages_distributions is ORIGINAL
    assert escaped() == ORIGINAL()


def test_reader_completes_after_exit(monkeypatch):
    paused, resume = threading.Event(), threading.Event()
    result, errors = [], []
    copy = m._copy_mapping
    reader = None

    def delayed(mapping):
        if threading.current_thread() is reader:
            paused.set()
            assert resume.wait(5)
        return copy(mapping)

    monkeypatch.setattr(m, '_copy_mapping', delayed)
    with m.reuse_package_mapping():
        escaped = md.packages_distributions
        expected = escaped()
        def read():
            try:
                result.append(escaped())
            except BaseException as error:
                errors.append(error)
        reader = threading.Thread(target=read)
        reader.start()
        assert paused.wait(5)
    # Reader has passed active checks and is paused at the actual copy boundary.
    assert md.packages_distributions is ORIGINAL
    resume.set()
    reader.join(5)
    assert not reader.is_alive() and not errors
    assert result == [expected] and expected


def test_direct_contexts_serialize(monkeypatch):
    real_lock = threading.RLock()
    attempted, entered, done = threading.Event(), threading.Event(), threading.Event()
    errors = []
    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is worker:
                attempted.set()
            real_lock.acquire()
        def __exit__(self, *args):
            real_lock.release()
    worker = None
    monkeypatch.setattr(m, '_CONTEXT_LOCK', ObservedLock())
    def work():
        try:
            with m.reuse_package_mapping():
                entered.set()
                assert md.packages_distributions is not ORIGINAL
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    with m.reuse_package_mapping():
        outer = md.packages_distributions
        worker = threading.Thread(target=work)
        worker.start()
        assert attempted.wait(5)
        assert not entered.is_set()
        assert md.packages_distributions is outer
    assert done.wait(5)
    worker.join(5)
    assert entered.is_set() and not errors


@pytest.mark.parametrize('kind', ['path', 'finders', 'cwd'])
def test_nested_changed_search_keeps_outer_baseline(monkeypatch, kind, tmp_path):
    with m.reuse_package_mapping():
        outer = md.packages_distributions
        with monkeypatch.context() as patch:
            if kind == 'path':
                patch.setattr(sys, 'path', [p for p in sys.path
                                          if not p.endswith(('site-packages', 'dist-packages'))])
            elif kind == 'finders':
                patch.setattr(sys, 'meta_path', [object(), *sys.meta_path])
            else:
                changed_cwd = tmp_path / 'changed-cwd'
                changed_cwd.mkdir(exist_ok=True)
                patch.chdir(changed_cwd)
            def unexpected_copy(mapping):
                pytest.fail('changed search must delegate without copying or reseeding')
            patch.setattr(m, '_copy_mapping', unexpected_copy)
            expected = ORIGINAL()
            if kind == 'path':
                assert 'transformers' not in expected
            assert outer() == expected
            with m.reuse_package_mapping():
                assert md.packages_distributions is outer
                assert md.packages_distributions() == expected
        assert md.packages_distributions is outer


@pytest.mark.parametrize('before_load', [False, True])
def test_preexisting_hook_declined(monkeypatch, before_load):
    @functools.wraps(ORIGINAL)
    def hook():
        return {'hook-only': ['value']}
    with monkeypatch.context() as patch:
        patch.setattr(md, 'packages_distributions', hook)
        module = m
        if before_load:
            spec = importlib.util.spec_from_file_location('late_helper', HERE / 'startup_metadata.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ENABLED = True
        with module.reuse_package_mapping():
            assert md.packages_distributions is hook
            assert md.packages_distributions() == {'hook-only': ['value']}


@pytest.mark.parametrize('kind', ['extra', 'pathfinder', 'metadatafinder', 'discovery'])
def test_custom_resolver_declined(monkeypatch, kind):
    class CustomFinder:
        @staticmethod
        def find_distributions(context):
            return iter(())
    with monkeypatch.context() as patch:
        if kind == 'extra':
            patch.setattr(sys, 'meta_path', [*sys.meta_path, CustomFinder()])
        elif kind == 'pathfinder':
            patch.setattr(importlib.machinery.PathFinder, 'find_distributions',
                          CustomFinder.find_distributions)
        elif kind == 'metadatafinder':
            patch.setattr(md.MetadataPathFinder, 'find_distributions',
                          CustomFinder.find_distributions)
        else:
            patch.setattr(md.Distribution, '_discover_resolvers',
                          staticmethod(lambda: iter((CustomFinder.find_distributions,))))
        with m.reuse_package_mapping():
            assert md.packages_distributions is ORIGINAL


def test_in_scope_resolver_replacement_falls_back(monkeypatch):
    with m.reuse_package_mapping():
        with monkeypatch.context() as patch:
            patch.setattr(md.MetadataPathFinder, 'find_distributions', lambda context: iter(()))
            assert md.packages_distributions() == {}
            with m.reuse_package_mapping():
                assert md.packages_distributions() == {}


def test_later_hook_preserved(monkeypatch):
    with monkeypatch.context() as patch:
        with m.reuse_package_mapping():
            outer = md.packages_distributions
            later = lambda: outer()
            patch.setattr(md, 'packages_distributions', later)
        assert md.packages_distributions is later
        assert later() == ORIGINAL()
    # monkeypatch restores the escaped wrapper, because it was the prior value.
    md.packages_distributions = ORIGINAL


@pytest.mark.parametrize('case', ['disabled', 'missing', 'version', 'bad_mapping', 'bad_list'])
def test_unsupported(monkeypatch, case):
    with monkeypatch.context() as patch:
        if case == 'disabled':
            patch.setattr(m, 'ENABLED', False)
        elif case == 'missing':
            patch.delitem(sys.modules, 'transformers.utils.import_utils')
        elif case == 'version':
            patch.setattr(transformers, '__version__', 'other')
        elif case == 'bad_mapping':
            patch.setattr(ti, 'PACKAGE_DISTRIBUTION_MAPPING', [])
        else:
            patch.setattr(ti, 'PACKAGE_DISTRIBUTION_MAPPING', {'torch': ('torch',)})
        with m.reuse_package_mapping():
            assert md.packages_distributions is ORIGINAL


def test_wrapped_apply_reentrant_and_exception():
    def inner(value, *, fail=False):
        assert md.packages_distributions is not ORIGINAL
        if fail:
            raise LookupError('expected')
        return value
    wrapped = m.wrap_apply(inner)
    assert m.wrap_apply(wrapped) is wrapped
    outer = m.wrap_apply(lambda: wrapped(7))
    assert outer() == 7
    with pytest.raises(LookupError):
        wrapped(0, fail=True)


def test_default_is_disabled():
    spec = importlib.util.spec_from_file_location("fresh_startup_metadata", HERE / "startup_metadata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ENABLED is False
    with module.reuse_package_mapping():
        assert md.packages_distributions is ORIGINAL
