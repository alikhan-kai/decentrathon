import importlib.metadata
import sys
from types import SimpleNamespace

import pytest

from solution import startup_metadata


@pytest.mark.parametrize("fail", [False, True])
def test_copies_and_restores_function(monkeypatch, fail):
    mapping = {"PIL": ["Pillow"], "torch": ["torch"]}
    original = importlib.metadata.packages_distributions
    monkeypatch.delitem(sys.modules, "trl.import_utils", raising=False)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils",
                        SimpleNamespace(PACKAGE_DISTRIBUTION_MAPPING=mapping))

    def import_trl(name):
        assert name == "trl"
        first = importlib.metadata.packages_distributions()
        assert first == mapping
        first["PIL"].append("mutated")
        first["new"] = ["new"]
        assert importlib.metadata.packages_distributions() == mapping
        assert mapping["PIL"] == ["Pillow"]
        if fail:
            raise RuntimeError("import failed")

    monkeypatch.setattr(startup_metadata.importlib, "import_module", import_trl)
    if fail:
        with pytest.raises(RuntimeError, match="import failed"):
            startup_metadata.prepare_trl_import()
    else:
        startup_metadata.prepare_trl_import()
    assert importlib.metadata.packages_distributions is original


@pytest.mark.parametrize("mapping", [None, {"x": "not a list"}, {"x": [1]}])
def test_unsupported_mapping_keeps_original(monkeypatch, mapping):
    original = importlib.metadata.packages_distributions
    monkeypatch.delitem(sys.modules, "trl.import_utils", raising=False)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils",
                        SimpleNamespace(PACKAGE_DISTRIBUTION_MAPPING=mapping))
    startup_metadata.prepare_trl_import()
    assert importlib.metadata.packages_distributions is original
