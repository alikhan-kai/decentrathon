import pytest

pytest.importorskip("soup_cli")


def test_overhead_patch_idempotent():
    from solution import overhead

    overhead.patch()
    overhead.patch()


def test_device_map_cpu_untouched():
    from soup_cli.trainer import sft

    from solution import overhead

    overhead.patch()
    assert sft.resolve_device_map("cpu") != {"": 0}
