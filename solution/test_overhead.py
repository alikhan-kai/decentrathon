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


def test_fast_map_matches_soup():
    from solution import overhead

    overhead.patch()
    if overhead._orig_map_rows is None:
        pytest.skip("FAST_MAP off")

    def format_row(r):
        n = r["n"]
        return {"input_ids": list(range(n)), "attention_mask": [1] * n,
                "labels": [-100] + list(range(1, n)), "messages": [{"role": "user"}]}

    from soup_cli.trainer import sft

    rows = [{"n": 5}, {"n": 8}, {"n": 3}]
    kw = dict(format_row=format_row, split="train", max_length=16)
    ref = overhead._orig_map_rows(rows, **kw)
    got = sft._map_text_sft_rows(rows, **kw)
    assert sorted(got.column_names) == sorted(ref.column_names)
    assert got.features == ref.features
    assert got.to_list() == ref.to_list()
