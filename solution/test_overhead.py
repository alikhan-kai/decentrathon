import pytest

pytest.importorskip("soup_cli")


def test_overhead_patch_idempotent():
    from solution import overhead

    overhead.patch()
    overhead.patch()


def test_fast_trl_import():
    from solution import overhead

    overhead.patch()
    import trl

    assert hasattr(trl, "SFTTrainer")


def test_device_map_cpu_untouched():
    from soup_cli.trainer import sft

    from solution import overhead

    overhead.patch()
    assert sft.resolve_device_map("cpu") != {"": 0}


def _compare(rows, format_row, max_length=512):
    from soup_cli.trainer import sft

    from solution import overhead

    overhead.patch()
    assert overhead._orig_map_rows is not None, "FAST_MAP is off"
    kw = dict(format_row=format_row, split="train", max_length=max_length)
    ref = overhead._orig_map_rows(rows, **kw)
    got = sft._map_text_sft_rows(rows, **kw)
    assert got.column_names == ref.column_names
    assert got.features == ref.features
    assert len(got) == len(ref)
    for name in ref.column_names:
        assert got[name] == ref[name], name  # every value, in row order
    return ref


def _format_row(max_length):

    def format_row(r):
        n = min(r["n"], max_length)  # truncation, like Soup's format_row
        ids = [(r["n"] * 7 + j) % 1000 for j in range(n)]
        return {
            "input_ids": ids,
            "attention_mask": [1] * n,
            "labels": [-100] * 2 + ids[2:],
        }

    return format_row


def test_fast_map_matches_soup():
    rows = [
        {"n": n, "messages": [{"role": "user", "content": str(n)}]}
        for n in (5, 8, 3, 511, 512, 513, 700, 2000, 4)
    ]
    ref = _compare(rows, _format_row(512))
    assert max(len(x) for x in ref["input_ids"]) == 512  # truncated to 512
    assert [len(x) for x in ref["input_ids"]][:3] == [5, 8, 3]  # order kept


def test_fast_map_keeps_extra_columns():
    rows = [{"n": n, "extra": f"e{n}", "messages": []} for n in (4, 9, 6)]
    ref = _compare(rows, _format_row(512))
    assert "extra" in ref.column_names


def test_fast_map_error_matches_soup():
    from soup_cli.trainer import sft

    from solution import overhead

    overhead.patch()

    def format_row(r):
        return {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, -100],
        }

    rows = [{"n": 2, "messages": []}]
    kw = dict(format_row=format_row, split="train", max_length=512)
    with pytest.raises(ValueError) as ref:
        overhead._orig_map_rows(rows, **kw)
    with pytest.raises(ValueError) as got:
        sft._map_text_sft_rows(rows, **kw)
    assert str(got.value) == str(ref.value)
