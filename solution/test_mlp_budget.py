"""solution/swiglu_pack.native_layers: the per-step MLP save budget above 1024 rows (CPU only)."""

import pytest

pytest.importorskip("torch")


def saved_kb(n, k, layers=28):
    from solution.swiglu_pack import NATIVE_KB, REMAT_KB

    return n * (k * NATIVE_KB + (layers - k) * REMAT_KB)


@pytest.mark.parametrize("n", range(1025, 2049, 7))
def test_budget_never_exceeds_all_native_1024_rows(n):
    from solution.swiglu_pack import NATIVE_KB, native_layers

    k = native_layers(n, 28)
    assert 0 <= k <= 28
    assert saved_kb(n, k) <= 1024 * 28 * NATIVE_KB


def test_more_rows_keep_fewer_native_layers():
    from solution.swiglu_pack import native_layers

    ks = [native_layers(n, 28) for n in range(1025, 2049)]
    assert ks == sorted(ks, reverse=True)
    assert ks[0] > 0  # just above the threshold most layers skip recomputation
