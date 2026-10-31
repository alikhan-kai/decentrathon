"""CPU checks for the MLP activation budget, including its irreducible floor."""

import pytest

pytest.importorskip("torch")


def saved_kb(n, k, layers=28):
    from solution.swiglu_pack import NATIVE_KB, REMAT_KB

    return n * (k * NATIVE_KB + (layers - k) * REMAT_KB)


@pytest.mark.parametrize("n", range(1025, 2049, 7))
def test_budget_or_irreducible_floor(n):
    from solution.swiglu_pack import BUDGET_ROWS, NATIVE_KB, REMAT_KB, native_layers

    k = native_layers(n, 28)
    assert 0 <= k <= 28
    limit = max(BUDGET_ROWS * 28 * NATIVE_KB, n * 28 * REMAT_KB)
    assert saved_kb(n, k) <= limit
    if k < 28:
        assert saved_kb(n, k + 1) > limit


def test_more_rows_keep_fewer_native_layers():
    from solution.swiglu_pack import native_layers

    ks = [native_layers(n, 28) for n in range(1025, 2049)]
    assert ks == sorted(ks, reverse=True)
    assert ks[0] > 0  # just above the threshold most layers skip recomputation
