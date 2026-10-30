"""packed_attn.py on CPU: variants A (per-row sdpa_attention_forward), A2 (repeat_kv once + per-row causal SDPA) and
B (one SDPA with a block-diagonal causal mask, N padded to a multiple of 8) on a tiny random Qwen2 (fp32).
Compared: hidden states on the real tokens, loss and parameter gradients (tolerance 1e-5), also against the padded
transformers forward. Odd N, rows of different length, right padding."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from solution import packed, packed_attn  # noqa: E402
from solution.test_packed import loss_of, make_batch, run, tiny_model  # noqa: E402

TOL = 1e-5
CASES = {"even": [20, 13, 7, 16], "odd_N": [19, 12, 7, 15], "one_short": [18, 1, 9, 18], "single_row": [11]}


@pytest.fixture(autouse=True)
def restore():
    old = packed_attn.VARIANT
    yield
    packed_attn.VARIANT = old
    packed.unpatch()
    packed.reset_stats()


def packed_run(model, ids, mask, variant):
    packed.patch()
    packed_attn.VARIANT = variant
    out, loss = run(model, ids, mask)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    return out, loss.detach(), grads


@pytest.mark.parametrize("case", sorted(CASES))
def test_variants_agree_with_each_other_and_with_padded(case):
    lens = CASES[case]
    if len(lens) == 1:
        lens = lens + [lens[0] - 1]            # a single full-length row has no pad: add a shorter row
    model = tiny_model()
    ids, mask = make_batch(lens)
    keep = mask.bool()

    packed.unpatch()
    ref_out, ref_loss = run(model, ids, mask)
    ref_loss.backward()
    ref_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    for variant in packed_attn.VARIANTS:
        out, loss, grads = packed_run(model, ids, mask, variant)
        assert out.packed_keep is not None
        assert out.last_hidden_state.shape == (1, sum(lens), 64), variant        # B: filler tokens are dropped
        assert (out.last_hidden_state[0] - ref_out.last_hidden_state[keep]).abs().max() < TOL, variant
        assert abs(loss.item() - ref_loss.item()) < TOL, variant
        assert grads.keys() == ref_grads.keys()
        assert max((grads[n] - ref_grads[n]).abs().max().item() for n in ref_grads) < TOL, variant


def test_variant_b_pads_to_a_multiple_of_8():
    packed_attn.VARIANT = "B"
    assert [packed_attn.extra_tokens(n) for n in (56, 53, 1, 8)] == [0, 3, 7, 0]
    packed_attn.VARIANT = "A2"
    assert packed_attn.extra_tokens(53) == 0


def test_block_mask_is_block_diagonal_causal_in_the_dtype_of_q():
    packed_attn.VARIANT = "B"
    packed_attn.begin([3, 2, 3], torch.device("cpu"))
    mask = packed_attn.block_causal_mask(torch.float32)
    assert mask.shape == (1, 1, 8, 8) and mask.dtype == torch.float32
    allowed = mask[0, 0] == 0
    expect = torch.zeros(8, 8, dtype=torch.bool)
    for lo, hi in ((0, 3), (3, 5), (5, 8)):
        expect[lo:hi, lo:hi] = torch.ones(hi - lo, hi - lo).tril().bool()
    assert torch.equal(allowed, expect)
    assert packed_attn.block_causal_mask(torch.float16).dtype == torch.float16
    packed_attn.end()


@pytest.mark.parametrize("variant", ["A2", "B"])
def test_rows_do_not_see_each_other(variant):
    model = tiny_model()
    ids, mask = make_batch([14, 9, 5, 12])
    packed.patch()
    packed_attn.VARIANT = variant
    base = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    ids2 = ids.clone()
    ids2[1, :9] = (ids2[1, :9] + 5) % 999 + 1                      # change only row 1
    other = model.model(input_ids=ids2, attention_mask=mask).last_hidden_state
    starts = [0, 14, 23, 28]
    for row, (lo, n) in enumerate(zip(starts, [14, 9, 5, 12])):
        same = torch.equal(base[0, lo:lo + n], other[0, lo:lo + n])
        assert same == (row != 1), (variant, row)
