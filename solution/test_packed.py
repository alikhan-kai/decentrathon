"""packed.py on CPU: padded transformers forward vs packed forward on a tiny random Qwen2 (fp32).

Compared: hidden states on the real positions, loss, gradients of every parameter (tolerance 1e-5).
Also: pad content does not matter, fallbacks give the original result, LoRA (peft) works on [1, N, H].
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from solution import packed  # noqa: E402

TOL = 1e-5
LENS = [20, 13, 7, 16]


def tiny_model():
    config = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=1000, max_position_embeddings=64, attention_dropout=0.0,
                         use_sliding_window=False, pad_token_id=0)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).float().train()


def make_batch(lens=LENS, pad_id=0, left=False, seed=1):
    gen = torch.Generator().manual_seed(seed)
    t = max(lens)
    ids = torch.full((len(lens), t), pad_id, dtype=torch.long)
    mask = torch.zeros(len(lens), t, dtype=torch.long)
    for i, n in enumerate(lens):
        sl = slice(t - n, t) if left else slice(0, n)
        ids[i, sl] = torch.randint(1, 1000, (n,), generator=gen)
        mask[i, sl] = 1
    return ids, mask


def loss_of(model, hidden, ids, mask):
    """Same loss for both layouts: hidden is [B, T, H] (padded) or [1, N, H] (packed, row-major real tokens)."""
    keep = mask.bool()
    target = torch.nn.functional.pad(ids, (0, 1), value=-100)[:, 1:]
    target = torch.where(keep & (torch.nn.functional.pad(keep, (0, 1))[:, 1:]), target, torch.full_like(target, -100))
    h = hidden[0] if hidden.shape[0] == 1 and keep.shape[0] != 1 else hidden[keep]
    logits = model.lm_head(h)
    return torch.nn.functional.cross_entropy(logits, target[keep], ignore_index=-100)


@pytest.fixture
def patched():
    packed.patch()
    yield
    packed.unpatch()


@pytest.fixture(autouse=True)
def clean_stats():
    """The tests' counters must not end up in the PACKED_STATS line printed at exit."""
    yield
    packed.reset_stats()


def run(model, ids, mask):
    out = model.model(input_ids=ids, attention_mask=mask)
    return out, loss_of(model, out.last_hidden_state, ids, mask)


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_plan():
    _, mask = make_batch()
    lens, flat, positions = packed.plan(mask)
    assert lens == LENS and flat.numel() == sum(LENS)
    assert torch.equal(mask.reshape(-1)[flat], torch.ones(sum(LENS), dtype=torch.long))
    assert positions.shape == (1, sum(LENS)) and positions.max() == max(LENS) - 1
    assert packed.plan(make_batch(left=True)[1]) == "left_pad_or_hole"
    assert packed.plan(make_batch([9, 9, 9])[1]) == "no_pad"        # nothing to gain
    assert packed.plan(torch.tensor([[1, 0, 1], [1, 1, 1]])) == "left_pad_or_hole"
    assert packed.plan(torch.tensor([[1, 1], [0, 0]])) == "empty_row"


def test_packed_matches_padded(patched):
    model = tiny_model()
    ids, mask = make_batch()
    keep = mask.bool()

    packed.unpatch()
    ref_out, ref_loss = run(model, ids, mask)
    ref_loss.backward()
    ref_grads = grads(model)
    model.zero_grad()

    packed.patch()
    out, loss = run(model, ids, mask)
    assert out.packed_keep is not None and out.last_hidden_state.shape == (1, sum(LENS), 64)
    assert (out.last_hidden_state[0] - ref_out.last_hidden_state[keep]).abs().max() < TOL
    assert abs(loss.item() - ref_loss.item()) < TOL
    loss.backward()
    new_grads = grads(model)
    assert new_grads.keys() == ref_grads.keys() and len(ref_grads) > 10
    worst = max((new_grads[n] - ref_grads[n]).abs().max().item() for n in ref_grads)
    assert worst < TOL, worst


def test_pad_content_does_not_leak(patched):
    model = tiny_model()
    ids, mask = make_batch(pad_id=0)
    ids2 = ids.clone()
    ids2[mask == 0] = 777
    a = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    b = model.model(input_ids=ids2, attention_mask=mask).last_hidden_state
    assert torch.equal(a, b)


def test_padded_reference_ignores_pad_too():
    """Right padding: pad does not change real tokens in the ORIGINAL padded forward either (the premise of packing)."""
    model = tiny_model()
    ids, mask = make_batch()
    ids2 = ids.clone()
    ids2[mask == 0] = 777
    a = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    b = model.model(input_ids=ids2, attention_mask=mask).last_hidden_state
    assert (a[mask.bool()] - b[mask.bool()]).abs().max() < TOL


@pytest.mark.parametrize("case", ["left", "no_pad", "eval", "position_ids"])
def test_fallback_is_the_original(patched, case):
    model = tiny_model()
    ids, mask = make_batch([9, 9, 9] if case == "no_pad" else LENS, left=(case == "left"))
    kwargs = {}
    if case == "eval":
        model.eval()
    if case == "position_ids":
        kwargs["position_ids"] = torch.arange(ids.shape[1]).unsqueeze(0)
    with torch.no_grad() if case == "eval" else torch.enable_grad():
        out = model.model(input_ids=ids, attention_mask=mask, **kwargs)
        packed.unpatch()
        ref = model.model(input_ids=ids, attention_mask=mask, **kwargs)
    assert getattr(out, "packed_keep", None) is None
    assert out.last_hidden_state.shape == ref.last_hidden_state.shape
    assert torch.equal(out.last_hidden_state, ref.last_hidden_state)


def test_lora_accepts_packed(patched):
    peft = pytest.importorskip("peft")
    model = tiny_model()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
    try:
        model = peft.get_peft_model(model, cfg)
    except ImportError as error:        # e.g. Kaggle image: preinstalled torchao older than peft wants
        pytest.skip(f"peft cannot be imported here: {error}")
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.1)                  # B = 0 would hide errors in the A branch
    inner = model.get_base_model()
    ids, mask = make_batch()
    keep = mask.bool()

    packed.unpatch()
    ref_out, ref_loss = run(inner, ids, mask)
    ref_loss.backward()
    ref_grads = {n: g for n, g in grads(model).items() if "lora" in n}
    model.zero_grad()

    packed.patch()
    out, loss = run(inner, ids, mask)
    assert (out.last_hidden_state[0] - ref_out.last_hidden_state[keep]).abs().max() < TOL
    loss.backward()
    new_grads = {n: g for n, g in grads(model).items() if "lora" in n}
    assert new_grads.keys() == ref_grads.keys() and len(ref_grads) == 8
    assert max((new_grads[n] - ref_grads[n]).abs().max().item() for n in ref_grads) < TOL


def test_stats_count_steps_and_fallbacks(patched):
    model = tiny_model()
    before = packed.stats()
    ids, mask = make_batch()
    model.model(input_ids=ids, attention_mask=mask)
    ids2, mask2 = make_batch([9, 9, 9])
    model.model(input_ids=ids2, attention_mask=mask2)
    after = packed.stats()
    assert after["packed_steps"] == before["packed_steps"] + 1
    assert after["n_real"][-1] == sum(LENS) and after["n_padded"][-1] == len(LENS) * max(LENS)
    assert after["fallbacks"].get("no_pad", 0) == before["fallbacks"].get("no_pad", 0) + 1


def test_patch_is_idempotent():
    packed.patch()
    first = packed._ORIGINAL["model"]
    packed.patch()
    assert packed._ORIGINAL["model"] is first
    packed.unpatch()
    packed.unpatch()
