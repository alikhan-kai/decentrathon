import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_matches_torch():
    from solution.head_loss import fused_head_loss

    torch.manual_seed(0)
    n, hdim, vocab = 37, 64, 5000          # vocab is not a multiple of BLOCK on purpose
    h = torch.randn(n, hdim, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(vocab, hdim, device="cuda") * 0.1
    tgt = torch.randint(0, vocab, (n,), device="cuda")
    tgt[::5] = -100
    denom = torch.tensor(float((tgt != -100).sum()), device="cuda")

    loss, row, ent, hit = fused_head_loss(h, w, tgt, denom, chunk=16)
    (loss * 3.0).backward()

    h2 = h.detach().clone().requires_grad_(True)
    lf = (h2 @ w.half().t()).float()
    ref_row = torch.nn.functional.cross_entropy(lf, tgt, ignore_index=-100, reduction="none")
    (ref_row.sum() / denom * 3.0).backward()
    logp = lf.log_softmax(-1)
    ref_ent = -(logp.exp() * logp).sum(-1)
    ref_hit = (lf.argmax(-1) == tgt) & (tgt != -100)

    assert torch.allclose(row, ref_row, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ent, ref_ent, atol=1e-3, rtol=1e-3)
    assert (hit.bool() == ref_hit).all()
    assert torch.allclose(h.grad.float(), h2.grad.float(), atol=2e-3, rtol=2e-2)
