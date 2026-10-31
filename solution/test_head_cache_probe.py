import pytest
import torch
from solution import head_loss as candidate

@pytest.mark.parametrize('scale', [1.0, 128.0, -32.0])
@pytest.mark.parametrize('cache', [True, False])
def test_cached_projection_preserves_scaled_gradient(scale, cache, monkeypatch):
    monkeypatch.setattr(candidate, 'CACHE_LOGITS_BYTES', 512*1024*1024 if cache else 0)
    torch.manual_seed(832)
    h = (torch.randn(259, 64, device='cuda')*.2).requires_grad_()
    hr = h.detach().clone().requires_grad_()
    w = torch.randn(5000, 64, device='cuda', dtype=torch.float16)*.1
    t = torch.randint(0, 5000, (259,), device='cuda')
    t[::3] = -100
    denom = (t >= 0).sum()
    a = candidate.fused_head_loss(h,w,t,denom,chunk=64)
    monkeypatch.setattr(candidate, 'CACHE_LOGITS_BYTES', 0)
    r = candidate.fused_head_loss(hr,w,t,denom,chunk=64)
    for x,y in zip(a,r):
        torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
    (a[0]*scale).backward()
    (r[0]*scale).backward()
    torch.testing.assert_close(h.grad,hr.grad,atol=2e-5,rtol=2e-3)
    assert torch.count_nonzero(h.grad[::3]) == 0

def test_cached_logits_support_repeated_backward():
    torch.manual_seed(3)
    h=torch.randn(37,64,device='cuda',requires_grad=True)
    w=torch.randn(5000,64,device='cuda',dtype=torch.float16)*.1
    t=torch.randint(0,5000,(37,),device='cuda')
    loss,*_=candidate.fused_head_loss(h,w,t,37,chunk=16)
    loss.backward(retain_graph=True)
    first=h.grad.clone()
    h.grad.zero_()
    loss.backward()
    assert torch.equal(first,h.grad)

def test_ignored_targets_have_zero_gradient_even_with_zero_denominator():
    h=torch.randn(37,64,device='cuda',requires_grad=True)
    w=torch.randn(5000,64,device='cuda',dtype=torch.float16)*.1
    t=torch.full((37,),-100,device='cuda',dtype=torch.long)
    loss,*_=candidate.fused_head_loss(h,w,t,0,chunk=16)
    assert torch.isnan(loss)
    loss.backward()
    assert torch.count_nonzero(h.grad)==0
