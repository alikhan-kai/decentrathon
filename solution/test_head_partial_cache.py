import pytest
import torch
from solution import head_loss as candidate

@pytest.mark.parametrize('rows_to_cache', [0,1,63,64,65,173,259])
@pytest.mark.parametrize('scale', [1.0,128.0])
def test_partial_cache_boundary_parity(rows_to_cache,scale,monkeypatch):
    torch.manual_seed(57)
    h=torch.randn(259,64,device='cuda',requires_grad=True)
    hr=h.detach().clone().requires_grad_()
    w=torch.randn(5000,64,device='cuda',dtype=torch.float16)*.1
    t=torch.randint(0,5000,(259,),device='cuda')
    t[::5]=-100
    denom=(t>=0).sum()
    monkeypatch.setattr(candidate,'CACHE_LOGITS_BYTES',rows_to_cache*5000*2)
    a=candidate.fused_head_loss(h,w,t,denom,chunk=64)
    saved=a[0].grad_fn.saved_tensors[-2]
    assert saved.numel()*saved.element_size() <= rows_to_cache*5000*2
    monkeypatch.setattr(candidate,'CACHE_LOGITS_BYTES',0)
    r=candidate.fused_head_loss(hr,w,t,denom,chunk=64)
    for x,y in zip(a,r): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
    (a[0]*scale).backward(retain_graph=True)
    (r[0]*scale).backward()
    torch.testing.assert_close(h.grad,hr.grad,atol=2e-5*scale,rtol=2e-3)
    first=h.grad.clone(); h.grad.zero_(); (a[0]*scale).backward()
    torch.testing.assert_close(h.grad,first,atol=0,rtol=0)
