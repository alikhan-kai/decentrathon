"""Bit packing must preserve native dropout values, RNG and scaled gradients."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(3, 37), (1, 941, 1536), (1, 1427, 1536)])
@pytest.mark.parametrize('scale', [1.0, 128.0])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_native_dropout_exactness(shape, scale, dtype):
    from solution.dropout_pack import dropout_or_pack
    torch.manual_seed(1837)
    x = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    dropout = torch.nn.Dropout(.05).train()
    saved = []
    def observe(t):
        saved.append((t.dtype, t.numel()))
        return t
    torch.manual_seed(912)
    reference = dropout(xr)
    reference_rng = torch.cuda.get_rng_state()
    torch.manual_seed(912)
    with torch.autograd.graph.saved_tensors_hooks(observe, lambda t: t):
        actual = dropout_or_pack(dropout, x)
    assert torch.equal(reference_rng, torch.cuda.get_rng_state())
    assert saved == [(torch.uint8, (x.numel() + 7) // 8)]
    assert torch.equal(actual, reference)
    grad = torch.randn_like(x) * scale
    actual.backward(grad)
    reference.backward(grad)
    assert torch.equal(x.grad, xr.grad)


def test_module_hooks_and_inactive_modes_fall_back():
    from solution.dropout_pack import dropout_or_pack
    x = torch.ones(2, 17, device='cuda', requires_grad=True)
    dropout = torch.nn.Dropout(.05).train()
    called = []
    hook = dropout.register_forward_hook(lambda *_: called.append(True))
    dropout_or_pack(dropout, x)
    assert called == [True]
    hook.remove()
    dropout.eval()
    assert dropout_or_pack(dropout, x) is x


def test_higher_order_backward_matches_fixed_native_mask():
    from solution.dropout_pack import dropout_or_pack
    x = torch.randn(3, 37, device='cuda', requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    dropout = torch.nn.Dropout(.05).train()
    torch.manual_seed(172)
    actual = dropout_or_pack(dropout, x).square().sum()
    torch.manual_seed(172)
    reference = dropout(xr).square().sum()
    ga = torch.autograd.grad(actual, x, create_graph=True)[0]
    gr = torch.autograd.grad(reference, xr, create_graph=True)[0]
    torch.testing.assert_close(ga, gr, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(ga.sum(), x)[0],
                               torch.autograd.grad(gr.sum(), xr)[0], rtol=0, atol=0)


def test_nonfinite_gradient_signaling_matches_native():
    from solution.dropout_pack import dropout_or_pack
    x = torch.ones(256, device='cuda', requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    dropout = torch.nn.Dropout(.05).train()
    torch.manual_seed(712)
    actual = dropout_or_pack(dropout, x)
    torch.manual_seed(712)
    reference = dropout(xr)
    grad = torch.tensor([float('inf'), -float('inf'), float('nan'), 1e-38],
                        device='cuda').repeat(64)
    actual.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0, equal_nan=True)
