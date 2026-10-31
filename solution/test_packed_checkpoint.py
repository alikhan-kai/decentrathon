"""One-layer checkpoint preserves stochastic outputs, gradients and context."""
import copy
import pytest
import torch
from solution.packed import _ACTIVE, _checkpoint_first_layer


@pytest.mark.parametrize("requires_input_grad", [False, True])
def test_checkpoint_rng_gradients_and_context(requires_input_grad):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 16)
            self.drop = torch.nn.Dropout(0.2)

        def forward(self, x):
            assert _ACTIVE.get() == "expected"
            return self.drop(self.proj(x)).square()

    torch.manual_seed(100)
    reference = Layer()
    candidate = copy.deepcopy(reference)
    x = torch.randn(2, 16, requires_grad=requires_input_grad)
    y = x.detach().clone().requires_grad_(requires_input_grad)
    token = _ACTIVE.set("expected")
    try:
        torch.manual_seed(200)
        out = reference(x)
    finally:
        _ACTIVE.reset(token)
    torch.manual_seed(200)
    got = _checkpoint_first_layer(candidate, y, "expected", {})
    assert _ACTIVE.get() is None
    torch.testing.assert_close(got, out, rtol=0, atol=0)
    out.sum().backward()
    got.sum().backward()
    assert _ACTIVE.get() is None
    for a, b in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    if requires_input_grad:
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
