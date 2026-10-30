"""Original split-vocabulary statistics and first-order logit gradients.

``vocab_stats(logits, targets, ignore_index=-100)`` returns float32 token loss,
entropy, int64 predictions, and float32 logsumexp, each shaped ``[N]``.
``vocab_backward(logits, targets, lse, scale, ignore_index=-100)`` returns
``scale * (softmax(logits.float()) - one_hot(targets))`` in the logits dtype.
Ignored rows have zero loss/gradient, but still have entropy and predictions.

Supported inputs are finite fp16/fp32 logits [N, V], V >= 1, and int32/int64
targets [N] containing ignore_index or indices in [0, V). NaNs and infinities
are unsupported. CUDA label checks are asynchronous device assertions; target
loads are additionally clamped/masked so an invalid label cannot read outside
the tensor. CPU invalid labels raise IndexError. Inputs may be noncontiguous.
``lse`` must be the matching float32 result of vocab_stats (or logsumexp), and
``scale`` a floating scalar tensor. Scaling happens in fp32 before casting.

These are first-order primitives, not differentiable autograd operations.
An integrating autograd.Function must call backward explicitly and reject
double backward. Torch is imported lazily; CPU references work without Triton.
CUDA always uses the original Triton kernels below, with 2048-entry tiles and
O(N * ceil(V / 2048)) scratch, never one full-vocabulary row block.
"""

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_VOCAB_TILE = 2048


if triton is not None:

    @triton.jit
    def ember_vocab_partials(
        Logits, Maxima, Sums, Moments, Winners,
        V: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        part = tl.program_id(1)
        columns = part * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            Logits + row.to(tl.int64) * V + columns,
            columns < V, other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(values, 0)
        centered = values - maximum
        weights = tl.exp(centered)
        mass = tl.sum(weights, 0)
        # Avoid 0 * -inf at padded lanes and at underflowed extreme logits.
        moment = tl.sum(weights * tl.where(weights > 0, centered, 0.0), 0)
        winner = tl.min(tl.where((columns < V) & (values == maximum), columns, 2147483647), 0)
        slot = row.to(tl.int64) * SPLITS + part
        tl.store(Maxima + slot, maximum)
        tl.store(Sums + slot, mass)
        tl.store(Moments + slot, moment)
        tl.store(Winners + slot, winner)

    @triton.jit
    def ember_vocab_merge(
        Logits, Targets, Maxima, Sums, Moments, Winners,
        Loss, Entropy, Predictions, Lse,
        V: tl.constexpr, SPLITS: tl.constexpr, IGNORE: tl.constexpr,
        REDUCE: tl.constexpr,
    ):
        row = tl.program_id(0)
        parts = tl.arange(0, REDUCE)
        slot = row.to(tl.int64) * SPLITS + parts
        maxima = tl.load(Maxima + slot, parts < SPLITS, other=-float("inf"))
        masses = tl.load(Sums + slot, parts < SPLITS, other=0.0)
        moments = tl.load(Moments + slot, parts < SPLITS, other=0.0)
        winners = tl.load(Winners + slot, parts < SPLITS, other=2147483647)
        maximum = tl.max(maxima, 0)
        shifts = maxima - maximum
        weights = tl.exp(shifts)
        safe_shifts = tl.where(weights > 0, shifts, 0.0)
        mass = tl.sum(weights * masses, 0)
        moment = tl.sum(weights * (moments + safe_shifts * masses), 0)
        log_mass = tl.log(mass)
        prediction = tl.min(tl.where(maxima == maximum, winners, 2147483647), 0)
        target = tl.load(Targets + row).to(tl.int64)
        ignored = target == IGNORE
        valid = (target >= 0) & (target < V)
        tl.device_assert(ignored | valid, "vocab target must be ignore_index or in [0, V)")
        safe_target = tl.minimum(tl.maximum(target, 0), V - 1)
        target_value = tl.load(
            Logits + row.to(tl.int64) * V + safe_target, valid & ~ignored, other=0.0,
        ).to(tl.float32)
        loss = tl.where(ignored, 0.0, (maximum - target_value) + log_mass)
        tl.store(Loss + row, loss)
        tl.store(Entropy + row, log_mass - moment / mass)
        tl.store(Predictions + row, prediction.to(tl.int64))
        tl.store(Lse + row, maximum + log_mass)

    @triton.jit
    def ember_vocab_backward_mass(
        Logits, Lse, Masses,
        V: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        part = tl.program_id(1)
        columns = part * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            Logits + row.to(tl.int64) * V + columns,
            columns < V, other=-float("inf"),
        ).to(tl.float32)
        shift = tl.load(Lse + row)
        mass = tl.sum(tl.exp(values - shift), 0)
        tl.store(Masses + row.to(tl.int64) * SPLITS + part, mass)

    @triton.jit
    def ember_vocab_backward_write(
        Logits, Targets, Lse, Scale, Masses, Gradient,
        V: tl.constexpr, SPLITS: tl.constexpr, IGNORE: tl.constexpr,
        BLOCK: tl.constexpr, REDUCE: tl.constexpr,
    ):
        row = tl.program_id(0)
        part = tl.program_id(1)
        parts = tl.arange(0, REDUCE)
        partial_mass = tl.load(
            Masses + row.to(tl.int64) * SPLITS + parts, parts < SPLITS, other=0.0,
        )
        mass = tl.sum(partial_mass, 0)
        columns = part * BLOCK + tl.arange(0, BLOCK)
        offset = row.to(tl.int64) * V + columns
        values = tl.load(Logits + offset, columns < V, other=-float("inf")).to(tl.float32)
        shift = tl.load(Lse + row)
        target = tl.load(Targets + row).to(tl.int64)
        ignored = target == IGNORE
        tl.device_assert(ignored | ((target >= 0) & (target < V)),
                         "vocab target must be ignore_index or in [0, V)")
        probability = tl.exp(values - shift) / mass
        gradient = (probability - (columns == target).to(tl.float32)) * tl.load(Scale).to(tl.float32)
        gradient = tl.where(ignored, 0.0, gradient)
        # Store performs the only narrowing conversion, after upstream scaling.
        tl.store(Gradient + offset, gradient, columns < V)


def _validate_inputs(logits, targets, ignore_index):
    import torch

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be tensors")
    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [N, V] with V >= 1")
    if logits.shape[1] > 2147483647:
        raise ValueError("V must fit a signed int32 vocabulary index")
    if logits.dtype not in (torch.float16, torch.float32):
        raise TypeError("logits must be fp16 or fp32")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [N]")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("targets must be int32 or int64")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    if logits.device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU references and CUDA kernels are supported")
    if not isinstance(ignore_index, int) or not -(2**63) <= ignore_index < 2**63:
        raise ValueError("ignore_index must be a signed int64 integer")
    if logits.device.type == "cpu":
        valid = (targets == ignore_index) | ((targets >= 0) & (targets < logits.shape[1]))
        if not bool(valid.all()):
            raise IndexError("vocab target must be ignore_index or in [0, V)")


def _validate_backward(logits, targets, lse, scale, ignore_index):
    import torch

    _validate_inputs(logits, targets, ignore_index)
    if not isinstance(lse, torch.Tensor) or lse.shape != targets.shape:
        raise ValueError("lse must have shape [N]")
    if lse.dtype != torch.float32 or lse.device != logits.device:
        raise ValueError("lse must be fp32 on the logits device")
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1 or not scale.is_floating_point():
        raise ValueError("scale must be a floating scalar tensor")


def vocab_stats_reference(logits, targets, ignore_index=-100):
    """CPU fp32 reference; no autograd graph is constructed."""
    import torch

    _validate_inputs(logits, targets, ignore_index)
    if logits.device.type != "cpu":
        raise ValueError("vocab_stats_reference requires CPU tensors")
    with torch.no_grad():
        values = logits.float()
        maxima = values.amax(dim=-1, keepdim=True)
        centered = values - maxima
        log_mass = torch.logsumexp(centered, dim=-1, keepdim=True)
        log_probabilities = centered - log_mass
        probabilities = log_probabilities.exp()
        entropy = -torch.where(probabilities > 0, probabilities * log_probabilities, 0.0).sum(dim=-1)
        indices = targets.clamp(0, logits.shape[1] - 1).long().unsqueeze(-1)
        loss = -log_probabilities.gather(1, indices).squeeze(-1)
        loss = torch.where(targets == ignore_index, 0.0, loss)
        predictions = values.argmax(dim=-1)
        lse = (maxima + log_mass).squeeze(-1)
        return loss, entropy, predictions, lse


def vocab_backward_reference(logits, targets, lse, scale, ignore_index=-100):
    """CPU first-order reference; scale before converting back to logits dtype."""
    import torch

    _validate_backward(logits, targets, lse, scale, ignore_index)
    if logits.device.type != "cpu":
        raise ValueError("vocab_backward_reference requires CPU tensors")
    with torch.no_grad():
        gradient = torch.softmax(logits.float(), dim=-1)
        indices = targets.clamp(0, logits.shape[1] - 1).long().unsqueeze(-1)
        gradient.scatter_add_(1, indices, -torch.ones_like(indices, dtype=torch.float32))
        gradient *= scale.to(device=logits.device, dtype=torch.float32)
        gradient.masked_fill_((targets == ignore_index).unsqueeze(-1), 0.0)
        return gradient.to(logits.dtype)


def _require_triton():
    if triton is None:
        raise RuntimeError("CUDA vocab kernels require Triton; CPU references do not")


def vocab_stats(logits, targets, ignore_index=-100):
    """Compute (token_loss, entropy, lowest-index argmax, lse), each [N]."""
    import torch

    _validate_inputs(logits, targets, ignore_index)
    if logits.device.type == "cpu":
        return vocab_stats_reference(logits, targets, ignore_index)
    _require_triton()
    with torch.no_grad(), torch.cuda.device(logits.device):
        logits = logits.contiguous()
        targets = targets.contiguous()
        rows, vocab = logits.shape
        loss = torch.empty((rows,), dtype=torch.float32, device=logits.device)
        entropy = torch.empty_like(loss)
        predictions = torch.empty((rows,), dtype=torch.int64, device=logits.device)
        lse = torch.empty_like(loss)
        if rows:
            splits = triton.cdiv(vocab, _VOCAB_TILE)
            maxima = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
            masses = torch.empty_like(maxima)
            moments = torch.empty_like(maxima)
            winners = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
            ember_vocab_partials[(rows, splits)](
                logits, maxima, masses, moments, winners,
                vocab, splits, _VOCAB_TILE, num_warps=4,
            )
            ember_vocab_merge[(rows,)](
                logits, targets, maxima, masses, moments, winners,
                loss, entropy, predictions, lse,
                vocab, splits, ignore_index, triton.next_power_of_2(splits),
                num_warps=4, debug=True,
            )
        return loss, entropy, predictions, lse


def vocab_backward(logits, targets, lse, scale, ignore_index=-100):
    """Return scaled d(logit) in logits.dtype, with ignored rows exactly zero.

    Renormalizing exp(logits - lse) compensates for rounding in stored fp32 lse
    at large common offsets. This costs one extra logit read, without saving
    dense probabilities or synchronizing labels/scale through the host.
    """
    import torch

    _validate_backward(logits, targets, lse, scale, ignore_index)
    if logits.device.type == "cpu":
        return vocab_backward_reference(logits, targets, lse, scale, ignore_index)
    _require_triton()
    with torch.no_grad(), torch.cuda.device(logits.device):
        logits = logits.contiguous()
        targets = targets.contiguous()
        lse = lse.contiguous()
        scale = scale.to(device=logits.device, dtype=torch.float32).contiguous()
        rows, vocab = logits.shape
        gradient = torch.empty_like(logits)
        if rows:
            splits = triton.cdiv(vocab, _VOCAB_TILE)
            masses = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
            ember_vocab_backward_mass[(rows, splits)](
                logits, lse, masses, vocab, splits, _VOCAB_TILE, num_warps=4,
            )
            ember_vocab_backward_write[(rows, splits)](
                logits, targets, lse, scale, masses, gradient,
                vocab, splits, ignore_index, _VOCAB_TILE, triton.next_power_of_2(splits),
                num_warps=4, debug=True,
            )
        return gradient
