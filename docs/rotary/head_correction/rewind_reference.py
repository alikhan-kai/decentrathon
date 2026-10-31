"""Exact test-only excerpts from Soup rewind_hf.py; production recorder unchanged.
Source: /tmp/decentrathon-soup-cli/soup_cli/trainer/rewind_hf.py
SHA256: 1fe9542e1ab634d1e6f3ce9dc50f28872d0096cd0f7fdd2be344cce10afd5604
"""
from __future__ import annotations
from collections import deque
from typing import Any, Optional, Sequence

class RewindState:
    """FIFO of sampler indices plus step/micro bookkeeping.

    Pure Python -- no torch -- so the bookkeeping is unit-testable without a
    trainer, and so importing this module stays light.

    ``micro`` is derived from the step changing, not from the configured
    gradient-accumulation count (the log header records that, via ``RewindLog``).
    """

    def __init__(self, sink: RewindSink) -> None:
        self._sink = sink
        self._pending: deque[int] = deque()
        self._last_step: Optional[int] = None
        self._micro = -1
        self.failed = False
        self.dropped = 0

    def on_index(self, i: int) -> None:
        """Called by the sampler wrapper for every index it yields."""
        if self.failed:
            # A disabled recorder must not keep growing a FIFO nobody pops.
            return
        self._pending.append(int(i))

    def on_micro_batch(
        self,
        *,
        step: int,
        row_loss: Sequence[float],
        row_tokens: Sequence[int],
    ) -> None:
        """Pop ``len(row_loss)`` ids (oldest first) and write one batch record.

        ``micro`` restarts at 0 whenever ``step`` differs from the last step seen,
        otherwise it increments. If fewer ids are pending than there are rows the
        batch is dropped and counted -- that means the sampler was bypassed (a
        custom ``get_train_dataloader``, say) and guessing would put wrong row ids
        in the log, which is worse than a gap. A dropped micro-batch does not
        advance ``micro``, so the remaining micro-batches of that step are
        renumbered one lower than their true position.
        """
        n = len(row_loss)
        if len(self._pending) < n:
            self.dropped += 1
            return
        rows = [self._pending.popleft() for _ in range(n)]
        if step != self._last_step:
            self._micro = 0
            self._last_step = step
        else:
            self._micro += 1
        self._sink.record_batch(
            step=int(step),
            micro=self._micro,
            rows=rows,
            row_loss=[float(x) for x in row_loss],
            row_tokens=[int(t) for t in row_tokens],
        )

    def disable(self, reason: str) -> None:
        """Disable the recorder after one warning line. Later calls are silent."""
        if self.failed:
            return
        self.failed = True
        self._pending.clear()
        try:
            # escape(): the reason is arbitrary text (an exception message, a
            # path like "[/tmp/x]") and unescaped markup raises MarkupError.
            console.print(f"[yellow]Rewind recorder disabled:[/] {escape(reason)}")
        except Exception:  # a broken console must not stop training
            pass

    def fail(self, exc: BaseException) -> None:
        """Disable the recorder because ``exc`` was raised while recording."""
        self.disable(str(exc))

    def summary(self) -> Optional[str]:
        """One line describing dropped micro-batches, or ``None`` if there were none."""
        if self.dropped <= 0:
            return None
        noun = "micro-batch" if self.dropped == 1 else "micro-batches"
        return f"rewind: {self.dropped} {noun} dropped (FIFO underflow)"

def row_losses_from_token_losses(token_loss: Any, labels: Any) -> tuple[list[float], list[int]]:
    """:func:`row_losses` for a step that has no logits (#1693).

    ``token_loss`` is ``(B, T)``: position ``t`` holds the cross-entropy of
    ``labels[:, t + 1]``, and 0.0 where that label is ``-100`` or there is none.
    """
    import torch

    with torch.no_grad():
        tokens = (labels[:, 1:] != -100).sum(dim=1)
        totals = token_loss.float().sum(dim=1).to(tokens.device)
        means = totals / tokens.clamp(min=1).to(totals.dtype)
    return [float(mean) for mean in means], [int(count) for count in tokens]
