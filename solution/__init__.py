"""Ember: original Triton kernels for diagnostic-preserving QLoRA training."""


def apply():
    """Install idempotent, capability-guarded patches before Soup setup."""
    from .integration import patch_training
    from .rmsnorm import patch_rmsnorm

    patch_rmsnorm()
    patch_training()
