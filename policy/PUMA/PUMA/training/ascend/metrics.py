"""NPU-aware training metric helpers that keep CUDA collection unchanged."""

from typing import Optional

import torch

from PUMA.util.device import normalize_device_type


def collect_training_metrics(metrics: dict, device: Optional[object] = None) -> dict:
    """Avoid per-step device synchronization for NPU scalar metrics."""
    if normalize_device_type(device) == "npu":
        return {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in metrics.items()
        }
    return {
        name: value.item() if isinstance(value, torch.Tensor) else value
        for name, value in metrics.items()
    }


def materialize_training_metrics(metrics: dict) -> dict:
    """Convert deferred scalar tensors only when a metric record is emitted."""
    return {
        name: value.detach().float().item()
        if isinstance(value, torch.Tensor) and value.numel() == 1
        else value
        for name, value in metrics.items()
    }


def raise_for_non_finite_loss(loss: torch.Tensor, step: Optional[int] = None):
    if torch.isfinite(loss.detach()).all():
        return

    step_label = "unknown" if step is None else step
    loss_value = loss.detach().float().item() if loss.numel() == 1 else loss.detach().float()
    raise FloatingPointError(f"Non-finite training loss at step {step_label}: {loss_value}")


def should_check_non_finite_loss(
    step: Optional[int],
    interval: Optional[int] = 1,
    warmup_steps: Optional[int] = 0,
) -> bool:
    if interval is None:
        return True

    interval = int(interval)
    if interval <= 1:
        return True

    step = 0 if step is None else int(step)
    warmup_steps = 0 if warmup_steps is None else int(warmup_steps)
    if step < warmup_steps:
        return True

    return step % interval == 0
