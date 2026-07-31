"""Device helpers shared by training, inference and deployment entry points.

Backend-neutral by design: CUDA behavior is preserved bit-for-bit, while NPU
and CPU receive the closest equivalent context. Ascend-specific model
adapters live in `PUMA.model.modules.vlm.ascend`, not here.
"""

from contextlib import nullcontext
from typing import Optional

import torch


def normalize_device_type(device: Optional[object] = None) -> str:
    """Return the bare device type string ("cuda", "npu", "cpu", ...)."""
    if isinstance(device, torch.device):
        return device.type

    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type

    return str(device).split(":", 1)[0] if device is not None else "cpu"


def get_autocast_context(
    device: Optional[object] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    """Return the matching autocast context without changing CUDA behavior."""
    device_type = normalize_device_type(device)
    if dtype is None or device_type == "cpu":
        return nullcontext()
    if dtype == torch.float32 and device_type != "cuda":
        return torch.autocast(device_type=device_type, enabled=False)
    return torch.autocast(device_type=device_type, dtype=dtype)


def resolve_device(device) -> str:
    """Normalize a requested device string and fail fast on unusable NPUs."""
    requested = str(device).lower()
    if torch.device(requested).type == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("--device npu requires torch_npu") from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("--device npu requested, but no NPU is available")
    return requested
