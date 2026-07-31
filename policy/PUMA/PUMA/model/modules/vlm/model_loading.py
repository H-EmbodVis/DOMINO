"""VLM loading helpers shared by Qwen wrappers."""

from __future__ import annotations

import torch


def resolve_torch_dtype(value):
    """Normalize config dtype strings for Hugging Face model loading."""

    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"auto", ""}:
        return "auto"
    if normalized in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "torch.float16"}:
        return torch.float16

    raise ValueError(f"Unsupported torch dtype value: {value!r}")
