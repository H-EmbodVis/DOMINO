"""Ascend-compatible Qwen vision patch embedding helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LinearizedConv3dProjection(nn.Module):
    """Conv3d patch projection expressed as a linear layer with Conv3d-shaped weights."""

    def __init__(self, conv3d: nn.Conv3d):
        super().__init__()
        if not isinstance(conv3d, nn.Conv3d):
            raise TypeError(f"Expected nn.Conv3d, got {type(conv3d)!r}")
        self.weight = nn.Parameter(conv3d.weight.detach().clone())
        if conv3d.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(conv3d.bias.detach().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat_input = hidden_states.reshape(hidden_states.shape[0], -1)
        flat_weight = self.weight.reshape(self.weight.shape[0], -1)
        return F.linear(flat_input.to(dtype=flat_weight.dtype), flat_weight, self.bias)


class LinearizedConv3dPatchEmbed(nn.Module):
    """Drop-in replacement for Qwen Conv3d patch embed when kernel size equals stride."""

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
        proj: LinearizedConv3dProjection,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = proj

    @classmethod
    def from_patch_embed(cls, patch_embed: nn.Module) -> "LinearizedConv3dPatchEmbed":
        proj = getattr(patch_embed, "proj", None)
        if not isinstance(proj, nn.Conv3d):
            raise TypeError("Qwen vision patch_embed.proj must be nn.Conv3d")
        kernel_size = tuple(proj.kernel_size)
        stride = tuple(proj.stride)
        if kernel_size != stride:
            raise ValueError(f"Linearized patch embedding requires kernel_size == stride, got {kernel_size} and {stride}")
        return cls(
            patch_size=int(getattr(patch_embed, "patch_size")),
            temporal_patch_size=int(getattr(patch_embed, "temporal_patch_size")),
            in_channels=int(getattr(patch_embed, "in_channels")),
            embed_dim=int(getattr(patch_embed, "embed_dim")),
            proj=LinearizedConv3dProjection(proj),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states


def _get_nested_attr(root: Any, dotted_path: str) -> Any:
    current = root
    for attr in dotted_path.split("."):
        current = getattr(current, attr)
    return current


def linearize_qwen_vision_patch_embed(model: nn.Module) -> bool:
    """Replace Qwen's nested vision patch Conv3d with an equivalent linearized module."""

    try:
        visual = _get_nested_attr(model, "model.visual")
    except AttributeError:
        return False

    patch_embed = getattr(visual, "patch_embed", None)
    if isinstance(patch_embed, LinearizedConv3dPatchEmbed):
        return False
    if patch_embed is None or not isinstance(getattr(patch_embed, "proj", None), nn.Conv3d):
        return False

    visual.patch_embed = LinearizedConv3dPatchEmbed.from_patch_embed(patch_embed)
    return True
