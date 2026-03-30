import re
from typing import Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from PUMA.model.modules.grounding_sam.world_feature import (
    WorldFeatureLoss,
    WorldFeatureSupervisor,
    WorldQueryHead,
)


class WorldQueryModule(nn.Module):
    def __init__(self, hidden_dim: int, world_cfg: Optional[dict]) -> None:
        super().__init__()
        world_cfg = world_cfg or {}
        self.enabled = bool(world_cfg.get("enabled", False))
        self.query_num = int(world_cfg.get("world_query_num", 0))
        self.loss_weight = float(world_cfg.get("loss_weight", 0.0))
        self.supervision = world_cfg.get("supervision", "per_frame")
        self.feature_loss = world_cfg.get("feature_loss", "cosine")
        self.grounding_mode = world_cfg.get("grounding_mode", "image")
        self.future_view_index = int(world_cfg.get("future_view_index", 0))

        if not self.enabled or self.query_num <= 0 or self.loss_weight <= 0:
            self.enabled = False
            return

        grounding_cfg = world_cfg.get("grounding", None)
        if grounding_cfg is None:
            raise ValueError("world_model.grounding is required when world query is enabled")

        self.supervisor = WorldFeatureSupervisor(
            dino_backbone=world_cfg.get("dino_backbone", "dinov2_vitb14"),
            grounding_cfg=grounding_cfg,
            grounding_mode=self.grounding_mode,
            future_view_index=self.future_view_index,
        )
        self.head = WorldQueryHead(hidden_dim, self.supervisor.feature_dim)
        self.loss_fn = WorldFeatureLoss(self.feature_loss, self.supervision)

    def forward(
        self,
        world_hidden: torch.Tensor,
        future_images: Sequence[Sequence[Sequence[Image.Image]]],
        text_prompts: Sequence[str],
        cache_infos: Optional[Sequence[dict]] = None,
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        target_features = self.supervisor.compute_target_features(
            future_images, text_prompts, cache_infos=cache_infos
        )
        pred_features = self.head(world_hidden)
        loss_raw = self.loss_fn(pred_features, target_features)
        return {
            "loss": loss_raw * self.loss_weight,
            "loss_raw": loss_raw,
            "pred_features": pred_features,
            "target_features": target_features,
        }


def parse_world_text(lang: str) -> str:
    if not lang:
        return ""
    patterns = [
        r"(?:target|object|item|goal|subject)\s*[:=]\s*([^.;,\n]+)",
        r"(?:target|object|item|goal|subject)\s+is\s+([^.;,\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lang, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    quoted = re.findall(r"[\"“”']([^\"“”']+)[\"“”']", lang)
    if quoted:
        return max(quoted, key=len).strip()

    return lang
