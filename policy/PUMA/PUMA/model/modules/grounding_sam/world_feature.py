import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from PUMA.model.modules.dino_model.dino import get_dino_model
from PUMA.model.modules.grounding_sam.grounders import (
    GroundedSAM2ImageGrounder,
    GroundedSAM2VideoGrounder,
)


class WorldQueryHead(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, world_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(world_hidden)


class WorldFeatureLoss(nn.Module):
    def __init__(self, loss_type: str = "cosine", supervision: str = "per_frame") -> None:
        super().__init__()
        self.loss_type = loss_type
        self.supervision = supervision

    def forward(self, pred_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        pred = pred_features.float()
        target = target_features.float()
        if self.supervision == "per_frame":
            if pred.shape[1] != target.shape[1]:
                raise ValueError(
                    f"world_query_num {pred.shape[1]} must match future_k {target.shape[1]}"
                )
            pred_flat = pred.reshape(-1, pred.shape[-1])
            target_flat = target.reshape(-1, target.shape[-1])
        elif self.supervision == "fused":
            pred_flat = pred.mean(dim=1)
            target_flat = target.mean(dim=1)
        else:
            raise ValueError(f"Unknown world supervision: {self.supervision}")

        if self.loss_type == "cosine":
            return (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)).mean()
        if self.loss_type == "mse":
            return F.mse_loss(pred_flat, target_flat)
        raise ValueError(f"Unknown world feature loss: {self.loss_type}")


class WorldFeatureSupervisor(nn.Module):
    def __init__(
        self,
        dino_backbone: str,
        grounding_cfg: dict,
        grounding_mode: str = "image",
        future_view_index: int = 0,
    ) -> None:
        super().__init__()
        self.future_view_index = future_view_index
        self.grounding_mode = grounding_mode
        self.dino = get_dino_model(dino_backbone)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.image_grounder = GroundedSAM2ImageGrounder(
            sam2_model_config=grounding_cfg["sam2_model_config"],
            sam2_checkpoint=grounding_cfg["sam2_checkpoint"],
            grounding_dino_config=grounding_cfg["grounding_dino_config"],
            grounding_dino_checkpoint=grounding_cfg["grounding_dino_checkpoint"],
            box_threshold=grounding_cfg.get("box_threshold", 0.35),
            text_threshold=grounding_cfg.get("text_threshold", 0.25),
            multimask_output=grounding_cfg.get("multimask_output", False),
            max_boxes=grounding_cfg.get("max_boxes", 1),
            device=grounding_cfg.get("device", None),
        )
        self.video_grounder = None
        if grounding_mode == "video":
            self.video_grounder = GroundedSAM2VideoGrounder(
                image_grounder=self.image_grounder,
                sam2_model_config=grounding_cfg["sam2_model_config"],
                sam2_checkpoint=grounding_cfg["sam2_checkpoint"],
                prompt_type=grounding_cfg.get("video_prompt", "mask"),
                device=grounding_cfg.get("device", None),
            )
        self._freeze_module(self.image_grounder.grounding_model)
        self._freeze_module(getattr(self.image_grounder.sam2_predictor, "model", None))
        if self.video_grounder is not None:
            self._freeze_module(self.video_grounder.video_predictor)

        cache_cfg = grounding_cfg.get("cache", {}) if isinstance(grounding_cfg, dict) else {}
        self.cache_enabled = bool(cache_cfg.get("enabled", True))
        self.cache_read = bool(cache_cfg.get("read", True))
        self.cache_write = bool(cache_cfg.get("write", True))
        self.cache_dirname = cache_cfg.get("dirname", grounding_cfg.get("cache_dirname", "grounding_cache"))
        self.cache_signature = self._build_cache_signature(grounding_cfg, grounding_mode, cache_cfg)
        self._usage_reported = {"cache": False, "online": False}

    @staticmethod
    def _freeze_module(module: Optional[nn.Module]) -> None:
        if module is None:
            return
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _report_usage(self, mode: str, reason: str) -> None:
        if self._usage_reported.get(mode):
            return
        if mode == "cache":
            print(f"[WorldFeatureSupervisor] using cached mask/box ({reason})")
        else:
            print(f"[WorldFeatureSupervisor] computing mask/box online ({reason})")
        self._usage_reported[mode] = True

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_cache_signature(
        grounding_cfg: dict, grounding_mode: str, cache_cfg: Optional[dict] = None
    ) -> str:
        cache_cfg = cache_cfg or {}
        identity = {
            "cache_version": cache_cfg.get("version", grounding_cfg.get("cache_version", "v1")),
            "grounding_mode": grounding_mode,
            "box_threshold": grounding_cfg.get("box_threshold", 0.35),
            "text_threshold": grounding_cfg.get("text_threshold", 0.25),
            "multimask_output": grounding_cfg.get("multimask_output", False),
            "max_boxes": grounding_cfg.get("max_boxes", 1),
            "video_prompt": grounding_cfg.get("video_prompt", "mask"),
            "sam2_model_config": grounding_cfg.get("sam2_model_config", ""),
            "sam2_checkpoint": grounding_cfg.get("sam2_checkpoint", ""),
            "grounding_dino_config": grounding_cfg.get("grounding_dino_config", ""),
            "grounding_dino_checkpoint": grounding_cfg.get("grounding_dino_checkpoint", ""),
        }
        payload = json.dumps(identity, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _normalize_cache_root(self, cache_root: Optional[Union[str, Path]]) -> Optional[Path]:
        if not cache_root:
            return None
        return Path(cache_root)

    def _make_cache_path(self, cache_root: Path, frame_key: str, prompt: str) -> Path:
        cache_id = self._hash_text(f"{self.cache_signature}|{frame_key}|{prompt}")
        return cache_root / self.cache_dirname / cache_id[:2] / f"{cache_id}.npz"

    @staticmethod
    def _mask_to_numpy(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.detach().to("cpu")
            if mask_tensor.dtype != torch.uint8:
                mask_tensor = (mask_tensor > 0).to(torch.uint8)
            return mask_tensor.numpy()
        mask_np = np.asarray(mask)
        if mask_np.dtype != np.uint8:
            mask_np = (mask_np > 0).astype(np.uint8)
        return mask_np

    @staticmethod
    def _box_to_numpy(box: Optional[Union[torch.Tensor, np.ndarray]]) -> np.ndarray:
        if box is None:
            return np.zeros((0,), dtype=np.float32)
        if isinstance(box, torch.Tensor):
            box_np = box.detach().to("cpu").numpy()
        else:
            box_np = np.asarray(box)
        return box_np.astype(np.float32)

    def _load_cache(self, cache_path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                mask = data["mask"]
                box = data["box"] if "box" in data else None
            if box is not None and np.asarray(box).size == 0:
                box = None
            return mask, box
        except Exception:
            return None

    def _save_cache(
        self,
        cache_path: Path,
        mask: Union[torch.Tensor, np.ndarray],
        box: Optional[Union[torch.Tensor, np.ndarray]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mask_np = self._mask_to_numpy(mask)
        box_np = self._box_to_numpy(box)
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as tmp_file:
            np.savez(tmp_file, mask=mask_np, box=box_np)
            tmp_name = tmp_file.name
        os.replace(tmp_name, cache_path)

    def _get_mask_with_cache(
        self,
        frame: Image.Image,
        prompt: str,
        cache_root: Optional[Path],
        frame_key: Optional[str],
    ) -> Union[torch.Tensor, np.ndarray]:
        if not self.cache_enabled or cache_root is None or not frame_key:
            self._report_usage("online", "cache disabled or missing key")
            return self.image_grounder.predict_mask_and_box(frame, prompt)[0]

        cache_path = self._make_cache_path(cache_root, frame_key, prompt)
        if self.cache_read and cache_path.exists():
            cached = self._load_cache(cache_path)
            if cached is not None:
                self._report_usage("cache", "cache hit")
                return cached[0]

        self._report_usage("online", "cache miss")
        mask, box = self.image_grounder.predict_mask_and_box(frame, prompt)
        if self.cache_write:
            self._save_cache(cache_path, mask, box)
        return mask

    def _get_video_masks_with_cache(
        self,
        frames: Sequence[Image.Image],
        prompt: str,
        cache_root: Optional[Path],
        frame_keys: Sequence[Optional[str]],
    ) -> List[Union[torch.Tensor, np.ndarray]]:
        if not self.cache_enabled or cache_root is None or not frame_keys:
            self._report_usage("online", "cache disabled or missing key")
            return self.video_grounder.predict_video_masks(frames, prompt)

        masks_out: List[Optional[Union[torch.Tensor, np.ndarray]]] = []
        missing = []
        for idx, frame_key in enumerate(frame_keys):
            if not frame_key:
                masks_out.append(None)
                missing.append(idx)
                continue
            cache_path = self._make_cache_path(cache_root, frame_key, prompt)
            if self.cache_read and cache_path.exists():
                cached = self._load_cache(cache_path)
                if cached is not None:
                    self._report_usage("cache", "cache hit")
                    masks_out.append(cached[0])
                    continue
            masks_out.append(None)
            missing.append(idx)

        if missing:
            self._report_usage("online", "cache miss")
            computed = self.video_grounder.predict_video_masks(frames, prompt)
            for idx in missing:
                mask = computed[idx]
                masks_out[idx] = mask
                frame_key = frame_keys[idx]
                if self.cache_write and frame_key:
                    cache_path = self._make_cache_path(cache_root, frame_key, prompt)
                    self._save_cache(cache_path, mask, None)

        if any(mask is None for mask in masks_out):
            width, height = frames[0].size
            empty = np.zeros((height, width), dtype=np.uint8)
            return [mask if mask is not None else empty for mask in masks_out]
        return [mask for mask in masks_out]

    @property
    def feature_dim(self) -> int:
        return self.dino.num_channels

    def _select_future_frames(
        self,
        future_images: Sequence[Sequence[Sequence[Image.Image]]],
        cache_infos: Optional[Sequence[dict]] = None,
    ) -> Tuple[List[List[Image.Image]], List[List[Optional[str]]], List[Optional[Path]]]:
        selected_frames: List[List[Image.Image]] = []
        selected_keys: List[List[Optional[str]]] = []
        cache_roots: List[Optional[Path]] = []
        for sample_idx, sample in enumerate(future_images):
            frames: List[Image.Image] = []
            keys: List[Optional[str]] = []
            sample_keys = None
            cache_root = None
            if cache_infos is not None and sample_idx < len(cache_infos):
                cache_info = cache_infos[sample_idx]
                if isinstance(cache_info, dict):
                    cache_root = self._normalize_cache_root(
                        cache_info.get("dataset_path") or cache_info.get("cache_root")
                    )
                    sample_keys = cache_info.get("future_frame_keys")

            for step_idx, step in enumerate(sample):
                if not step:
                    raise ValueError("Future frame views are empty")
                view_index = self.future_view_index if self.future_view_index < len(step) else 0
                frames.append(step[view_index])

                key = None
                if sample_keys is not None and step_idx < len(sample_keys):
                    step_keys = sample_keys[step_idx]
                    if isinstance(step_keys, (list, tuple)):
                        if view_index < len(step_keys):
                            key = step_keys[view_index]
                        elif len(step_keys) > 0:
                            key = step_keys[0]
                    else:
                        key = step_keys
                keys.append(key)

            selected_frames.append(frames)
            selected_keys.append(keys)
            cache_roots.append(cache_root)

        return selected_frames, selected_keys, cache_roots

    @torch.no_grad()
    def compute_target_features(
        self,
        future_images: Sequence[Sequence[Sequence[Image.Image]]],
        text_prompts: Sequence[str],
        cache_infos: Optional[Sequence[dict]] = None,
    ) -> torch.Tensor:
        if len(future_images) == 0:
            raise ValueError("future_images is empty")
        frames_per_sample, frame_keys_per_sample, cache_roots = self._select_future_frames(
            future_images, cache_infos
        )
        if len(frames_per_sample) != len(text_prompts):
            raise ValueError("future_images and text_prompts must align in batch size")

        all_masks: List[Union[torch.Tensor, np.ndarray]] = []
        for idx, (frames, prompt) in enumerate(zip(frames_per_sample, text_prompts)):
            prompt_text = (prompt or "").strip()
            frame_keys = frame_keys_per_sample[idx] if idx < len(frame_keys_per_sample) else []
            cache_root = cache_roots[idx] if idx < len(cache_roots) else None
            if len(frame_keys) < len(frames):
                frame_keys = list(frame_keys) + [None] * (len(frames) - len(frame_keys))
            if self.grounding_mode == "video" and self.video_grounder is not None:
                masks = self._get_video_masks_with_cache(
                    frames=frames,
                    prompt=prompt_text,
                    cache_root=cache_root,
                    frame_keys=frame_keys,
                )
            else:
                masks = [
                    self._get_mask_with_cache(frame, prompt_text, cache_root, frame_key)
                    for frame, frame_key in zip(frames, frame_keys)
                ]
            all_masks.extend(masks)

        flat_frames = [[frame] for frames in frames_per_sample for frame in frames]
        dino_input = self.dino.prepare_dino_input(flat_frames)
        patch_tokens = self.dino(dino_input)
        num_tokens = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Unexpected DINO token count: {num_tokens}")

        device = patch_tokens.device
        processed_masks: List[torch.Tensor] = []
        for mask in all_masks:
            mask_tensor = mask.to(device) if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=device)
            # Normalize to a single 2D mask (H, W).
            if mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.reshape(-1, *mask_tensor.shape[-2:]).max(dim=0)[0]
            if mask_tensor.dim() != 2:
                raise ValueError(f"Unexpected mask shape for pooling: {tuple(mask_tensor.shape)}")
            processed_masks.append(mask_tensor)

        mask_tensor = torch.stack(processed_masks, dim=0).unsqueeze(1).float()
        mask_tensor = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest")
        mask_flat = mask_tensor.flatten(2).squeeze(1)
        mask_sum = mask_flat.sum(dim=1, keepdim=True)
        mask_flat = torch.where(mask_sum > 0, mask_flat, torch.ones_like(mask_flat))
        mask_sum = mask_flat.sum(dim=1, keepdim=True)

        masked_tokens = patch_tokens * mask_flat.unsqueeze(-1)
        pooled = masked_tokens.sum(dim=1) / mask_sum

        batch_size = len(frames_per_sample)
        future_k = len(frames_per_sample[0])
        pooled = pooled.view(batch_size, future_k, -1)
        return pooled

    @torch.no_grad()
    def precompute_mask_cache(
        self,
        future_images: Sequence[Sequence[Sequence[Image.Image]]],
        text_prompts: Sequence[str],
        cache_infos: Optional[Sequence[dict]] = None,
        skip_existing: bool = True,
        validate_existing: bool = False,
    ) -> dict:
        if len(future_images) == 0:
            raise ValueError("future_images is empty")
        frames_per_sample, frame_keys_per_sample, cache_roots = self._select_future_frames(
            future_images, cache_infos
        )
        if len(frames_per_sample) != len(text_prompts):
            raise ValueError("future_images and text_prompts must align in batch size")

        stats = {"total": 0, "cached": 0, "computed": 0}
        for idx, (frames, prompt) in enumerate(zip(frames_per_sample, text_prompts)):
            prompt_text = (prompt or "").strip()
            frame_keys = frame_keys_per_sample[idx] if idx < len(frame_keys_per_sample) else []
            cache_root = cache_roots[idx] if idx < len(cache_roots) else None
            if len(frame_keys) < len(frames):
                frame_keys = list(frame_keys) + [None] * (len(frames) - len(frame_keys))

            stats["total"] += len(frames)
            if self.grounding_mode == "video" and self.video_grounder is not None:
                missing = []
                for frame_idx, frame_key in enumerate(frame_keys):
                    if not self.cache_enabled or cache_root is None or not frame_key:
                        missing.append(frame_idx)
                        continue
                    cache_path = self._make_cache_path(cache_root, frame_key, prompt_text)
                    if self.cache_read and cache_path.exists() and skip_existing:
                        if validate_existing:
                            cached = self._load_cache(cache_path)
                            if cached is not None:
                                stats["cached"] += 1
                                continue
                        else:
                            stats["cached"] += 1
                            continue
                    missing.append(frame_idx)

                if missing:
                    self._get_video_masks_with_cache(
                        frames=frames,
                        prompt=prompt_text,
                        cache_root=cache_root,
                        frame_keys=frame_keys,
                    )
                    stats["computed"] += len(missing)
                continue

            for frame, frame_key in zip(frames, frame_keys):
                if not self.cache_enabled or cache_root is None or not frame_key:
                    _ = self.image_grounder.predict_mask_and_box(frame, prompt_text)
                    stats["computed"] += 1
                    continue

                cache_path = self._make_cache_path(cache_root, frame_key, prompt_text)
                if self.cache_read and cache_path.exists() and skip_existing:
                    if validate_existing:
                        cached = self._load_cache(cache_path)
                        if cached is not None:
                            stats["cached"] += 1
                            continue
                    else:
                        stats["cached"] += 1
                        continue

                _ = self._get_mask_with_cache(frame, prompt_text, cache_root, frame_key)
                stats["computed"] += 1

        return stats
