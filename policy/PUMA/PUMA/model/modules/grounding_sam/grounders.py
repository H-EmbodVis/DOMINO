import contextlib
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


class GroundedSAM2ImageGrounder:
    def __init__(
        self,
        sam2_model_config: str,
        sam2_checkpoint: str,
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        multimask_output: bool = False,
        max_boxes: int = 1,
        device: Optional[str] = None,
    ) -> None:
        try:
            from grounding_dino.groundingdino.util.inference import (
                load_model,
                predict,
                preprocess_caption,
            )
            import grounding_dino.groundingdino.datasets.transforms as T
        except Exception as exc:
            raise ImportError("GroundingDINO from Grounded-SAM-2 is required") from exc

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError("SAM2 is required for GroundedSAM2ImageGrounder") from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.multimask_output = multimask_output
        self.max_boxes = max_boxes
        self._predict = predict
        self._preprocess_caption = preprocess_caption
        self._gdino_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.grounding_model = load_model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device,
        )
        sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _prepare_gdino_image(self, image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
        image_source = np.array(image.convert("RGB"), copy=True)
        image_tensor, _ = self._gdino_transform(Image.fromarray(image_source), None)
        return image_source, image_tensor

    def _predict_fp16(
        self, image_tensor: torch.Tensor, caption: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str]]:
        original_dtype = image_tensor.dtype
        device_str = str(self.device)
        use_cuda = "cuda" in device_str and torch.cuda.is_available()
        if use_cuda:
            # Force fp16 to avoid bf16 in ms_deform_attn.
            image_tensor = image_tensor.to(dtype=torch.float16)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            boxes, scores, phrases = self._predict(
                model=self.grounding_model,
                image=image_tensor,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        if boxes is not None and boxes.dtype != original_dtype:
            boxes = boxes.to(dtype=original_dtype)
        if scores is not None and scores.dtype != original_dtype:
            scores = scores.to(dtype=original_dtype)
        return boxes, scores, phrases

    @torch.no_grad()
    def predict_mask_and_box(
        self, image: Image.Image, text_prompt: str
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        image_source, image_tensor = self._prepare_gdino_image(image)
        caption = self._preprocess_caption(text_prompt)
        boxes, scores, _ = self._predict_fp16(image_tensor, caption)
        height, width = image_source.shape[:2]
        if boxes is None or len(boxes) == 0:
            empty = torch.zeros((height, width), dtype=torch.float32, device=self.device)
            return empty, None

        order = torch.argsort(scores, descending=True)[: max(1, self.max_boxes)]
        boxes = boxes.index_select(0, order)

        boxes = boxes * boxes.new_tensor([width, height, width, height])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        self.sam2_predictor.set_image(image_source)
        masks, mask_scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=self.multimask_output,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if self.multimask_output and mask_scores is not None:
            best = mask_scores.argmax(axis=1)
            masks = np.stack([masks[i, best[i]] for i in range(len(best))], axis=0)

        if masks.shape[0] > 1:
            mask = np.any(masks, axis=0).astype(np.float32)
        else:
            mask = masks[0].astype(np.float32)
        return torch.from_numpy(mask).to(device=self.device), input_boxes[0].detach().cpu()


class GroundedSAM2VideoGrounder:
    def __init__(
        self,
        image_grounder: GroundedSAM2ImageGrounder,
        sam2_model_config: str,
        sam2_checkpoint: str,
        prompt_type: str = "mask",
        device: Optional[str] = None,
    ) -> None:
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except Exception as exc:
            raise ImportError("SAM2 is required for GroundedSAM2VideoGrounder") from exc

        if device is None:
            device = image_grounder.device
        self.device = device
        self.image_grounder = image_grounder
        self.prompt_type = prompt_type
        self.video_predictor = build_sam2_video_predictor(sam2_model_config, sam2_checkpoint, device=device)

    @torch.no_grad()
    def predict_video_masks(
        self, frames: Sequence[Image.Image], text_prompt: str
    ) -> List[torch.Tensor]:
        if len(frames) == 0:
            return []
        init_mask, init_box = self.image_grounder.predict_mask_and_box(frames[0], text_prompt)
        if init_mask.sum() == 0:
            return [init_mask for _ in range(len(frames))]

        inference_state = self.video_predictor.init_state()
        for frame in frames:
            self.video_predictor.add_new_frame(
                inference_state, np.array(frame.convert("RGB"), copy=True)
            )

        obj_id = 1
        if self.prompt_type == "box" and init_box is not None:
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=init_box,
            )
        else:
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=init_mask,
            )

        masks_out: List[torch.Tensor] = [init_mask for _ in range(len(frames))]
        for frame_idx, _, video_res_masks in self.video_predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=len(frames),
        ):
            masks_out[frame_idx] = video_res_masks[0]
        return masks_out
