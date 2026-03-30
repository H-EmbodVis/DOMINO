"""
PUMA Framework

A predictive VLA architecture that couples historical motion cues with future state anticipation to achieve highly reactive embodied intelligence.
Key Points:
  - Qwen vision-language backbone with flash_attention_2
  - Injects an action special token into the VLM
  - Historical flow input as auxiliary input to the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states
  - World-query module for future object-centric feature prediction
"""
from typing import Any, List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from PUMA.training.trainer_utils import initialize_overwatch
from PUMA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from PUMA.model.framework.base_framework import baseframework
from PUMA.model.modules.vlm import get_vlm_model
from PUMA.model.modules.action_model.MLP_ActionHeader import get_action_model
from PUMA.model.modules.world_query import WorldQueryModule, parse_world_text
from PUMA.training.trainer_utils.trainer_tools import resize_images

@FRAMEWORK_REGISTRY.register("PUMA")
class PUMA(baseframework):
    """
    PUMA: Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - MLP action head for continuous action prediction
      - World-query module for future object-centric feature prediction
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_token = "\U0001f50d"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
            "\U0001f50d", add_special_tokens=False
        )["input_ids"][0]

        self.l1_loss = nn.L1Loss()

        world_cfg = self.config.framework.get("world_model", {}) if self.config is not None else {}
        self.world_query_module = WorldQueryModule(
            hidden_dim=self.qwen_vl_interface.model.config.hidden_size,
            world_cfg=world_cfg,
        )
        self.world_enabled = self.world_query_module.enabled
        self.world_query_num = self.world_query_module.query_num
        self.world_token = "<|world|>"
        if self.world_enabled:
            tokenizer = self.qwen_vl_interface.processor.tokenizer
            if self.world_token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [self.world_token]}
                )
                self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
                logger.info(
                    f"Added special token '{self.world_token}' to tokenizer and "
                    f"resized embeddings to {len(tokenizer)}."
                )
            self.world_token_id = tokenizer.convert_tokens_to_ids(self.world_token)

    # ------------------------------------------------------------------
    # Prompt & image assembly helpers
    # ------------------------------------------------------------------

    def _build_instruction_with_queries(self, instruction: str) -> str:
        """Construct the text prompt with world / action query tokens.

        Token ordering (left -> right in causal attention):
          [history images] -> [current images] -> instruction -> [world queries] -> action queries
        """
        parts = [instruction]

        if self.world_enabled:
            world_tokens = self.world_token * self.world_query_num
            parts.append(f" Predict future object states: {world_tokens}.")

        action_tokens = self.action_token * self.chunk_len
        parts.append(f" Predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>.")

        return "".join(parts)

    @staticmethod
    def _assemble_images(examples: List[dict]) -> List[list]:
        """Prepend compressed history frames before current observations."""
        batch_images = []
        for example in examples:
            images = []
            history = example.get("history_images")
            if history:
                images.extend(history)
            images.extend(example["image"])
            batch_images.append(images)
        return batch_images

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: direct regression of future actions (no diffusion).

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
                - (optional) history_images: List[PIL.Image], compressed past frames
                - (optional) future_images: for world feature supervision

        Returns:
            dict with action_loss and optional world_loss.
        """
        batch_images = self._assemble_images(examples)
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        instructions = [
            self._build_instruction_with_queries(instruction)
            for instruction in instructions
        ]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size+1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        world_out = None
        if self.world_enabled:
            future_images = [example.get("future_images") or example.get("future_image") for example in examples]
            if future_images[0] is not None:
                world_queries = self._gather_token_embeddings(
                    last_hidden,
                    input_ids,
                    token_id=self.world_token_id,
                    num_tokens=self.world_query_num,
                    token_name="world",
                )
                text_prompts = []
                for example in examples:
                    world_text = example.get("world_text")
                    if world_text:
                        text_prompts.append(world_text)
                    else:
                        lang = example.get("lang") or example.get("language") or ""
                        text_prompts.append(parse_world_text(lang))
                cache_infos = [example.get("world_cache") for example in examples]
                world_out = self.world_query_module(
                    world_queries,
                    future_images=future_images,
                    text_prompts=text_prompts,
                    cache_infos=cache_infos,
                )

        output = {"action_loss": action_loss}
        if world_out is not None:
            output["world_loss"] = world_out["loss"]
            output["world_loss_raw"] = world_out["loss_raw"]
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to regress future actions.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - (optional) history_images: List[PIL.Image]

        Returns:
            dict: normalized_actions (np.ndarray) of shape [B, T, action_dim].
        """
        vla_cfg = getattr(self.config.datasets, "vla_data", None)
        train_obs_size = tuple(vla_cfg.image_size) if vla_cfg and getattr(vla_cfg, "image_size", None) else None
        hist_size = tuple(vla_cfg.history_image_size) if vla_cfg and getattr(vla_cfg, "history_image_size", None) else None

        batch_images = []
        for example in examples:
            images = []
            history = example.get("history_images")
            if history:
                hist_imgs = to_pil_preserve(history)
                if hist_size:
                    hist_imgs = [img.resize(hist_size) for img in hist_imgs]
                images.extend(hist_imgs)
            current_imgs = to_pil_preserve(example["image"])
            if train_obs_size:
                current_imgs = [img.resize(train_obs_size) for img in current_imgs]
            images.extend(current_imgs)
            batch_images.append(images)

        instructions = [example["lang"] for example in examples]

        instructions = [
            self._build_instruction_with_queries(instruction)
            for instruction in instructions
        ]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        return self._gather_token_embeddings(
            last_hidden,
            input_ids,
            token_id=action_token_id,
            num_tokens=self.chunk_len,
            token_name="action",
        )

    def _gather_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_id=None,
        num_tokens: int = 1,
        token_name: str = "token",
    ) -> torch.Tensor:
        if token_id is None:
            raise ValueError("token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape
        if isinstance(token_id, (list, tuple, set)):
            id_list = torch.tensor(list(token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == token_id)

        counts = mask.sum(dim=1)
        if (counts < num_tokens).any():
            insufficient = (counts < num_tokens).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Following samples have insufficient {token_name} tokens: {insufficient}, Need {num_tokens} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=num_tokens, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)
