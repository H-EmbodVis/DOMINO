# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# This file is modified from the starVLA repository (https://github.com/starVLA/starVLA).

import torch
import os
import logging
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from PUMA.model.modules.vlm.model_loading import resolve_torch_dtype
from PUMA.model.modules.vlm.ascend import (
    QWEN3_ASCEND_INFERENCE_PLAN_KEY,
    Qwen3AscendInferencePlan,
    configure_qwen3_ascend_inference,
    linearize_qwen_vision_patch_embed,
    prepare_qwen3_ascend_inference_inputs,
    qwen3_ascend_inference_plan_context,
)
from PUMA.util.device import get_autocast_context


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 153716 # here only for fast_tokenizer, see PUMA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        raw_model_id = str(qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")).strip()
        expanded_model_id = os.path.expanduser(raw_model_id)

        # Treat absolute/relative paths (including symlinks) as local
        is_local_path = (
            os.path.isabs(expanded_model_id)
            or expanded_model_id.startswith("./")
            or expanded_model_id.startswith("../")
            or os.path.lexists(expanded_model_id)
        )
        model_id = expanded_model_id if is_local_path else raw_model_id
        load_kwargs = {"local_files_only": True} if is_local_path else {}

        attn_impl = str(qwenvl_config.get("attn_implementation", "flash_attention_2"))
        model_dtype = resolve_torch_dtype(qwenvl_config.get("model_dtype", "bfloat16"))
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=model_dtype,
            **load_kwargs
        )
        if bool(qwenvl_config.get("enable_ascend_inference_adapter", False)):
            configure_qwen3_ascend_inference(
                model,
                linearize_patch_embed=bool(
                    qwenvl_config.get("linearize_vision_patch_embed", False)
                ),
            )
            logger.info("Installed required Qwen3-VL Ascend inference adapters.")
        elif bool(qwenvl_config.get("linearize_vision_patch_embed", False)):
            # Standalone flag (e.g. NPU training): Conv3d with kernel == stride is
            # unsupported on Ascend, so replace the vision patch embed up front.
            if not linearize_qwen_vision_patch_embed(model):
                raise RuntimeError("Qwen3-VL Conv3d patch embedding could not be linearized")
            logger.info("Linearized Qwen3-VL vision patch embed.")
        processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3-VL backbone.
        """
        request_plan = kwargs.pop(QWEN3_ASCEND_INFERENCE_PLAN_KEY, None)
        skip_lm_head = bool(kwargs.pop("skip_lm_head", False))

        with qwen3_ascend_inference_plan_context(request_plan):
            with get_autocast_context(self.model.device, dtype=torch.bfloat16):
                if skip_lm_head:
                    kwargs.pop("labels", None)
                    kwargs.pop("logits_to_keep", None)
                    outputs = self.model.model(**kwargs)
                else:
                    outputs = self.model(
                        **kwargs,
                    )

        if (
            skip_lm_head
            and kwargs.get("output_hidden_states")
            and getattr(outputs, "hidden_states", None) is None
        ):
            # The Qwen3-VL base-model wrapper may omit hidden_states even when
            # output_hidden_states is requested; fall back to last_hidden_state.
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if not isinstance(last_hidden_state, torch.Tensor):
                raise RuntimeError("skip_lm_head forward did not return a last hidden state")
            outputs.hidden_states = (last_hidden_state,)

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        request_plan = kwargs.pop(QWEN3_ASCEND_INFERENCE_PLAN_KEY, None)
        if request_plan is not None:
            if self.model.device.type != "npu":
                raise RuntimeError(
                    "Qwen3 Ascend inference plans are only valid on NPU"
                )
            if not isinstance(request_plan, Qwen3AscendInferencePlan):
                raise RuntimeError("Qwen3 generation received an invalid PUMA request plan")
            kwargs.pop("position_ids", None)
            if request_plan.omit_attention_mask and kwargs.get("attention_mask") is None:
                input_ids = kwargs.get("input_ids")
                if not isinstance(input_ids, torch.Tensor):
                    raise RuntimeError(
                        "Qwen3 generation cannot restore an omitted attention mask without input_ids"
                    )
                kwargs["attention_mask"] = torch.ones_like(input_ids)

        with get_autocast_context(self.model.device, dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now. 
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see PUMA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning (f"action token are on in yout tokenizer, plz see PUMA/model/modules/vlm/tools/add_qwen_special_tokens/README.md.")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        if self.model.device.type != "npu" or self.training or solutions is not None:
            return batch_inputs.to(self.model.device)
        return prepare_qwen3_ascend_inference_inputs(
            batch_inputs,
            self.model.model,
            self.model.device,
        )




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/puma_train_robotwin_world.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
