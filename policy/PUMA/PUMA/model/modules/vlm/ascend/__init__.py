"""Ascend (NPU) inference adapters for the Qwen3-VL backbone.

This package owns every Ascend-specific model modification so that the
generic VLM wrappers and deployment entry points stay backend-neutral:

- `patch_embed`: linearized replacement for the Conv3d vision patch embed
  (Conv3d with kernel_size == stride is unsupported/slow on NPU).
- `qwen3_inference`: CPU-built request plans plus runtime class overrides
  that avoid NPU-hostile operators without touching checkpoint keys.

The current release covers inference only. Ascend training support will
land as a sibling module (e.g. `PUMA.training.ascend`) and should reuse
`PUMA.util.device` rather than adding backend branches to shared code.
"""

from PUMA.model.modules.vlm.ascend.patch_embed import (
    LinearizedConv3dPatchEmbed,
    linearize_qwen_vision_patch_embed,
)
from PUMA.model.modules.vlm.ascend.qwen3_inference import (
    QWEN3_ASCEND_INFERENCE_PLAN_KEY,
    Qwen3AscendInferencePlan,
    configure_qwen3_ascend_inference,
    prepare_qwen3_ascend_inference_inputs,
    qwen3_ascend_inference_plan_context,
)


def ascend_inference_config_overrides() -> dict:
    """Checkpoint config overrides applied when serving NVIDIA weights on NPU.

    Keeps the checkpoint untouched: FlashAttention (CUDA-only) is replaced by
    SDPA, and the Conv3d vision patch embed is linearized at load time.
    """
    return {
        "framework": {
            "qwenvl": {
                "attn_implementation": "sdpa",
                "model_dtype": "bfloat16",
                "linearize_vision_patch_embed": True,
                "enable_ascend_inference_adapter": True,
            }
        }
    }


__all__ = [
    "QWEN3_ASCEND_INFERENCE_PLAN_KEY",
    "Qwen3AscendInferencePlan",
    "LinearizedConv3dPatchEmbed",
    "ascend_inference_config_overrides",
    "configure_qwen3_ascend_inference",
    "linearize_qwen_vision_patch_embed",
    "prepare_qwen3_ascend_inference_inputs",
    "qwen3_ascend_inference_plan_context",
]
