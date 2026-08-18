"""Ascend (NPU) training runtime adapters.

This package owns every Ascend-specific training runtime specialization so that
the generic training entry point stays backend-neutral:

- `runtime`: DeepSpeed ZeRO grad-norm float reductions, Qwen RMSNorm via
  `npu_rms_norm`, and NPU-only gradient checkpointing.
- `metrics`: deferred scalar `.item()` materialization and non-finite loss
  checks.

Reuse `PUMA.util.device` for device-type and autocast decisions rather than
adding backend branches to shared training code.
"""

from PUMA.training.ascend.metrics import (
    collect_training_metrics,
    materialize_training_metrics,
    raise_for_non_finite_loss,
    should_check_non_finite_loss,
)
from PUMA.training.ascend.runtime import (
    maybe_enable_ascend_gradient_checkpointing,
    patch_deepspeed_zero_grad_norm_for_ascend,
    patch_qwen_rms_norm_for_ascend,
    setup_ascend_runtime,
)


__all__ = [
    "collect_training_metrics",
    "materialize_training_metrics",
    "maybe_enable_ascend_gradient_checkpointing",
    "patch_deepspeed_zero_grad_norm_for_ascend",
    "patch_qwen_rms_norm_for_ascend",
    "raise_for_non_finite_loss",
    "setup_ascend_runtime",
    "should_check_non_finite_loss",
]
