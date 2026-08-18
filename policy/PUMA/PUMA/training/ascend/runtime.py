"""Ascend-only training runtime patches and NPU-gated setup."""

import logging
import os
from typing import Optional

import torch

from PUMA.util.device import normalize_device_type


logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no", "off"}


def maybe_enable_ascend_gradient_checkpointing(
    model,
    device: Optional[object] = None,
    enabled: bool = False,
) -> bool:
    """Honor PUMA's checkpointing option without changing non-NPU backends."""
    if not enabled or normalize_device_type(device) != "npu":
        return False

    vlm_interface = getattr(model, "qwen_vl_interface", None)
    backbone = getattr(vlm_interface, "model", None)
    enable = getattr(backbone, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError(
            "Ascend gradient checkpointing requires a compatible Qwen backbone"
        )

    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone_config = getattr(backbone, "config", None)
    if backbone_config is not None and hasattr(backbone_config, "use_cache"):
        backbone_config.use_cache = False
    return True


def setup_ascend_runtime(current_accelerator, runtime_logger=None) -> None:
    """Install the configured Ascend training patches for one accelerator."""
    if normalize_device_type(current_accelerator.device) != "npu":
        return
    runtime_logger = runtime_logger or logger

    patch_specs = (
        (
            "PUMA_ASCEND_PATCH_DEEPSPEED_GRAD_NORM",
            "1",
            patch_deepspeed_zero_grad_norm_for_ascend,
            "Patched DeepSpeed ZeRO grad norm for Ascend NPU float reductions.",
        ),
        (
            "PUMA_ASCEND_PATCH_QWEN_RMSNORM",
            "1",
            patch_qwen_rms_norm_for_ascend,
            "Patched %s Qwen RMSNorm class(es) for Ascend NPU.",
        ),
    )
    for env_name, default, patch, message in patch_specs:
        if not _env_enabled(env_name, default):
            continue
        result = patch()
        if result:
            if "%s" in message:
                runtime_logger.info(message, result)
            else:
                runtime_logger.info(message)


def patch_deepspeed_zero_grad_norm_for_ascend(zero_stage_module=None) -> bool:
    if zero_stage_module is None:
        try:
            from deepspeed.runtime.zero import stage_1_and_2 as zero_stage
        except ImportError:
            return False
    else:
        zero_stage = zero_stage_module

    optimizer_cls = getattr(zero_stage, "DeepSpeedZeroOptimizer", None)
    if optimizer_cls is None:
        return False

    pipe_replicated_attr = getattr(zero_stage, "PIPE_REPLICATED", None)
    is_model_parallel_parameter = getattr(zero_stage, "is_model_parallel_parameter", lambda param: False)
    dist = zero_stage.dist
    patched = False

    def _zero_tensor_for_optimizer(optimizer):
        return torch.tensor(0.0, dtype=torch.float32).to(getattr(optimizer, "device", "cpu"))

    def _should_count_param(optimizer, param):
        if pipe_replicated_attr and hasattr(param, pipe_replicated_attr):
            if getattr(param, pipe_replicated_attr):
                return False
        return is_model_parallel_parameter(param) or getattr(optimizer, "model_parallel_rank", 0) == 0

    def _ascend_get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        if norm_type == float("inf"):
            local_max = None
            for grad, param in zip(gradients, params):
                if not _should_count_param(self, param):
                    continue
                value = grad.detach().float().abs().max()
                local_max = value if local_max is None else torch.maximum(local_max, value)
            total_norm = local_max if local_max is not None else _zero_tensor_for_optimizer(self)
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.MAX)
            return total_norm

        total_sq_or_power = None
        for grad, param in zip(gradients, params):
            if not _should_count_param(self, param):
                continue
            grad_float = grad.detach().float()
            if norm_type == 2.0:
                local = (grad_float * grad_float).sum()
            else:
                local = grad_float.abs().pow(norm_type).sum()
            total_sq_or_power = local if total_sq_or_power is None else total_sq_or_power + local

        total_norm = total_sq_or_power if total_sq_or_power is not None else _zero_tensor_for_optimizer(self)
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)
        return total_norm.pow(1.0 / norm_type)

    if not getattr(optimizer_cls, "_puma_ascend_grad_norm_patched", False):
        optimizer_cls._puma_original_get_grad_norm_direct = optimizer_cls.get_grad_norm_direct
        optimizer_cls.get_grad_norm_direct = _ascend_get_grad_norm_direct
        optimizer_cls._puma_ascend_grad_norm_patched = True
        patched = True

    if hasattr(optimizer_cls, "scaled_global_norm") and not getattr(
        optimizer_cls, "_puma_ascend_skip_no_clip_norm_patched", False
    ):
        optimizer_cls._puma_original_scaled_global_norm = optimizer_cls.scaled_global_norm

        def _ascend_scaled_global_norm(self, norm_type=2):
            try:
                clip_grad = float(getattr(self, "clip_grad", 0.0) or 0.0)
            except (TypeError, ValueError):
                clip_grad = 0.0
            if clip_grad <= 0.0:
                return _zero_tensor_for_optimizer(self)
            return self._puma_original_scaled_global_norm(norm_type=norm_type)

        optimizer_cls.scaled_global_norm = _ascend_scaled_global_norm
        optimizer_cls._puma_ascend_skip_no_clip_norm_patched = True
        patched = True

    return patched


def patch_qwen_rms_norm_for_ascend(
    torch_npu_module=None,
    class_modules=None,
    *,
    require_npu_device: bool = True,
) -> int:
    if torch_npu_module is None:
        try:
            import torch_npu as torch_npu_module
        except ImportError:
            return 0

    if not hasattr(torch_npu_module, "npu_rms_norm"):
        return 0

    if class_modules is None:
        class_modules = []
        for module_name, class_names in (
            ("transformers.models.qwen3.modeling_qwen3", ("Qwen3RMSNorm",)),
            ("transformers.models.qwen3_vl.modeling_qwen3_vl", ("Qwen3VLTextRMSNorm",)),
        ):
            try:
                module = __import__(module_name, fromlist=list(class_names))
            except ImportError:
                continue
            class_modules.append((module, class_names))

    patched_count = 0

    def _can_use_npu_rms_norm(hidden_states, weight):
        if require_npu_device and normalize_device_type(hidden_states.device) != "npu":
            return False
        if hidden_states.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False
        if weight.dtype != hidden_states.dtype:
            return False
        return True

    def _build_forward(original_forward):
        def _ascend_qwen_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if not _can_use_npu_rms_norm(hidden_states, self.weight):
                return original_forward(self, hidden_states)
            return torch_npu_module.npu_rms_norm(
                hidden_states,
                self.weight,
                epsilon=float(self.variance_epsilon),
            )[0]

        return _ascend_qwen_rms_norm_forward

    for module, class_names in class_modules:
        for class_name in class_names:
            rms_norm_cls = getattr(module, class_name, None)
            if rms_norm_cls is None or getattr(rms_norm_cls, "_puma_ascend_rms_norm_patched", False):
                continue

            rms_norm_cls._puma_original_forward = rms_norm_cls.forward
            rms_norm_cls.forward = _build_forward(rms_norm_cls._puma_original_forward)
            rms_norm_cls._puma_ascend_rms_norm_patched = True
            patched_count += 1

    return patched_count
