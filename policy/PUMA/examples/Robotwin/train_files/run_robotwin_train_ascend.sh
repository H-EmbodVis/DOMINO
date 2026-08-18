#!/usr/bin/env bash
set -euo pipefail

# Ascend / CANN launch wrapper for Robotwin PUMA training.
# Paths default to this PUMA tree (policy/PUMA) and the playground/ convention.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-3600}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-PUMA}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${REPO_ROOT}/playground/Pretrained_models}"
export TORCH_HOME="${TORCH_HOME:-${PRETRAINED_ROOT}/torch_cache}"
DEFAULT_BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
DEFAULT_CONFIG_YAML="${SCRIPT_DIR}/puma_train_robotwin_ascend.yaml"

CONFIG_YAML="${CONFIG_YAML:-${DEFAULT_CONFIG_YAML}}"
BASE_VLM="${BASE_VLM:-${DEFAULT_BASE_VLM}}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
LINEARIZE_VISION_PATCH_EMBED="${LINEARIZE_VISION_PATCH_EMBED:-true}"

DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/datasets/domino}"
DATA_MIX="${DATA_MIX:-robotwin_dynamic_task}"
ACTION_TYPE="${ACTION_TYPE:-abs_qpos}"
VLA_NUM_WORKERS="${VLA_NUM_WORKERS:-8}"
HISTORY_FLOW_CPU_WORKERS="${HISTORY_FLOW_CPU_WORKERS:-${VLA_NUM_WORKERS}}"
VLA_VIDEO_BACKEND="${VLA_VIDEO_BACKEND:-torchvision_av}"
LOAD_ALL_DATA_FOR_TRAINING="${LOAD_ALL_DATA_FOR_TRAINING:-true}"

NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
GRADIENT_CLIPPING="${GRADIENT_CLIPPING:-1.0}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
INCLUDE_STATE="${INCLUDE_STATE:-false}"
STATE_DIM="${STATE_DIM:-14}"

ACTION_MODEL_TYPE="${ACTION_MODEL_TYPE:-MLP}"
ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-2560}"
ACTION_DIM="${ACTION_DIM:-14}"
FUTURE_ACTION_WINDOW_SIZE="${FUTURE_ACTION_WINDOW_SIZE:-15}"
PAST_ACTION_WINDOW_SIZE="${PAST_ACTION_WINDOW_SIZE:-0}"
WORLD_MODEL_ENABLED="${WORLD_MODEL_ENABLED:-false}"
WORLD_QUERY_NUM="${WORLD_QUERY_NUM:-4}"
LOSS_WEIGHT="${LOSS_WEIGHT:-}"
case "${WORLD_MODEL_ENABLED,,}" in
  1|true|yes|y|on)
    DEFAULT_FUTURE_K="${WORLD_QUERY_NUM}"
    ;;
  *)
    DEFAULT_FUTURE_K="0"
    ;;
esac
FUTURE_K="${FUTURE_K:-${DEFAULT_FUTURE_K}}"
HISTORY_K="${HISTORY_K:-4}"

BASE_LR="${BASE_LR:-1.0e-05}"
QWEN_VL_INTERFACE_LR="${QWEN_VL_INTERFACE_LR:-1.0e-05}"
ACTION_MODEL_LR="${ACTION_MODEL_LR:-1.0e-04}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine_with_min_lr}"
MIN_LR="${MIN_LR:-5.0e-07}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-AdamW}"
OPTIMIZER_BETAS="${OPTIMIZER_BETAS:-[0.9, 0.95]}"
OPTIMIZER_EPS="${OPTIMIZER_EPS:-1.0e-08}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1.0e-08}"
ENABLE_MIXED_PRECISION_TRAINING="${ENABLE_MIXED_PRECISION_TRAINING:-true}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-false}"
NON_FINITE_CHECK_INTERVAL="${NON_FINITE_CHECK_INTERVAL:-1}"
NON_FINITE_CHECK_WARMUP_STEPS="${NON_FINITE_CHECK_WARMUP_STEPS:-0}"
export PUMA_ASCEND_NONFINITE_CHECK_INTERVAL="${PUMA_ASCEND_NONFINITE_CHECK_INTERVAL:-${NON_FINITE_CHECK_INTERVAL}}"
export PUMA_ASCEND_NONFINITE_WARMUP_STEPS="${PUMA_ASCEND_NONFINITE_WARMUP_STEPS:-${NON_FINITE_CHECK_WARMUP_STEPS}}"

if [[ -z "${RUN_ROOT_DIR:-}" ]]; then
  RUN_ROOT_DIR="${REPO_ROOT}/results/Checkpoints"
fi
TITLE="${TITLE:-${title:-puma-robotwin-ascend}}"
if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(date +%Y%m%d)-puma-${DATA_MIX}-${TITLE}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT_DIR}/${RUN_ID}}"
export PUMA_DATASET_STATS_CACHE_DIR="${PUMA_DATASET_STATS_CACHE_DIR:-${OUTPUT_DIR}/dataset_stats_cache}"
export PUMA_STEPS_CACHE_DIR="${PUMA_STEPS_CACHE_DIR:-${OUTPUT_DIR}/steps_cache}"
export PUMA_HISTORY_FLOW_CACHE_DIR="${PUMA_HISTORY_FLOW_CACHE_DIR:-${OUTPUT_DIR}/history_flow_cache_root}"
export PUMA_GROUNDING_CACHE_DIR="${PUMA_GROUNDING_CACHE_DIR:-${OUTPUT_DIR}/grounding_cache_root}"

ENABLE_WANDB="${ENABLE_WANDB:-0}"
if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB}" == "true" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  export WANDB_MODE="disabled"
fi
WANDB_PROJECT="${WANDB_PROJECT:-PUMA_Robotwin}"
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
else
  unset WANDB_API_KEY
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/PUMA/training/train_puma.py}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/PUMA/config/deepseeds/deepspeed_zero2_ascend.yaml}"

for required_path in CONFIG_YAML BASE_VLM DATA_ROOT_DIR TRAIN_SCRIPT ACCELERATE_CONFIG; do
  if [[ ! -e "${!required_path}" ]]; then
    echo "Missing ${required_path}: ${!required_path}" >&2
    echo "Set ${required_path} to a real path before launching." >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/"

LAUNCH_ARGS=()
if [[ -n "${MAIN_PROCESS_PORT:-${MASTER_PORT:-}}" ]]; then
  LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT:-${MASTER_PORT}}")
fi

CMD=(
  python -m accelerate.commands.launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_GPUS}"
  "${LAUNCH_ARGS[@]}"
  "${TRAIN_SCRIPT}"
  --config_yaml "${CONFIG_YAML}"
  --framework.name "${FRAMEWORK_NAME}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}"
  --framework.qwenvl.model_dtype "${MODEL_DTYPE}"
  --framework.qwenvl.linearize_vision_patch_embed "${LINEARIZE_VISION_PATCH_EMBED}"
  --framework.world_model.enabled "${WORLD_MODEL_ENABLED}"
  --framework.world_model.world_query_num "${WORLD_QUERY_NUM}"
  --framework.action_model.action_model_type "${ACTION_MODEL_TYPE}"
  --framework.action_model.action_hidden_dim "${ACTION_HIDDEN_DIM}"
  --framework.action_model.action_dim "${ACTION_DIM}"
  --framework.action_model.state_dim "${STATE_DIM}"
  --framework.action_model.future_action_window_size "${FUTURE_ACTION_WINDOW_SIZE}"
  --framework.action_model.past_action_window_size "${PAST_ACTION_WINDOW_SIZE}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.action_type "${ACTION_TYPE}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.num_workers "${VLA_NUM_WORKERS}"
  --datasets.vla_data.history_flow.cpu_worker_num "${HISTORY_FLOW_CPU_WORKERS}"
  --datasets.vla_data.video_backend "${VLA_VIDEO_BACKEND}"
  --datasets.vla_data.load_all_data_for_training "${LOAD_ALL_DATA_FOR_TRAINING}"
  --datasets.vla_data.include_state "${INCLUDE_STATE}"
  --datasets.vla_data.future_k "${FUTURE_K}"
  --datasets.vla_data.history_k "${HISTORY_K}"
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --trainer.gradient_clipping "${GRADIENT_CLIPPING}"
  --trainer.freeze_modules "${FREEZE_MODULES}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.learning_rate.base "${BASE_LR}"
  --trainer.learning_rate.qwen_vl_interface "${QWEN_VL_INTERFACE_LR}"
  --trainer.learning_rate.action_model "${ACTION_MODEL_LR}"
  --trainer.lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --trainer.scheduler_specific_kwargs.min_lr "${MIN_LR}"
  --trainer.optimizer.name "${OPTIMIZER_NAME}"
  --trainer.optimizer.betas "${OPTIMIZER_BETAS}"
  --trainer.optimizer.eps "${OPTIMIZER_EPS}"
  --trainer.optimizer.weight_decay "${OPTIMIZER_WEIGHT_DECAY}"
  --trainer.enable_gradient_checkpointing "${ENABLE_GRADIENT_CHECKPOINTING}"
  --trainer.enable_mixed_precision_training "${ENABLE_MIXED_PRECISION_TRAINING}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
)

if [[ -n "${LOSS_WEIGHT}" ]]; then
  CMD+=(
    --framework.world_model.loss_weight "${LOSS_WEIGHT}"
  )
fi

printf 'launch_command='
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
