#!/usr/bin/env bash
set -euo pipefail

# NCCL settings
# Auto-detect socket interface when not set
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if DEFAULT_IF=$(ip route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}'); [[ -n "${DEFAULT_IF:-}" ]]; then
    export NCCL_SOCKET_IFNAME="$DEFAULT_IF"
  else
    FIRST_IF=$(ip -o link show 2>/dev/null | awk -F': ' '{print $2}' | awk -F'@' '{print $1}' | grep -v '^lo$' | head -1)
    export NCCL_SOCKET_IFNAME="${FIRST_IF:-lo}"
  fi
  echo "Auto-detected NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
fi
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

# Resolve paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

FRAMEWORK_NAME="PUMA"
DEFAULT_BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
DEFAULT_CONFIG_YAML="${SCRIPT_DIR}/puma_train_robotwin_world.yaml"

CONFIG_YAML="${CONFIG_YAML:-${DEFAULT_CONFIG_YAML}}"
BASE_VLM="${BASE_VLM:-${DEFAULT_BASE_VLM}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"

# Robotwin dataset settings
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/datasets}"
DATA_MIX="${DATA_MIX:-robotwin_dynamic_task}"

TITLE="${TITLE:-${title:-}}"

# Output settings
if [[ -z "${RUN_ROOT_DIR:-}" ]]; then
  RUN_ROOT_DIR="${REPO_ROOT}/results/Checkpoints"
fi
if [[ -z "${RUN_ID:-}" ]]; then
  BASE_RUN_ID="$(date +%Y%m%d)_robotwin_puma"
  if [[ -n "${TITLE:-}" ]]; then
    RUN_ID="${BASE_RUN_ID}-${TITLE}"
  else
    RUN_ID="${BASE_RUN_ID}"
  fi
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
fi

# Training knobs (override via env if needed)
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
INCLUDE_STATE="${INCLUDE_STATE:-0}"
STATE_DIM="${STATE_DIM:-14}"

# W&B
WANDB_PROJECT="${WANDB_PROJECT:-PUMA_Robotwin}"
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
    export WANDB_MODE="online"
  else
    export WANDB_MODE="disabled"
  fi
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/PUMA/training/train_puma.py}"

mkdir -p "${OUTPUT_DIR}"
cp "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/"

EXTRA_ARGS=()
if [[ "${INCLUDE_STATE}" == "1" || "${INCLUDE_STATE}" == "true" ]]; then
  EXTRA_ARGS+=(
    --datasets.vla_data.include_state true
    --framework.action_model.state_dim "${STATE_DIM}"
  )
else
  EXTRA_ARGS+=(
    --datasets.vla_data.include_state false
  )
fi

# World model parameters
if [[ -n "${LOSS_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(--framework.world_model.loss_weight "${LOSS_WEIGHT}")
fi
if [[ -n "${WORLD_QUERY_NUM:-}" ]]; then
  EXTRA_ARGS+=(
    --framework.world_model.world_query_num "${WORLD_QUERY_NUM}"
    --datasets.vla_data.future_k "${WORLD_QUERY_NUM}"
  )
fi
if [[ -n "${HISTORY_K:-}" ]]; then
  EXTRA_ARGS+=(--datasets.vla_data.history_k "${HISTORY_K}")
fi

# Action model parameters
if [[ -n "${FUTURE_ACTION_WINDOW_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--framework.action_model.future_action_window_size "${FUTURE_ACTION_WINDOW_SIZE}")
fi
if [[ -n "${PAST_ACTION_WINDOW_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--framework.action_model.past_action_window_size "${PAST_ACTION_WINDOW_SIZE}")
fi

accelerate launch \
  --config_file "${REPO_ROOT}/PUMA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes "${NUM_GPUS}" \
  "${TRAIN_SCRIPT}" \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${EXTRA_ARGS[@]}"
