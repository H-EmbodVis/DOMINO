#!/bin/bash
set -euo pipefail
set -x
export PATH="${CUDA_HOME:-/path/to/cuda}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-/path/to/cuda}/lib64:${LD_LIBRARY_PATH:-}"
source "${CONDA_HOME:-/path/to/miniconda3}/bin/activate"
cd "${PROJECT_ROOT:-/path/to/PUMA}"
conda activate "${CONDA_ENV_NAME:-puma}"

export PYTHONPATH="${PROJECT_ROOT:-/path/to/PUMA}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Model & data
export BASE_VLM="${BASE_VLM:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/datasets/robotwin}"
export DATA_MIX="${DATA_MIX:-robotwin_dynamic_task}"
export WORLD_QUERY_NUM="${WORLD_QUERY_NUM:-4}"
export HISTORY_K="${HISTORY_K:-4}"
export LOSS_WEIGHT="${LOSS_WEIGHT:-0.05}"
export CONFIG_YAML="${CONFIG_YAML:-./examples/Robotwin/train_files/puma_train_robotwin.yaml}"

# W&B
export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-PUMA}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export TITLE="${TITLE:-${title:-puma-robotwin-dynamic-35task}}"

# Training resources
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/path/to/output}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d)-puma-${DATA_MIX}-${TITLE}}"

bash examples/Robotwin/train_files/run_robotwin_train.sh
