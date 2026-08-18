#!/usr/bin/env bash
set -euo pipefail

# Ascend / CANN environment + launch entry for Robotwin PUMA training.
# Run from anywhere; PROJECT_ROOT defaults to this PUMA tree (policy/PUMA).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

# Source CANN env. Override if your install is versioned, e.g.:
#   export ASCEND_SET_ENV=/usr/local/Ascend/cann-8.5.2/set_env.sh
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

if [[ -f "${ASCEND_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_SET_ENV}"
else
  echo "Missing ASCEND_SET_ENV: ${ASCEND_SET_ENV}" >&2
  echo "Set ASCEND_SET_ENV to your CANN set_env.sh (standard path: /usr/local/Ascend/ascend-toolkit/set_env.sh)." >&2
  exit 2
fi

# Optional conda activation: only when CONDA_HOME / CONDA_ROOT or CONDA_ENV_NAME is set.
CONDA_PREFIX_DIR="${CONDA_HOME:-${CONDA_ROOT:-}}"
if [[ -n "${CONDA_PREFIX_DIR}" ]]; then
  if [[ -f "${CONDA_PREFIX_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_PREFIX_DIR}/bin/activate"
    if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
      conda activate "${CONDA_ENV_NAME}"
    fi
  else
    echo "Missing conda activate script: ${CONDA_PREFIX_DIR}/bin/activate" >&2
    exit 2
  fi
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
  else
    echo "CONDA_ENV_NAME is set but conda is not on PATH; set CONDA_HOME or CONDA_ROOT." >&2
    exit 2
  fi
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-3600}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1836}"

unset CUDA_HOME CUDA_VISIBLE_DEVICES CUDA_PATH
unset NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_BLOCKING_WAIT NCCL_ASYNC_ERROR_HANDLING NCCL_TIMEOUT NCCL_SOCKET_TIMEOUT_MS

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PRETRAINED_ROOT="${PRETRAINED_ROOT:-${PROJECT_ROOT}/playground/Pretrained_models}"
export TORCH_HOME="${TORCH_HOME:-${PRETRAINED_ROOT}/torch_cache}"
export BASE_VLM="${BASE_VLM:-${PRETRAINED_ROOT}/Qwen3-VL-4B-Instruct}"
export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/datasets/domino}"
export DATA_MIX="${DATA_MIX:-robotwin_dynamic_task}"
export CONFIG_YAML="${CONFIG_YAML:-${PROJECT_ROOT}/examples/Robotwin/train_files/puma_train_robotwin_ascend.yaml}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/PUMA/config/deepseeds/deepspeed_zero2_ascend.yaml}"

export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
export LINEARIZE_VISION_PATCH_EMBED="${LINEARIZE_VISION_PATCH_EMBED:-true}"
export WORLD_MODEL_ENABLED="${WORLD_MODEL_ENABLED:-true}"
export LOSS_WEIGHT="${LOSS_WEIGHT:-}"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
export VLA_NUM_WORKERS="${VLA_NUM_WORKERS:-8}"
export HISTORY_FLOW_CPU_WORKERS="${HISTORY_FLOW_CPU_WORKERS:-${VLA_NUM_WORKERS}}"
export NON_FINITE_CHECK_INTERVAL="${NON_FINITE_CHECK_INTERVAL:-1}"
export NON_FINITE_CHECK_WARMUP_STEPS="${NON_FINITE_CHECK_WARMUP_STEPS:-0}"
export PUMA_ASCEND_NONFINITE_CHECK_INTERVAL="${PUMA_ASCEND_NONFINITE_CHECK_INTERVAL:-${NON_FINITE_CHECK_INTERVAL}}"
export PUMA_ASCEND_NONFINITE_WARMUP_STEPS="${PUMA_ASCEND_NONFINITE_WARMUP_STEPS:-${NON_FINITE_CHECK_WARMUP_STEPS}}"

export ENABLE_WANDB="${ENABLE_WANDB:-0}"
if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB}" == "true" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  export WANDB_MODE="disabled"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-PUMA_Robotwin}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export TITLE="${TITLE:-${title:-puma-robotwin-ascend}}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PROJECT_ROOT}/results/Checkpoints}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d)-puma-${DATA_MIX}-${TITLE}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT_DIR}/${RUN_ID}}"
export PUMA_DATASET_STATS_CACHE_DIR="${PUMA_DATASET_STATS_CACHE_DIR:-${OUTPUT_DIR}/dataset_stats_cache}"
export PUMA_STEPS_CACHE_DIR="${PUMA_STEPS_CACHE_DIR:-${OUTPUT_DIR}/steps_cache}"
export PUMA_HISTORY_FLOW_CACHE_DIR="${PUMA_HISTORY_FLOW_CACHE_DIR:-${OUTPUT_DIR}/history_flow_cache_root}"
export PUMA_GROUNDING_CACHE_DIR="${PUMA_GROUNDING_CACHE_DIR:-${OUTPUT_DIR}/grounding_cache_root}"

for required_path in BASE_VLM CONFIG_YAML DATA_ROOT_DIR ACCELERATE_CONFIG; do
  if [[ ! -e "${!required_path}" ]]; then
    echo "Missing ${required_path}: ${!required_path}" >&2
    echo "Set ${required_path} to a real path before launching." >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
cp "${SCRIPT_PATH}" "${OUTPUT_DIR}/"

bash "${PROJECT_ROOT}/examples/Robotwin/train_files/run_robotwin_train_ascend.sh"
