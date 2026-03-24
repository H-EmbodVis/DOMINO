#!/bin/bash
# Step 2: Convert ALOHA HDF5 data to LeRobot format
#
# Usage:
#   bash hdf52lerobot.sh <hdf5_data_dir> <repo_id> [--mode video|image]
#
# Environment Variables:
#   HF_LEROBOT_HOME: Path for LeRobot output (required)
#
# Example:
#   export HF_LEROBOT_HOME=/path/to/lerobot_data
#   bash hdf52lerobot.sh ./aloha_hdf5/beat_block_hammer-level1-50 local/beat_block_hammer

set -e

if [ $# -lt 2 ]; then
    echo "Usage: bash hdf52lerobot.sh <hdf5_data_dir> <repo_id> [--mode video|image]"
    echo "Example: bash hdf52lerobot.sh ./aloha_hdf5/beat_block_hammer-level1-50 local/beat_block_hammer"
    exit 1
fi

data_dir=${1}
repo_id=${2}
mode=${3:-"--mode video"}

if [ -z "${HF_LEROBOT_HOME}" ]; then
    echo "Error: HF_LEROBOT_HOME environment variable is not set"
    echo "Please set it: export HF_LEROBOT_HOME=/path/to/lerobot_data"
    exit 1
fi

echo "========================================"
echo "ALOHA HDF5 -> LeRobot Conversion"
echo "========================================"
echo "Input: ${data_dir}"
echo "Repo ID: ${repo_id}"
echo "Mode: ${mode}"
echo "HF_LEROBOT_HOME: ${HF_LEROBOT_HOME}"
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/scripts/convert_aloha_to_lerobot.py" --raw_dir "${data_dir}" --repo_id "${repo_id}" ${mode}

echo "========================================"
echo "Conversion completed!"
echo "Output: ${HF_LEROBOT_HOME}/${repo_id}"
echo "========================================"
