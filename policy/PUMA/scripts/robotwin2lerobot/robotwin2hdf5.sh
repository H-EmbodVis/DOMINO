#!/bin/bash
# Step 1: Convert RoboTwin raw data to ALOHA HDF5 format
#
# Usage:
#   bash robotwin2hdf5.sh <task_name> <setting> <expert_data_num>
#
# Environment Variables:
#   ROBOTWIN_DATA_PATH: Path to RoboTwin raw data (default: ../../data)
#   DATA_PATH: Base path for output data (default: current directory)
#
# Example:
#   export ROBOTWIN_DATA_PATH=/path/to/Dynamic_RoboTwin/data
#   export DATA_PATH=/path/to/output
#   bash robotwin2hdf5.sh beat_block_hammer aloha-agilex_clean_level1 50

set -e

if [ $# -lt 3 ]; then
    echo "Usage: bash robotwin2hdf5.sh <task_name> <setting> <expert_data_num>"
    echo "Example: bash robotwin2hdf5.sh beat_block_hammer aloha-agilex_clean_level1 50"
    exit 1
fi

task_name=${1}
setting=${2}
expert_data_num=${3}

# Set default paths if not provided
export ROBOTWIN_DATA_PATH=${ROBOTWIN_DATA_PATH:-"../../data"}
export DATA_PATH=${DATA_PATH:-"."}

echo "========================================"
echo "RoboTwin -> ALOHA HDF5 Conversion"
echo "========================================"
echo "Task: ${task_name}"
echo "Setting: ${setting}"
echo "Episodes: ${expert_data_num}"
echo "ROBOTWIN_DATA_PATH: ${ROBOTWIN_DATA_PATH}"
echo "DATA_PATH: ${DATA_PATH}"
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/scripts/convert_robotwin_to_aloha.py" "${task_name}" "${setting}" "${expert_data_num}"
