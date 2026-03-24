#!/bin/bash
# RoboTwin to LeRobot Data Conversion Pipeline
#
# This script provides a unified interface to convert RoboTwin data to LeRobot format.
# It automatically handles conda environment switching and environment variable setup.
#
# Usage:
#   Full pipeline:
#     bash convert.sh all <task_name> <setting> <expert_data_num> <repo_id>
#
#   Step 1 only (RoboTwin -> ALOHA HDF5):
#     bash convert.sh step1 <task_name> <setting> <expert_data_num>
#
#   Step 2 only (ALOHA HDF5 -> LeRobot):
#     bash convert.sh step2 <hdf5_data_dir> <repo_id>
#
# Configuration:
#   You can set environment variables before running, or modify defaults below:
#   - ROBOTWIN_DATA_PATH: Path to RoboTwin raw data
#   - DATA_PATH: Base path for intermediate ALOHA HDF5 data
#   - HF_LEROBOT_HOME: Path for LeRobot output
#   - CONDA_ENV_ROBOTWIN: Conda environment for Step 1 (default: RoboTwin)
#   - CONDA_ENV_LEROBOT: Conda environment for Step 2 (default: go1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Configuration - Modify these defaults or set as environment variables
# ============================================================================

# Default paths (can be overridden by environment variables)
ROBOTWIN_DATA_PATH_DEFAULT="${SCRIPT_DIR}/../../data"
DATA_PATH_DEFAULT="${SCRIPT_DIR}/../../converted_data"
HF_LEROBOT_HOME_DEFAULT="${SCRIPT_DIR}/../../converted_data/lerobot"

# Conda environment names
CONDA_ENV_ROBOTWIN="${CONDA_ENV_ROBOTWIN:-RoboTwin}"
CONDA_ENV_LEROBOT="${CONDA_ENV_LEROBOT:-go1}"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "Usage:"
    echo "  Full pipeline:"
    echo "    bash convert.sh all <task_name> <setting> <expert_data_num> <repo_id>"
    echo ""
    echo "  Step 1 only (RoboTwin -> ALOHA HDF5):"
    echo "    bash convert.sh step1 <task_name> <setting> <expert_data_num>"
    echo ""
    echo "  Step 2 only (ALOHA HDF5 -> LeRobot):"
    echo "    bash convert.sh step2 <hdf5_data_dir> <repo_id>"
    echo ""
    echo "  Batch processing (multiple tasks with regex filter):"
    echo "    bash convert.sh batch <task_name1> [task_name2 ...] --pattern <regex_pattern> [--merge] [--step <1|2|all>] [--max-episodes <num>]"
    echo "    Options:"
    echo "      --pattern <regex>    : Regex to match task configs (required)"
    echo "      --step <all|step1|step2> : Which step to run (default: all)"
    echo "      --max-episodes <num> : Max episodes per config (optional)"
    echo "      --merge              : Merge configs within each task into one dataset"
    echo "    Examples:"
    echo "      bash convert.sh batch click_bell beat_block_hammer --pattern 'level1' --step all"
    echo "      bash convert.sh batch click_bell --pattern '.*level.*' --merge --step step2"
    echo "      bash convert.sh batch click_bell --pattern 'level1' --max-episodes 50 --step step1"
    echo "    Merge mode output structure:"
    echo "      Without --merge: lerobot/{task_name}/{config}/"
    echo "      With --merge:    lerobot/{task_name}/"
    echo ""
    echo "Configuration (set as environment variables or modify script defaults):"
    echo "  ROBOTWIN_DATA_PATH: Path to RoboTwin raw data"
    echo "  DATA_PATH: Base path for ALOHA HDF5 output"
    echo "  HF_LEROBOT_HOME: Path for LeRobot output"
    echo "  CONDA_ENV_ROBOTWIN: Conda environment for Step 1 (default: RoboTwin)"
    echo "  CONDA_ENV_LEROBOT: Conda environment for Step 2 (default: go1)"
}

# Initialize conda if available
init_conda() {
    if command -v conda &> /dev/null; then
        # Initialize conda for bash shell
        eval "$(conda shell.bash hook)"
        return 0
    else
        echo "Warning: conda not found. Will try to proceed without conda activation."
        return 1
    fi
}

# Run command in specified conda environment
run_in_conda_env() {
    local env_name=$1
    shift
    local cmd="$@"

    if init_conda; then
        echo "Activating conda environment: ${env_name}"
        conda activate "${env_name}"
        eval "${cmd}"
        conda deactivate
    else
        # If conda not available, try to run directly
        echo "Running without conda activation..."
        eval "${cmd}"
    fi
}

# Set up environment variables
setup_env_vars() {
    export ROBOTWIN_DATA_PATH="${ROBOTWIN_DATA_PATH:-${ROBOTWIN_DATA_PATH_DEFAULT}}"
    export DATA_PATH="${DATA_PATH:-${DATA_PATH_DEFAULT}}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_LEROBOT_HOME_DEFAULT}}"

    # Create output directories if they don't exist
    mkdir -p "${DATA_PATH}/aloha_hdf5"
    mkdir -p "${HF_LEROBOT_HOME}"

    echo "Environment Variables:"
    echo "  ROBOTWIN_DATA_PATH: ${ROBOTWIN_DATA_PATH}"
    echo "  DATA_PATH: ${DATA_PATH}"
    echo "  HF_LEROBOT_HOME: ${HF_LEROBOT_HOME}"
}

# Count episodes in a task config directory
count_episodes() {
    local task_dir=$1
    local data_dir="${task_dir}/data"
    
    if [ ! -d "${data_dir}" ]; then
        echo 0
        return
    fi
    
    # Count episode*.hdf5 files
    local count=$(find "${data_dir}" -maxdepth 1 -name "episode*.hdf5" -type f | wc -l)
    echo ${count}
}

# Find matching task configs using regex pattern
find_matching_configs() {
    local task_name=$1
    local pattern=$2
    local task_path="${ROBOTWIN_DATA_PATH}/${task_name}"
    
    if [ ! -d "${task_path}" ]; then
        return
    fi
    
    # Find all directories and match against pattern
    find "${task_path}" -maxdepth 1 -type d | \
        while read -r config_dir; do
            local config_name=$(basename "${config_dir}")
            # Skip the task directory itself and hidden directories
            if [ "${config_name}" != "${task_name}" ] && [ "${config_name:0:1}" != "." ]; then
                # Use bash regex matching
                if [[ "${config_name}" =~ ${pattern} ]]; then
                    echo "${config_name}"
                fi
            fi
        done | sort
}

# Process batch conversion
process_batch() {
    local step_mode=$1
    local merge_mode=$2
    local max_episodes=$3
    shift 3
    local task_names=("$@")
    
    echo "========================================"
    echo "Batch Processing Mode"
    echo "========================================"
    echo "Tasks: ${task_names[*]}"
    echo "Pattern: ${BATCH_PATTERN}"
    echo "Step: ${step_mode}"
    if [ "${merge_mode}" = "true" ]; then
        echo "Merge Mode: ON (merge configs within each task)"
    else
        echo "Merge Mode: OFF (separate dataset per config)"
    fi
    if [ -n "${max_episodes}" ] && [ "${max_episodes}" -gt 0 ]; then
        echo "Max Episodes per config: ${max_episodes}"
    fi
    echo "========================================"
    echo ""
    
    # Collect all task-config pairs
    local all_pairs=()
    local total_episodes=0
    
    for task_name in "${task_names[@]}"; do
        echo "Scanning task: ${task_name}"
        local matching_configs=($(find_matching_configs "${task_name}" "${BATCH_PATTERN}"))
        
        if [ ${#matching_configs[@]} -eq 0 ]; then
            echo "  Warning: No matching configs found for pattern '${BATCH_PATTERN}'"
            continue
        fi
        
        for config in "${matching_configs[@]}"; do
            local task_dir="${ROBOTWIN_DATA_PATH}/${task_name}/${config}"
            local actual_count=$(count_episodes "${task_dir}")
            
            if [ ${actual_count} -eq 0 ]; then
                echo "  Skipping ${task_name}/${config}: No episodes found"
                continue
            fi
            
            # Apply max_episodes limit if specified
            local episode_count=${actual_count}
            if [ -n "${max_episodes}" ] && [ "${max_episodes}" -gt 0 ]; then
                if [ ${actual_count} -gt ${max_episodes} ]; then
                    episode_count=${max_episodes}
                    echo "  Found: ${task_name}/${config} (${actual_count} episodes available, limiting to ${episode_count})"
                else
                    echo "  Found: ${task_name}/${config} (${episode_count} episodes)"
                fi
            else
                echo "  Found: ${task_name}/${config} (${episode_count} episodes)"
            fi
            
            all_pairs+=("${task_name}|${config}|${episode_count}")
            total_episodes=$((total_episodes + episode_count))
        done
    done
    
    if [ ${#all_pairs[@]} -eq 0 ]; then
        echo "Error: No valid task-config pairs found"
        exit 1
    fi
    
    echo ""
    echo "Total task-config pairs: ${#all_pairs[@]}"
    echo "Total episodes: ${total_episodes}"
    echo ""
    
    # Process Step 1
    if [ "${step_mode}" = "all" ] || [ "${step_mode}" = "step1" ]; then
        echo ">>> Step 1: RoboTwin -> ALOHA HDF5 <<<"
        echo ""
        
        for pair in "${all_pairs[@]}"; do
            IFS='|' read -r task_name config episode_count <<< "${pair}"
            echo "Processing: ${task_name}/${config} (${episode_count} episodes)"
            
            run_in_conda_env "${CONDA_ENV_ROBOTWIN}" \
                "bash '${SCRIPT_DIR}/robotwin2hdf5.sh' '${task_name}' '${config}' '${episode_count}'"
            
            echo "  Completed: ${task_name}/${config}"
            echo ""
        done
    fi
    
    # Process Step 2
    if [ "${step_mode}" = "all" ] || [ "${step_mode}" = "step2" ]; then
        echo ">>> Step 2: ALOHA HDF5 -> LeRobot <<<"
        echo ""
        
        if [ "${merge_mode}" = "true" ]; then
            # Merge mode: combine configs within each task (not across tasks)
            # Each task gets its own merged dataset
            echo "Merge Mode: Combining configs within each task"
            echo ""
            
            # Get unique task names from all_pairs
            declare -A unique_tasks
            for pair in "${all_pairs[@]}"; do
                IFS='|' read -r task_name config episode_count <<< "${pair}"
                unique_tasks[$task_name]=1
            done
            
            # Process each task separately
            for current_task in "${!unique_tasks[@]}"; do
                echo "Processing task: ${current_task}"
                
                local merged_repo_id="${current_task}"
                local temp_merge_dir="${DATA_PATH}/aloha_hdf5/_temp_merge_${merged_repo_id}"
                
                # Set up cleanup trap for unexpected exit
                cleanup_on_exit() {
                    local dir_to_clean="${DATA_PATH}/aloha_hdf5/_temp_merge_${current_task}"
                    if [ -d "${dir_to_clean}" ]; then
                        echo ""
                        echo "  Cleaning up temporary directory on exit: ${dir_to_clean}"
                        rm -rf "${dir_to_clean}"
                    fi
                }
                trap cleanup_on_exit EXIT INT TERM
                
                # Clean and create temp directory
                rm -rf "${temp_merge_dir}"
                mkdir -p "${temp_merge_dir}"
                
                local episode_counter=0
                local config_count=0
                
                # Filter pairs for current task
                for pair in "${all_pairs[@]}"; do
                    IFS='|' read -r task_name config episode_count <<< "${pair}"
                    
                    # Skip if not current task
                    if [ "${task_name}" != "${current_task}" ]; then
                        continue
                    fi
                    
                    local hdf5_dir="${DATA_PATH}/aloha_hdf5/${task_name}-${config}-${episode_count}"
                    
                    if [ ! -d "${hdf5_dir}" ]; then
                        echo "  Warning: ${hdf5_dir} not found, skipping"
                        continue
                    fi
                    
                    echo "  Adding: ${config} (${episode_count} episodes)"
                    config_count=$((config_count + 1))
                    
                    # Create symlinks to all episode directories
                    while IFS= read -r episode_dir; do
                        local abs_episode_dir=$(cd "${episode_dir}" && pwd)
                        local new_episode_name="episode_$(printf "%06d" ${episode_counter})"
                        ln -s "${abs_episode_dir}" "${temp_merge_dir}/${new_episode_name}"
                        episode_counter=$((episode_counter + 1))
                    done < <(find "${hdf5_dir}" -mindepth 1 -maxdepth 1 -type d -name "episode_*" | sort)
                done
                
                if [ ${episode_counter} -eq 0 ]; then
                    echo "  Warning: No episodes found for ${current_task}, skipping"
                    rm -rf "${temp_merge_dir}"
                    trap - EXIT INT TERM
                    continue
                fi
                
                echo "  Total: ${episode_counter} episodes from ${config_count} configs"
                echo ""
                
                # Process merged directory
                run_in_conda_env "${CONDA_ENV_LEROBOT}" \
                    "bash '${SCRIPT_DIR}/hdf52lerobot.sh' '${temp_merge_dir}' '${merged_repo_id}'"
                
                # Clean up temporary directory on success
                rm -rf "${temp_merge_dir}"
                trap - EXIT INT TERM
                
                echo "  Merged dataset created: ${merged_repo_id}"
                echo ""
            done
            
        else
            # Separate mode: create LeRobot dataset for each task-config
            for pair in "${all_pairs[@]}"; do
                IFS='|' read -r task_name config episode_count <<< "${pair}"
                local hdf5_dir="${DATA_PATH}/aloha_hdf5/${task_name}-${config}-${episode_count}"
                local repo_id="${task_name}/${config}"
                
                if [ ! -d "${hdf5_dir}" ]; then
                    echo "  Skipping ${task_name}/${config}: HDF5 directory not found"
                    continue
                fi
                
                echo "Processing: ${task_name}/${config} -> ${repo_id}"
                run_in_conda_env "${CONDA_ENV_LEROBOT}" \
                    "bash '${SCRIPT_DIR}/hdf52lerobot.sh' '${hdf5_dir}' '${repo_id}'"
                
                echo "  Completed: ${task_name}/${config}"
                echo ""
            done
        fi
    fi
    
    echo "========================================"
    echo "Batch Processing Completed!"
    echo "========================================"
}

# ============================================================================
# Main Script Logic
# ============================================================================

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

mode=${1}

# Set up environment variables
setup_env_vars

case ${mode} in
    "all")
        if [ $# -lt 5 ]; then
            echo "Error: Missing arguments for 'all' mode"
            print_usage
            exit 1
        fi
        task_name=${2}
        setting=${3}
        expert_data_num=${4}
        repo_id=${5}

        echo "========================================"
        echo "Full Pipeline: RoboTwin -> LeRobot"
        echo "========================================"
        echo "Task: ${task_name}"
        echo "Setting: ${setting}"
        echo "Episodes: ${expert_data_num}"
        echo "Repo ID: ${repo_id}"
        echo "========================================"

        # Step 1: RoboTwin -> ALOHA HDF5
        echo ""
        echo ">>> Step 1: RoboTwin -> ALOHA HDF5 <<<"
        run_in_conda_env "${CONDA_ENV_ROBOTWIN}" \
            "bash '${SCRIPT_DIR}/robotwin2hdf5.sh' '${task_name}' '${setting}' '${expert_data_num}'"

        # Step 2: ALOHA HDF5 -> LeRobot
        hdf5_dir="${DATA_PATH}/aloha_hdf5/${task_name}-${setting}-${expert_data_num}"
        echo ""
        echo ">>> Step 2: ALOHA HDF5 -> LeRobot <<<"
        run_in_conda_env "${CONDA_ENV_LEROBOT}" \
            "bash '${SCRIPT_DIR}/hdf52lerobot.sh' '${hdf5_dir}' '${repo_id}'"

        echo ""
        echo "========================================"
        echo "Full Pipeline Completed!"
        echo "Output: ${HF_LEROBOT_HOME}/${repo_id}"
        echo "========================================"
        ;;

    "step1")
        if [ $# -lt 4 ]; then
            echo "Error: Missing arguments for 'step1' mode"
            print_usage
            exit 1
        fi
        task_name=${2}
        setting=${3}
        expert_data_num=${4}

        echo "========================================"
        echo "Step 1: RoboTwin -> ALOHA HDF5"
        echo "========================================"
        echo "Task: ${task_name}"
        echo "Setting: ${setting}"
        echo "Episodes: ${expert_data_num}"
        echo "========================================"

        run_in_conda_env "${CONDA_ENV_ROBOTWIN}" \
            "bash '${SCRIPT_DIR}/robotwin2hdf5.sh' '${task_name}' '${setting}' '${expert_data_num}'"

        echo ""
        echo "Step 1 Completed!"
        echo "Output: ${DATA_PATH}/aloha_hdf5/${task_name}-${setting}-${expert_data_num}"
        ;;

    "step2")
        if [ $# -lt 3 ]; then
            echo "Error: Missing arguments for 'step2' mode"
            print_usage
            exit 1
        fi
        hdf5_dir=${2}
        repo_id=${3}

        echo "========================================"
        echo "Step 2: ALOHA HDF5 -> LeRobot"
        echo "========================================"
        echo "Input: ${hdf5_dir}"
        echo "Repo ID: ${repo_id}"
        echo "========================================"

        run_in_conda_env "${CONDA_ENV_LEROBOT}" \
            "bash '${SCRIPT_DIR}/hdf52lerobot.sh' '${hdf5_dir}' '${repo_id}'"

        echo ""
        echo "Step 2 Completed!"
        echo "Output: ${HF_LEROBOT_HOME}/${repo_id}"
        ;;

    "batch")
        # Parse batch arguments
        task_names=()
        batch_pattern=""
        batch_step="all"
        merge_mode="false"
        max_episodes=""
        
        shift  # Remove "batch"
        while [ $# -gt 0 ]; do
            case $1 in
                --pattern)
                    batch_pattern="${2}"
                    shift 2
                    ;;
                --step)
                    batch_step="${2}"
                    shift 2
                    ;;
                --merge)
                    merge_mode="true"
                    shift
                    ;;
                --max-episodes)
                    max_episodes="${2}"
                    # Validate it's a positive integer
                    if ! [[ "${max_episodes}" =~ ^[1-9][0-9]*$ ]]; then
                        echo "Error: --max-episodes must be a positive integer"
                        exit 1
                    fi
                    shift 2
                    ;;
                *)
                    # Assume it's a task name
                    task_names+=("${1}")
                    shift
                    ;;
            esac
        done
        
        if [ ${#task_names[@]} -eq 0 ]; then
            echo "Error: No task names provided"
            print_usage
            exit 1
        fi
        
        if [ -z "${batch_pattern}" ]; then
            echo "Error: --pattern is required for batch mode"
            print_usage
            exit 1
        fi
        
        if [ "${batch_step}" != "all" ] && [ "${batch_step}" != "step1" ] && [ "${batch_step}" != "step2" ]; then
            echo "Error: --step must be 'all', 'step1', or 'step2'"
            print_usage
            exit 1
        fi
        
        # Export pattern for use in process_batch function
        export BATCH_PATTERN="${batch_pattern}"
        
        process_batch "${batch_step}" "${merge_mode}" "${max_episodes}" "${task_names[@]}"
        ;;

    *)
        echo "Error: Unknown mode '${mode}'"
        print_usage
        exit 1
        ;;
esac
