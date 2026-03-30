# PUMA (VLA Policy)

> PUMA is a predictive VLA architecture that couples historical motion cues with future state anticipation to achieve highly reactive embodied intelligence.

---

## 🛠️ 1. Environment Setup

The codebase is provided in `policy/PUMA`. Please set up the environment from this directory.

### 1.1 Installation Steps

**Step 1: Create Conda Environment**
```bash
conda create -n puma python=3.10 -y
conda activate puma
```

**Step 2: Install Dependencies and PUMA**
Make sure to install a PyTorch version that matches your CUDA toolkit. We recommend CUDA 12.4.

```bash
# 1. Install PUMA Core Dependencies
cd policy/PUMA
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 2. Install GroundingDINO for Grounded-SAM-2
cd PUMA/model/modules/grounding_sam/grounding_dino
pip install -r requirements.txt
pip install --no-build-isolation -e .
python setup.py build_ext --inplace
cd ..

# 3. Install SAM2
pip install --no-build-isolation -e .
cd ../../../..

# 4. Install PUMA Package
pip install -e .
```

<details close>
<summary><b>Common Issues (Flash-Attn)</b></summary>

`flash-attn` can be tricky to install because it must match your system’s CUDA toolkit (`nvcc`) and PyTorch versions. The `--no-build-isolation` flag resolves most issues, but on newer systems you may need to manually choose a compatible `flash-attn` version. Ensure your CUDA driver/toolkit and torch versions are aligned. Check your environment:

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

If issues persist, pick a `flash-attn` release that matches your versions (CUDA and torch) or ask ChatGPT to help with the outputs above. We have verified that `flash-attn==2.7.4.post1` works well with nvcc versions `12.0` and `12.4`.
</details>

### 1.2 Download Pre-trained Weights

PUMA requires both a Vision-Language-Action base model and grounding models (SAM2 + GroundingDINO). Please download the following weights and place them under `policy/PUMA/playground/Pretrained_models`.

1. **Base VLM Model**
   - Download the `Qwen3-VL-4B-Instruct-Action` base model from Hugging Face: [StarVLA/Qwen3-VL-4B-Instruct-Action](https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action)
   - Place it at: `policy/PUMA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`

2. **Grounded-SAM-2 Models**
   - **SAM 2.1 Large**: Download `sam2.1_hiera_large.pt` from [Meta Segment Anything 2.1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
   - **GroundingDINO Swin-T**: Download `groundingdino_swint_ogc.pth` from [IDEA-Research GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
   - Place all downloaded files at: `policy/PUMA/playground/Pretrained_models/grounded_sam2/`

<details close>
<summary><b>Click to view example directory structure</b></summary>
The resulting directory structure should look like this:

```text
policy/PUMA/playground/Pretrained_models/
├── Qwen3-VL-4B-Instruct-Action/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── ...
└── grounded_sam2/
    ├── groundingdino_swint_ogc.pth
    └── sam2.1_hiera_large.pt
```
</details>

---

## 🚀 2. Training

### 2.1 Data Format Conversion

To train PUMA on RoboTwin data, you first need to convert the raw RoboTwin data into the LeRobot format. This process requires the environment from the [AgiBot-World (go1)](https://github.com/OpenDriveLab/AgiBot-World) repository.

<details close>
<summary><b>Minimal <code>go1</code> Environment Installation</b> (Click to expand)</summary>

```bash
conda create -n go1 python=3.10 -y
conda activate go1
git clone https://github.com/OpenDriveLab/AgiBot-World.git
cd AgiBot-World
pip install -e .
pip install --no-build-isolation flash-attn==2.4.2
```
</details>

<br>

**Conversion Steps:**

The conversion process involves two steps: converting RoboTwin raw data to ALOHA HDF5 format, and then converting the HDF5 format to LeRobot format.

```bash
# Step 1: Convert RoboTwin raw data to ALOHA HDF5 format
export ROBOTWIN_DATA_PATH=/path/to/Dynamic_RoboTwin/data
export DATA_PATH=/path/to/output_hdf5
bash scripts/robotwin2lerobot/robotwin2hdf5.sh <task_name> <setting> <expert_data_num>

# Example: 
# bash scripts/robotwin2lerobot/robotwin2hdf5.sh beat_block_hammer demo_clean_dynamic 50

# Step 2: Convert ALOHA HDF5 data to LeRobot format
export HF_LEROBOT_HOME=/path/to/lerobot_dataset_output
bash scripts/robotwin2lerobot/hdf52lerobot.sh <hdf5_data_dir> <repo_id>

# Example: 
# bash scripts/robotwin2lerobot/hdf52lerobot.sh /path/to/output_hdf5/beat_block_hammer-level1-50 local/beat_block_hammer
```

### 2.2 Modality Configuration

After converting the dataset, you must place the modality configuration file into each task's `meta` folder within the LeRobot dataset directory. 

Copy `examples/Robotwin/train_files/modality.json` to `<TASK_NAME>/meta/modality.json` for **every task** you intend to train on.

### 2.3 Training Script Configuration

The main training launch script is located at `scripts/run_scripts/run_lerobot_robotwin_puma.sh`. This script configures the environment variables required for training.

### 2.4 Launching Training

Once the data is prepared and the script is configured, you can start the training process:

```bash
conda activate puma
cd policy/PUMA
bash scripts/run_scripts/run_lerobot_robotwin_puma.sh
```

---

## 🧪 3. Evaluation

The evaluation process involves a client-server architecture where the `PUMA` policy server communicates with the `DOMINO` simulation environment via WebSockets. Both the policy server and the simulation client load the checkpoint path and network configurations from `examples/Robotwin/eval_files/deploy_policy.yml`.

### 3.1 Additional Dependencies

Ensure you install the required packages for the evaluation communication protocol in **both** your `puma` and simulation (`domino`) environments. 

**Note on CUDA Versions:** We recommend using **CUDA 12.4** for the `puma` environment and **CUDA 12.1** for the `domino` simulation environment to ensure the best compatibility with their respective dependencies.

```bash
pip install -r examples/Robotwin/eval_files/requirements.txt
```

### 3.2 Configuration Setup

Before running the evaluation, you must update the paths and network settings in the following configuration files to match your system:

1. **`examples/Robotwin/eval_files/deploy_policy.yml`**
   - `policy_ckpt_path`: Set this to the absolute path of your trained model checkpoint.
   - `port`: Ensure this matches the port you intend to use for the policy server (default is `9001`, but you can change it).

2. **`examples/Robotwin/eval_files/run_policy_server.sh`**
   - `your_ckpt`: Update this to match the checkpoint path used above.
   - `puma_python`: Set this to the absolute path of the python executable in your `puma` conda environment.
   - `port`: Ensure this matches the port configured in `deploy_policy.yml` (e.g., `9001`).

3. **`examples/Robotwin/eval_files/eval.sh`**
   - `ROBOTWIN_PATH`: Set this to the absolute path of your DOMINO simulation root directory.

### 3.3 Task Evaluation

**Step 1: Start the Policy Server**

Open a new terminal, activate the `puma` environment, and launch the policy server:

```bash
conda activate puma
cd policy/PUMA

# The script will automatically load the checkpoint and port specified inside it
bash examples/Robotwin/eval_files/run_policy_server.sh
```

**Step 2: Start the Simulation Client**

In a separate terminal, activate your simulation environment (`domino`) and launch the evaluation loop:

```bash
conda activate domino
cd policy/PUMA/examples/Robotwin/eval_files

# Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <port>
bash eval.sh adjust_bottle demo_clean_dynamic puma_demo 0 0 9001
```
