# Training PUMA on Huawei Ascend NPUs

PUMA trains on Ascend NPUs out of the box. NVIDIA-trained PUMA / Qwen3-VL
weights load as-is — no conversion step — and the CUDA workflow is left
untouched: all Ascend-specific behavior is switched on only when the job
actually runs on an NPU. Serving a trained checkpoint on NPU is covered in
[ascend_inference.md](ascend_inference.md).

## Verified setup

| Component | Version |
| --- | --- |
| NPU | Atlas 910B3 × 8 |
| CANN | 8.5.2 |
| torch / torch-npu | 2.5.1 / 2.5.1.post1 |
| Python | 3.10 |

Nearby CANN / torch-npu combinations will likely work, but this is the stack
we validated. The supported topology is **8-card DeepSpeed ZeRO-2 (bf16)**;
single-card full-parameter training runs out of memory at the optimizer step,
which is expected for this ~4.5B model.

## Installation

```bash
# 1. Source the CANN toolkit environment (adjust to your install path)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. Create the environment and install the pinned Ascend runtime
conda create -n puma_ascend python=3.10 -y
conda activate puma_ascend
cd policy/PUMA
pip install -r requirements-ascend.txt
pip install --no-build-isolation -e .
```

Three things to watch out for:

- Install `requirements-ascend.txt` **first**, so pip cannot resolve CUDA
  builds of torch.
- Do **not** install `flash-attn`, `decord`, or `eva-decord` — they pull
  CUDA wheels. The Ascend recipe uses `sdpa` attention and the
  `torchvision_av` video backend instead.
- Keep `numpy==1.26.4`. Later pip installs can silently upgrade NumPy to
  2.x, which breaks torch-npu; re-pin it if that happens.

## Data and weights

Data conversion and modality setup are identical to the CUDA path — follow
the [PUMA README training section](../README.md#2-training). Then:

- Put the base VLM under `playground/Pretrained_models/` (default:
  `Qwen3-VL-4B-Instruct`; override with `BASE_VLM` if yours differs).
- Put Grounded-SAM-2 weights under
  `playground/Pretrained_models/grounded_sam2/` for world-model training.
- Point `DATA_ROOT_DIR` at your LeRobot-format dataset, with
  `modality.json` copied into each task's `meta/` folder.

If the dataset lives on a **read-only** mount, no extra setup is needed:
the launch scripts redirect all rebuilt artifacts (dataset statistics, step
index, optical-flow and grounding caches) into the run's output directory
via the `PUMA_*_CACHE_DIR` environment variables.

## Launch training

From `policy/PUMA`:

```bash
# Preview the resolved command without touching the NPUs
DRY_RUN=1 DATA_ROOT_DIR=/path/to/lerobot_dataset \
  bash scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh

# 8-card ZeRO-2 training
DATA_ROOT_DIR=/path/to/lerobot_dataset \
  bash scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh
```

The launcher sources CANN, clears any leftover `CUDA_*` / `NCCL_*`
variables, sets HCCL timeouts, and then starts Accelerate + DeepSpeed with
`examples/Robotwin/train_files/puma_train_robotwin_ascend.yaml` and the
ZeRO-2 config in `PUMA/config/deepseeds/`. Commonly overridden variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_ROOT_DIR` | — | LeRobot dataset root (required) |
| `BASE_VLM` | `playground/Pretrained_models/Qwen3-VL-4B-Instruct` | Base VLM weights |
| `NUM_GPUS` | `8` | Number of NPUs |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | Ascend counterpart of `CUDA_VISIBLE_DEVICES` |
| `ASCEND_SET_ENV` | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | CANN `set_env.sh` path |
| `WORLD_MODEL_ENABLED` | `true` | World-model supervision |
| `PER_DEVICE_BATCH_SIZE` | `4` | Per-NPU batch size |
| `MAX_TRAIN_STEPS` | `200000` | Training steps |
| `ENABLE_WANDB` | `0` | Set `1` to log to Weights & Biases |
| `RUN_ROOT_DIR` | `results/Checkpoints` | Output root; checkpoints land in `RUN_ROOT_DIR/RUN_ID` |

Every other recipe knob (learning rates, optimizer, save intervals, …) has
the same meaning as on CUDA and can be overridden the same way — see
`examples/Robotwin/train_files/run_robotwin_train_ascend.sh` for the full
list. Running that example script directly keeps `WORLD_MODEL_ENABLED=false`
as a conservative action-only default.

## Notes

- On NPU the trainer automatically applies two small runtime patches and
  keeps the action head in FP32. One patch computes DeepSpeed's gradient
  norms in float32 (stock DeepSpeed casts each gradient to float64, which
  Ascend does not support well); the other routes Qwen RMSNorm through
  `npu_rms_norm`. On CUDA none of this activates and training behaves
  exactly as before, including the stock float64 grad-norm path.
- If startup seems to hang, it is usually first-time kernel compilation or
  dataset indexing; the launcher already raises the HCCL timeouts to
  accommodate this. Use `DRY_RUN=1` to sanity-check paths first.
- When sharing a machine with another NPU job, give each job its own HCCL
  port ranges (`HCCL_IF_BASE_PORT`, `HCCL_HOST_SOCKET_PORT_RANGE`,
  `HCCL_NPU_SOCKET_PORT_RANGE`), or the second job will fail to bind its
  communication ports.
