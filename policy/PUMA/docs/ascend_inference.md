# Running PUMA on Huawei Ascend NPUs

PUMA supports Huawei Ascend NPUs natively. You can take a PUMA checkpoint
trained on NVIDIA GPUs and serve it on an Ascend machine as-is — the weights
are loaded without any conversion, and nothing about the CUDA workflow
changes. All Ascend-specific behavior is switched on by a single flag
(`--device npu`) at serving time; if the environment cannot support it, the
server fails at startup instead of serving a misconfigured model.

Ascend training is available in this repository — see
[ascend_training.md](ascend_training.md) for the training recipe.

---

## Verified Environment

We develop and validate the Ascend support on the following stack:

| Component | Version |
| --- | --- |
| NPU | Atlas 910 series |
| CANN | 8.5.2 |
| torch | 2.5.1 |
| torch-npu | 2.5.1.post1 |
| transformers | 4.57.0 |
| Python | 3.10 |

Nearby CANN / torch-npu combinations will likely work, but only the stack
above has been verified end-to-end. Make sure your CANN version matches the
requirement of your `torch-npu` build (see the
[torch-npu compatibility table](https://gitee.com/ascend/pytorch#安装)).

## Installation

```bash
# 1. Source the CANN toolkit environment.
#    The path depends on where CANN is installed on your machine; the
#    default location for a standard installation is:
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. Create the environment and install the pinned Ascend runtime
conda create -n puma_ascend python=3.10 -y
conda activate puma_ascend
cd policy/PUMA
pip install -r requirements-ascend.txt
pip install --no-build-isolation -e .

# 3. Evaluation communication protocol (same as on CUDA)
pip install -r examples/Robotwin/eval_files/requirements.txt
```

A few things to watch out for:

- **Don't install `flash-attn`, `decord`, or `eva-decord`** in this
  environment. They pull CUDA wheels and break the install;
  `requirements-ascend.txt` deliberately leaves them out.
- Install `requirements-ascend.txt` *before* anything else, so pip doesn't
  resolve CUDA builds of torch first.
- Keep `numpy==1.26.4`. Installing packages such as `supervision` or
  `opencv-python` afterwards may silently upgrade NumPy to 2.x, which
  breaks the pinned torch-npu runtime — re-pin it if that happens.
- Model weights are exactly the same as in the CUDA setup — follow the
  [download instructions](../README.md#12-download-pre-trained-weights) and
  place or symlink them under `policy/PUMA/playground/Pretrained_models`.

## Serving a Checkpoint

Start the policy server the same way you would on a GPU, just with
`--device npu`:

```bash
cd policy/PUMA
python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/checkpoints/steps_100000_pytorch_model.pt \
  --port 9001 \
  --device npu \
  --use_bf16
```

Or set `device=npu` in `examples/Robotwin/eval_files/run_policy_server.sh`
and launch it as usual. Pick the NPU with `ASCEND_RT_VISIBLE_DEVICES=<id>`
(the Ascend counterpart of `CUDA_VISIBLE_DEVICES`).

Expect the very first request to be noticeably slower than the rest: CANN
compiles operators on first use and caches them in a `kernel_meta/`
directory under the working directory. Subsequent requests run at full
speed, and the cache is reused across restarts.

Everything on the simulation side stays the same: follow the
[evaluation steps](../README.md#3-evaluation) in the PUMA README and point
`deploy_policy.yml` at this server's host and port.
