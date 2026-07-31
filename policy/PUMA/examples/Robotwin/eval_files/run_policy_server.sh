#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
export puma_python=/path/to/puma/bin/python
your_ckpt=/path/to/your/checkpoint/steps_100000_pytorch_model.pt
gpu_id=0
port=9001
device=cuda   # cuda | npu (Ascend inference, see docs/ascend_inference.md)

CUDA_VISIBLE_DEVICES=$gpu_id ${puma_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --device ${device} \
    --use_bf16
