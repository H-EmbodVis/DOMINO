# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# This file is adopted from the starVLA repository (https://github.com/starVLA/starVLA).

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from PUMA.model.framework.base_framework import baseframework
from PUMA.model.modules.vlm.ascend import ascend_inference_config_overrides
from PUMA.util.device import resolve_device
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    device = resolve_device(args.device)
    on_npu = torch.device(device).type == "npu"
    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
        config_overrides=ascend_inference_config_overrides() if on_npu else None,
    )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()
    if on_npu:
        configure_action_precision = getattr(vla, "configure_action_model_precision", None)
        if callable(configure_action_precision):
            configure_action_precision()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
