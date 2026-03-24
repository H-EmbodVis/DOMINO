#!/usr/bin/env python3
"""
Extract dynamic target-object GT from raw RoboTwin data via seed replay.

Output for each episode:
  - dynamic_motion_info
  - trajectory_params
  - pose7_sim (simulation-step trajectory)
  - pose7_frame_aligned (resampled to episode frame count)

Quaternion order is SAPIEN order: [qw, qx, qy, qz].
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import gc
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py  # pyright: ignore[reportMissingImports]
import numpy as np
import sapien.core as sapien  # pyright: ignore[reportMissingImports]
import yaml


SUPPORTED_TRAJECTORY_TYPES = {"velocity", "trajectory", "segmented"}

# Keep in sync with policy/openvla-oft/preprocess_aloha.sh DEFAULT_TASKS.
DEFAULT_35_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_can_basket",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
]


def ensure_repo_root() -> Path:
    """Ensure repo root is importable and active cwd."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))
    return repo_root


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def resolve_repo_path(repo_root: Path, path_str: str) -> str:
    if path_str.startswith("./"):
        return str((repo_root / path_str[2:]).resolve())
    return path_str


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    config_path = Path(robot_file) / "config.yml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def create_task_instance(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def build_task_args(repo_root: Path, task_name: str, task_config: str, raw_data_dir: Path) -> dict[str, Any]:
    cfg_path = repo_root / "task_config" / f"{task_config}.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Task config not found: {cfg_path}")

    args = load_yaml(cfg_path)
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    if not isinstance(embodiment_type, list) or len(embodiment_type) not in (1, 3):
        raise ValueError(f"Unexpected embodiment config in {cfg_path}: {embodiment_type}")

    embodiment_cfg = load_yaml(repo_root / "task_config" / "_embodiment_config.yml")

    def get_embodiment_file(name: str) -> str:
        if name not in embodiment_cfg:
            raise KeyError(f"Embodiment '{name}' missing in _embodiment_config.yml")
        file_path = embodiment_cfg[name].get("file_path")
        if not file_path:
            raise ValueError(f"Embodiment '{name}' has empty file_path")
        return resolve_repo_path(repo_root, file_path)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    else:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment_type[0]}+{embodiment_type[1]}"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    args["save_path"] = str(raw_data_dir)
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = False
    return args


def safe_close_env(env: Any, clear_cache: bool = False) -> None:
    try:
        env.close_env(clear_cache=bool(clear_cache))
    except Exception:
        pass
    try:
        env.release_episode_resources()
    except Exception:
        pass


def read_seed_list(seed_path: Path) -> list[int]:
    if not seed_path.exists():
        raise FileNotFoundError(f"seed.txt not found: {seed_path}")
    raw = seed_path.read_text(encoding="utf-8")
    vals = [x for x in re.split(r"\s+", raw.strip()) if x]
    return [int(x) for x in vals]


def list_episode_indices(raw_data_dir: Path) -> list[int]:
    data_dir = raw_data_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    indices = []
    pattern = re.compile(r"^episode(\d+)\.hdf5$")
    for name in os.listdir(data_dir):
        m = pattern.match(name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def episode_frame_count(raw_data_dir: Path, episode_idx: int) -> int:
    ep_path = raw_data_dir / "data" / f"episode{episode_idx}.hdf5"
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode HDF5 not found: {ep_path}")
    with h5py.File(ep_path, "r") as f:
        if "joint_action" in f and "vector" in f["joint_action"]:
            return int(f["joint_action"]["vector"].shape[0])
        if "joint_action" in f and "left_arm" in f["joint_action"]:
            return int(f["joint_action"]["left_arm"].shape[0])
    raise RuntimeError(f"Unable to infer frame count from {ep_path}")


def load_dynamic_info_from_traj(raw_data_dir: Path, episode_idx: int) -> dict[str, Any] | None:
    traj_path = raw_data_dir / "_traj_data" / f"episode{episode_idx}.pkl"
    if not traj_path.exists():
        return None
    with traj_path.open("rb") as f:
        data = pickle.load(f)
    dynamic_info = data.get("dynamic_motion_info")
    if isinstance(dynamic_info, dict):
        return copy.deepcopy(dynamic_info)
    return None


def planning_replay_dynamic_info(
    task_name: str,
    base_args: dict[str, Any],
    episode_idx: int,
    seed: int,
    clear_cache_on_close: bool = False,
) -> dict[str, Any]:
    env = create_task_instance(task_name)
    try:
        args = copy.deepcopy(base_args)
        args["need_plan"] = True
        args["save_data"] = False
        args["render_freq"] = 0
        env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
        env.play_once()
        saved_info = getattr(env, "_saved_dynamic_motion_info", None)
        if not isinstance(saved_info, dict):
            raise RuntimeError("No _saved_dynamic_motion_info generated in planning replay")
        return copy.deepcopy(saved_info)
    finally:
        safe_close_env(env, clear_cache=clear_cache_on_close)


def actor_pose7(target_actor: Any) -> np.ndarray:
    pose = target_actor.get_pose()
    return np.asarray(pose.p.tolist() + pose.q.tolist(), dtype=np.float64)


def simulate_pose7_from_dynamic_info(
    task_name: str,
    base_args: dict[str, Any],
    episode_idx: int,
    seed: int,
    dynamic_info: dict[str, Any],
    clear_cache_on_close: bool = False,
) -> tuple[np.ndarray, float, bool]:
    env = create_task_instance(task_name)
    try:
        args = copy.deepcopy(base_args)
        args["need_plan"] = False
        args["save_data"] = False
        args["render_freq"] = 0

        env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
        dynamic_cfg = env.get_dynamic_motion_config()
        if not dynamic_cfg or "target_actor" not in dynamic_cfg:
            raise RuntimeError("Task does not expose target_actor via get_dynamic_motion_config()")

        target_actor = dynamic_cfg["target_actor"]
        start_position = np.asarray(dynamic_info["start_position"], dtype=np.float64)
        original_orientation = np.asarray(dynamic_info["original_orientation"], dtype=np.float64)
        kinematic_duration = float(dynamic_info["kinematic_duration"])
        trajectory_params = dynamic_info.get("trajectory_params")
        if not isinstance(trajectory_params, dict):
            raise RuntimeError("dynamic_motion_info missing trajectory_params")

        ok = env.setup_dynamic_motion_from_params(
            target_actor=target_actor,
            start_position=start_position,
            trajectory_params=trajectory_params,
            kinematic_duration=kinematic_duration,
        )
        if not ok:
            raise RuntimeError("setup_dynamic_motion_from_params returned False")

        # Keep behavior consistent with execute_dynamic_workflow's loaded-trajectory path:
        # set actor to saved start pose after registering kinematic task.
        target_actor.actor.set_pose(sapien.Pose(p=start_position.tolist(), q=original_orientation.tolist()))

        dt = float(env.scene.get_timestep())
        total_steps = max(1, int(np.ceil(max(kinematic_duration, dt) / dt)))
        max_steps = max(total_steps * 3, total_steps + 1000)

        poses = [actor_pose7(target_actor)]
        step_count = 0
        loop_guard_hit = False
        while step_count < total_steps or bool(getattr(env, "active_kinematic_tasks", [])):
            env._update_kinematic_tasks()
            env.scene.step()
            poses.append(actor_pose7(target_actor))
            step_count += 1
            if step_count >= max_steps:
                loop_guard_hit = True
                break

        return np.asarray(poses, dtype=np.float64), dt, loop_guard_hit
    finally:
        safe_close_env(env, clear_cache=clear_cache_on_close)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-12, norm, 1.0)
    out = quat / safe
    if out.ndim == 1 and np.linalg.norm(quat) <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if out.ndim == 2:
        bad = (norm[:, 0] <= 1e-12)
        out[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out


def resample_pose7(pose7: np.ndarray, target_len: int) -> np.ndarray:
    if pose7.ndim != 2 or pose7.shape[1] != 7:
        raise ValueError(f"Expected pose7 shape [N,7], got {pose7.shape}")
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")
    if pose7.shape[0] == 1:
        return np.repeat(pose7, target_len, axis=0)

    src_len = pose7.shape[0]
    src_t = np.linspace(0.0, float(src_len - 1), target_len)
    i0 = np.floor(src_t).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, src_len - 1)
    alpha = (src_t - i0).reshape(-1, 1)

    p0 = pose7[i0, :3]
    p1 = pose7[i1, :3]
    pos = (1.0 - alpha) * p0 + alpha * p1

    q0 = normalize_quat(pose7[i0, 3:])
    q1 = normalize_quat(pose7[i1, 3:])
    dots = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dots < 0.0, -q1, q1)
    quat = normalize_quat((1.0 - alpha) * q0 + alpha * q1)

    return np.concatenate([pos, quat], axis=1)


def sanity_checks(
    dynamic_info: dict[str, Any],
    pose7_sim: np.ndarray,
    pose7_aligned: np.ndarray,
    frame_count: int,
) -> tuple[list[str], dict[str, float]]:
    warnings: list[str] = []
    metrics: dict[str, float] = {}

    traj_params = dynamic_info.get("trajectory_params")
    traj_type = traj_params.get("type") if isinstance(traj_params, dict) else None
    if traj_type not in SUPPORTED_TRAJECTORY_TYPES:
        warnings.append(f"unsupported trajectory type: {traj_type}")

    start_position = np.asarray(dynamic_info.get("start_position", [np.nan, np.nan, np.nan]), dtype=np.float64)
    start_err = float(np.linalg.norm(pose7_sim[0, :3] - start_position))
    metrics["start_position_l2_error"] = start_err
    if start_err > 1e-3:
        warnings.append(f"large start pose error: {start_err:.6f}")

    if pose7_aligned.shape[0] != frame_count:
        warnings.append(f"aligned length mismatch: {pose7_aligned.shape[0]} vs {frame_count}")

    if not np.isfinite(pose7_sim).all():
        warnings.append("pose7_sim contains non-finite values")
    if not np.isfinite(pose7_aligned).all():
        warnings.append("pose7_frame_aligned contains non-finite values")

    sim_quat_norm = np.linalg.norm(pose7_sim[:, 3:], axis=1)
    aligned_quat_norm = np.linalg.norm(pose7_aligned[:, 3:], axis=1)
    metrics["sim_quat_norm_min"] = float(np.min(sim_quat_norm))
    metrics["sim_quat_norm_max"] = float(np.max(sim_quat_norm))
    metrics["aligned_quat_norm_min"] = float(np.min(aligned_quat_norm))
    metrics["aligned_quat_norm_max"] = float(np.max(aligned_quat_norm))
    if np.max(np.abs(sim_quat_norm - 1.0)) > 1e-3:
        warnings.append("pose7_sim quaternion norm deviates from 1")
    if np.max(np.abs(aligned_quat_norm - 1.0)) > 1e-3:
        warnings.append("pose7_frame_aligned quaternion norm deviates from 1")

    travel_distance = np.linalg.norm(np.diff(pose7_sim[:, :3], axis=0), axis=1).sum() if pose7_sim.shape[0] > 1 else 0.0
    metrics["sim_total_travel_distance"] = float(travel_distance)

    return warnings, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract dynamic GT via raw seed replay (single or batch)")

    # Task selection
    parser.add_argument("--task_name", type=str, default="", help="Single task name, e.g. adjust_bottle")
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="Comma-separated task names, e.g. adjust_bottle,click_bell",
    )
    parser.add_argument(
        "--all_35tasks",
        action="store_true",
        help="Process built-in 35-task list in one run",
    )

    # Shared config
    parser.add_argument(
        "--task_config",
        type=str,
        default="aloha-agilex_clean_level1",
        help="Task config name without extension, used for environment setup",
    )

    # Data path modes:
    # 1) Single-task explicit mode: --raw_data_dir [+ --output_dir]
    # 2) Root mode: --raw_root/{task}/{raw_setting} -> --output_root/{task}/epxxx
    parser.add_argument("--raw_data_dir", type=str, default="", help="Single task raw dataset dir override")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="/home/hfang/code/Dynamic_RoboTwin/data",
        help="Raw dataset root for batch mode",
    )
    parser.add_argument(
        "--raw_setting",
        type=str,
        default="aloha-agilex_clean_50",
        help="Raw setting folder under each task, e.g. aloha-agilex_clean_50",
    )

    parser.add_argument("--output_dir", type=str, default="", help="Single task output dir override")
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/hfang/code/Dynamic_RoboTwin/converted_data/dynamic_gt",
        help="Batch output root; per-task output goes to {output_root}/{task}/epxxx",
    )

    # Episode control
    parser.add_argument("--episode_start", type=int, default=0, help="Start episode index (inclusive)")
    parser.add_argument("--episode_end", type=int, default=-1, help="End episode index (inclusive), -1 means all")

    # Runtime options
    parser.add_argument(
        "--force_replay",
        action="store_true",
        help="Force replay even if dynamic_motion_info exists in _traj_data",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip episode if output {task}/epxxx/gt.pkl already exists",
    )
    parser.add_argument(
        "--clear_cache_freq",
        type=int,
        default=3,
        help="Force close_env(clear_cache=True) every N episodes; <=0 disables periodic forcing",
    )

    # Parallel scheduling (multi-thread launcher + per-task subprocess).
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=4,
        help="Parallel task workers. >1 enables multi-thread launcher for multi-task runs.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="4,5,6,7",
        help="Comma-separated GPU IDs used by launcher, e.g. 0,1,2,3,4,5,6,7",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Optional directory for per-task launcher logs. Default: {output_root}/_logs",
    )
    parser.add_argument(
        "--gpu_error_retries",
        type=int,
        default=2,
        help="Retry count for SAPIEN GPU device errors per episode",
    )
    parser.add_argument(
        "--fallback_subprocess_retries",
        type=int,
        default=3,
        help="Attempts for per-episode fallback subprocess before declaring failure",
    )
    parser.add_argument(
        "--disable_gpu_subprocess_fallback",
        action="store_true",
        help="Disable per-episode subprocess fallback when GPU device error persists",
    )
    parser.add_argument(
        "--task_requeue_limit",
        type=int,
        default=2,
        help="Pool mode: max requeue times for an incomplete/failed task",
    )
    return parser.parse_args()


def parse_task_names(task_names_arg: str) -> list[str]:
    if not task_names_arg.strip():
        return []
    names = []
    for token in task_names_arg.replace(" ", ",").split(","):
        token = token.strip()
        if token:
            names.append(token)
    return names


def resolve_task_list(args: argparse.Namespace) -> list[str]:
    explicit = []
    if args.task_name:
        explicit.append(args.task_name.strip())
    explicit.extend(parse_task_names(args.task_names))

    if args.all_35tasks:
        if explicit:
            raise ValueError("Do not combine --all_35tasks with --task_name/--task_names")
        return list(DEFAULT_35_TASKS)

    if not explicit:
        raise ValueError("Please specify one of: --task_name / --task_names / --all_35tasks")

    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for name in explicit:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def parse_gpu_ids(gpu_ids_arg: str) -> list[str]:
    vals = [x.strip() for x in gpu_ids_arg.split(",") if x.strip()]
    if not vals:
        raise ValueError("--gpu_ids is empty")
    for v in vals:
        if not re.fullmatch(r"\d+", v):
            raise ValueError(f"Invalid gpu id '{v}' in --gpu_ids")
    return vals


def episode_output_file(task_output_dir: Path, episode_idx: int) -> Path:
    # Requested layout: .../dynamic_gt/{task_name}/epxxx/gt.pkl
    ep_dir = task_output_dir / f"ep{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    return ep_dir / "gt.pkl"


def is_cuda_device_error(exc: Exception) -> bool:
    msg = str(exc)
    lower_msg = msg.lower()
    if "failed to find a supported physical device" in lower_msg and "cuda:" in lower_msg:
        return True
    if "cuda error" in lower_msg and ("device" in lower_msg and "not" in lower_msg):
        return True
    return False


def cleanup_gpu_resources() -> None:
    try:
        from sapien.render import clear_cache as sapien_clear_cache  # pyright: ignore[reportMissingImports]
        sapien_clear_cache()
    except Exception:
        pass
    gc.collect()


def compute_avg_speed_mps(pose7_sim: np.ndarray, sim_timestep: float) -> float:
    if pose7_sim.shape[0] <= 1:
        return 0.0
    travel_distance = float(np.linalg.norm(np.diff(pose7_sim[:, :3], axis=0), axis=1).sum())
    total_time = float((pose7_sim.shape[0] - 1) * sim_timestep)
    if total_time <= 1e-12:
        return 0.0
    return travel_distance / total_time


def load_result_from_output_file(out_file: Path) -> dict[str, Any]:
    with out_file.open("rb") as f:
        payload = pickle.load(f)
    pose7_sim = np.asarray(payload["pose7_sim"])
    sim_timestep = float(payload["sim_timestep"])
    avg_speed = compute_avg_speed_mps(pose7_sim, sim_timestep)
    return {
        "status": "success",
        "source_dynamic_info": payload.get("source_dynamic_info", "raw_replay"),
        "trajectory_type": payload.get("trajectory_params", {}).get("type"),
        "pose7_sim_steps": int(pose7_sim.shape[0]),
        "output_file": str(out_file),
        "sanity_warnings": payload.get("sanity_warnings", []),
        "sanity_metrics": payload.get("sanity_metrics", {}),
        "avg_speed_mps": float(avg_speed),
    }


def failed_episode_indices_from_summary(summary: dict[str, Any] | None) -> list[int]:
    if not isinstance(summary, dict):
        return []
    failed_eps: list[int] = []
    for rec in summary.get("results", []):
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "failed":
            continue
        ep = rec.get("episode_idx")
        if isinstance(ep, int):
            failed_eps.append(ep)
    return sorted(set(failed_eps))


def write_failed_episode_retry_list(
    output_root: Path,
    task_names: list[str],
    task_summaries: list[dict[str, Any]],
    global_failures: list[dict[str, Any]],
) -> tuple[Path, int]:
    summary_by_task: dict[str, dict[str, Any]] = {}
    for s in task_summaries:
        task_name = s.get("task_name")
        if isinstance(task_name, str) and task_name:
            summary_by_task[task_name] = s

    failure_by_task: dict[str, dict[str, Any]] = {}
    for f in global_failures:
        task_name = f.get("task_name")
        if isinstance(task_name, str) and task_name:
            failure_by_task[task_name] = f

    lines: list[str] = []
    for task_name in task_names:
        summary = summary_by_task.get(task_name)
        failed_eps = failed_episode_indices_from_summary(summary)
        if failed_eps:
            lines.append(f"{task_name}:{','.join(str(x) for x in failed_eps)}")
            continue

        failure = failure_by_task.get(task_name)
        if failure is None:
            continue
        failure_eps = failure.get("failed_episode_indices")
        if isinstance(failure_eps, list) and failure_eps:
            lines.append(f"{task_name}:{','.join(str(int(x)) for x in failure_eps if isinstance(x, int))}")
        else:
            lines.append(f"{task_name}:ALL")

    # stable de-dup
    uniq_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            uniq_lines.append(line)

    out_path = output_root / "failed_episodes_retry_list.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Failed episodes to retry\n")
        f.write("# Format: task_name:ep1,ep2,...  or  task_name:ALL\n")
        if uniq_lines:
            for line in uniq_lines:
                f.write(f"{line}\n")
        else:
            f.write("# empty\n")
    return out_path, len(uniq_lines)


def run_episode_subprocess_fallback(
    script_path: Path,
    task_name: str,
    task_config: str,
    raw_data_dir: Path,
    task_output_dir: Path,
    episode_idx: int,
    force_replay: bool,
    parent_skip_existing: bool,
    clear_cache_freq: int,
    fallback_subprocess_retries: int,
    fallback_cuda_visible_devices: str = "",
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--task_name",
        task_name,
        "--task_config",
        task_config,
        "--raw_data_dir",
        str(raw_data_dir),
        "--output_dir",
        str(task_output_dir),
        "--episode_start",
        str(episode_idx),
        "--episode_end",
        str(episode_idx),
        "--parallel_workers",
        "1",
        "--gpu_error_retries",
        "0",
        "--clear_cache_freq",
        str(clear_cache_freq),
        "--disable_gpu_subprocess_fallback",
    ]
    if force_replay:
        cmd.append("--force_replay")
    if parent_skip_existing:
        cmd.append("--skip_existing")

    env = os.environ.copy()
    if fallback_cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = fallback_cuda_visible_devices.strip()
    env.setdefault("PYTHONUNBUFFERED", "1")
    max_attempts = max(1, int(fallback_subprocess_retries))
    last_rc = -1
    last_tail = ""
    for attempt in range(max_attempts):
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return True, ""
        last_rc = int(proc.returncode)
        last_tail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-1200:]
        print(
            f"    - fallback_subprocess_failed attempt {attempt + 1}/{max_attempts} "
            f"(code={last_rc}), will retry"
        )
        cleanup_gpu_resources()
        if attempt < max_attempts - 1:
            time.sleep(0.5 * (attempt + 1))
    return False, (
        "fallback_subprocess_failed "
        f"attempts={max_attempts}, code={last_rc}, tail={last_tail}"
    )


def run_single_task(
    script_path: Path,
    repo_root: Path,
    task_name: str,
    task_config: str,
    raw_data_dir: Path,
    task_output_dir: Path,
    episode_start: int,
    episode_end: int,
    force_replay: bool,
    skip_existing: bool,
    gpu_error_retries: int,
    clear_cache_freq: int,
    fallback_subprocess_retries: int,
    disable_gpu_subprocess_fallback: bool,
) -> dict[str, Any]:
    task_output_dir.mkdir(parents=True, exist_ok=True)

    seeds = read_seed_list(raw_data_dir / "seed.txt")
    all_indices = list_episode_indices(raw_data_dir)
    if not all_indices:
        raise RuntimeError(f"No episode*.hdf5 found in {raw_data_dir / 'data'}")

    end_idx = episode_end if episode_end >= 0 else max(all_indices)
    selected = [idx for idx in all_indices if episode_start <= idx <= end_idx]
    if not selected:
        raise RuntimeError(f"No episode in range [{episode_start}, {end_idx}] under {raw_data_dir / 'data'}")

    base_task_args = build_task_args(repo_root, task_name, task_config, raw_data_dir)

    print("=" * 80)
    print(f"Dynamic GT extraction | task={task_name}")
    print("=" * 80)
    print(f"task_config:    {task_config}")
    print(f"raw_data_dir:   {raw_data_dir}")
    print(f"output_dir:     {task_output_dir}")
    print(f"episodes:       {selected[:5]}{' ...' if len(selected) > 5 else ''} (total={len(selected)})")
    print(f"force_replay:   {force_replay}")
    print(f"skip_existing:  {skip_existing}")
    print(f"clear_cache_freq: {clear_cache_freq}")
    print("quat_order:     wxyz (SAPIEN)")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, episode_idx in enumerate(selected):
        t0 = time.time()
        result: dict[str, Any] = {"episode_idx": episode_idx}
        print(f"[{i + 1}/{len(selected)}] {task_name} episode={episode_idx}")
        force_clear_cache_on_close = bool(clear_cache_freq > 0 and ((i + 1) % clear_cache_freq == 0))
        if force_clear_cache_on_close:
            print(f"  - info | forcing clear_cache=True on episode close (freq={clear_cache_freq})")
        try:
            if episode_idx >= len(seeds):
                raise RuntimeError(f"seed missing for episode {episode_idx}; seed.txt has {len(seeds)} entries")
            seed = int(seeds[episode_idx])
            result["seed"] = seed

            out_file = episode_output_file(task_output_dir, episode_idx)
            if skip_existing and out_file.exists():
                result.update(
                    {
                        "status": "skipped",
                        "reason": "output_exists",
                        "output_file": str(out_file),
                    }
                )
                skip_count += 1
                print(f"  - skipped | output exists: {out_file}")
                result["elapsed_sec"] = round(time.time() - t0, 4)
                results.append(result)
                continue

            frame_count = episode_frame_count(raw_data_dir, episode_idx)
            result["frame_count"] = frame_count

            def process_episode_once() -> dict[str, Any]:
                dynamic_info = None
                source = "raw_replay"
                if not force_replay:
                    dynamic_info = load_dynamic_info_from_traj(raw_data_dir, episode_idx)
                    if dynamic_info is not None:
                        source = "_traj_data"

                if dynamic_info is None:
                    dynamic_info = planning_replay_dynamic_info(
                        task_name,
                        base_task_args,
                        episode_idx,
                        seed,
                        clear_cache_on_close=force_clear_cache_on_close,
                    )

                traj_params = dynamic_info.get("trajectory_params")
                if not isinstance(traj_params, dict):
                    raise RuntimeError("dynamic_info has no trajectory_params")
                trajectory_type = traj_params.get("type")

                pose7_sim, sim_timestep, loop_guard_hit = simulate_pose7_from_dynamic_info(
                    task_name,
                    base_task_args,
                    episode_idx,
                    seed,
                    dynamic_info,
                    clear_cache_on_close=force_clear_cache_on_close,
                )
                pose7_frame_aligned = resample_pose7(pose7_sim, frame_count)
                warnings, metrics = sanity_checks(dynamic_info, pose7_sim, pose7_frame_aligned, frame_count)
                if loop_guard_hit:
                    warnings.append("simulation loop guard triggered before kinematic task cleanup")

                avg_speed_mps = compute_avg_speed_mps(pose7_sim, sim_timestep)

                payload = {
                    "task_name": task_name,
                    "task_config": task_config,
                    "raw_data_dir": str(raw_data_dir),
                    "episode_idx": episode_idx,
                    "seed": seed,
                    "quat_order": "wxyz",
                    "sim_timestep": sim_timestep,
                    "frame_count": frame_count,
                    "source_dynamic_info": source,
                    "dynamic_motion_info": dynamic_info,
                    "trajectory_params": traj_params,
                    "pose7_sim": pose7_sim,
                    "pose7_frame_aligned": pose7_frame_aligned,
                    "sanity_metrics": metrics,
                    "sanity_warnings": warnings,
                    "avg_speed_mps": float(avg_speed_mps),
                }
                with out_file.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

                return {
                    "status": "success",
                    "source_dynamic_info": source,
                    "trajectory_type": trajectory_type,
                    "pose7_sim_steps": int(pose7_sim.shape[0]),
                    "output_file": str(out_file),
                    "sanity_warnings": warnings,
                    "sanity_metrics": metrics,
                    "avg_speed_mps": float(avg_speed_mps),
                }

            processed = False
            last_err: Exception | None = None
            max_attempts = max(1, 1 + int(gpu_error_retries))

            for attempt in range(max_attempts):
                try:
                    episode_result = process_episode_once()
                    result.update(episode_result)
                    success_count += 1
                    print(
                        f"  - ok | seed={seed} | traj_type={episode_result.get('trajectory_type')} | "
                        f"sim_steps={episode_result.get('pose7_sim_steps')} | "
                        f"avg_speed={episode_result.get('avg_speed_mps', 0.0):.4f} m/s | "
                        f"warnings={len(episode_result.get('sanity_warnings', []))}"
                    )
                    processed = True
                    break
                except Exception as exc:
                    last_err = exc
                    if is_cuda_device_error(exc) and attempt < max_attempts - 1:
                        print(
                            f"  - warn | GPU runtime error, retry {attempt + 1}/{max_attempts - 1} "
                            f"after cleanup: {exc}"
                        )
                        cleanup_gpu_resources()
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    break

            if not processed and last_err is not None and is_cuda_device_error(last_err) and not disable_gpu_subprocess_fallback:
                print("  - warn | switching to per-episode subprocess fallback")
                ok, err_msg = run_episode_subprocess_fallback(
                    script_path=script_path,
                    task_name=task_name,
                    task_config=task_config,
                    raw_data_dir=raw_data_dir,
                    task_output_dir=task_output_dir,
                    episode_idx=episode_idx,
                    force_replay=force_replay,
                    parent_skip_existing=skip_existing,
                    clear_cache_freq=clear_cache_freq,
                    fallback_subprocess_retries=fallback_subprocess_retries,
                    fallback_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                )
                if ok and out_file.exists():
                    episode_result = load_result_from_output_file(out_file)
                    result.update(episode_result)
                    success_count += 1
                    print(
                        f"  - ok(fallback) | seed={seed} | traj_type={episode_result.get('trajectory_type')} | "
                        f"sim_steps={episode_result.get('pose7_sim_steps')} | "
                        f"avg_speed={episode_result.get('avg_speed_mps', 0.0):.4f} m/s | "
                        f"warnings={len(episode_result.get('sanity_warnings', []))}"
                    )
                    processed = True
                else:
                    raise RuntimeError(err_msg or "subprocess fallback failed without error detail")

            if not processed and last_err is not None:
                raise last_err
        except Exception as exc:
            fail_count += 1
            result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  - failed | {type(exc).__name__}: {exc}")

        result["elapsed_sec"] = round(time.time() - t0, 4)
        results.append(result)

    summary = {
        "task_name": task_name,
        "task_config": task_config,
        "raw_data_dir": str(raw_data_dir),
        "output_dir": str(task_output_dir),
        "quat_order": "wxyz",
        "episodes_total_selected": len(selected),
        "episodes_success": success_count,
        "episodes_failed": fail_count,
        "episodes_skipped": skip_count,
        "force_replay": bool(force_replay),
        "skip_existing": bool(skip_existing),
        "results": results,
    }

    episode_avg_speed_mps: dict[str, float] = {}
    for rec in results:
        if rec.get("status") == "success" and rec.get("avg_speed_mps") is not None:
            episode_avg_speed_mps[str(rec["episode_idx"])] = float(rec["avg_speed_mps"])
    summary["episode_avg_speed_mps"] = episode_avg_speed_mps

    summary_path = task_output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Task finished: {task_name} | success={success_count} failed={fail_count} skipped={skip_count}")
    print(f"Task summary: {summary_path}")
    return summary


def build_single_task_command(
    script_path: Path,
    task_name: str,
    task_config: str,
    raw_data_dir: Path,
    task_output_dir: Path,
    episode_start: int,
    episode_end: int,
    force_replay: bool,
    skip_existing: bool,
    gpu_error_retries: int,
    clear_cache_freq: int,
    fallback_subprocess_retries: int,
    disable_gpu_subprocess_fallback: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--task_name",
        task_name,
        "--task_config",
        task_config,
        "--raw_data_dir",
        str(raw_data_dir),
        "--output_dir",
        str(task_output_dir),
        "--episode_start",
        str(episode_start),
        "--episode_end",
        str(episode_end),
        "--parallel_workers",
        "1",
        "--gpu_error_retries",
        str(gpu_error_retries),
        "--clear_cache_freq",
        str(clear_cache_freq),
        "--fallback_subprocess_retries",
        str(fallback_subprocess_retries),
    ]
    if force_replay:
        cmd.append("--force_replay")
    if skip_existing:
        cmd.append("--skip_existing")
    if disable_gpu_subprocess_fallback:
        cmd.append("--disable_gpu_subprocess_fallback")
    return cmd


def run_parallel_launcher(
    script_path: Path,
    task_names: list[str],
    task_config: str,
    raw_root: Path,
    raw_setting: str,
    output_root: Path,
    episode_start: int,
    episode_end: int,
    force_replay: bool,
    skip_existing: bool,
    parallel_workers: int,
    gpu_ids: list[str],
    log_dir: Path,
    gpu_error_retries: int,
    clear_cache_freq: int,
    fallback_subprocess_retries: int,
    disable_gpu_subprocess_fallback: bool,
    task_requeue_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fixed-size slot pool:
    - one running subprocess per slot/GPU
    - monitor each subprocess pid and recycle freed slots
    - requeue unfinished tasks to queue tail
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    task_requeue_limit = max(0, int(task_requeue_limit))

    n_workers = max(1, parallel_workers)
    n_workers = min(n_workers, len(gpu_ids), len(task_names))
    if n_workers <= 1 or len(task_names) <= 1:
        raise ValueError("run_parallel_launcher requires workers>1 and multiple tasks")

    active_gpu_ids = gpu_ids[:n_workers]
    pending_queue: deque[dict[str, Any]] = deque({"task_name": t, "attempt": 0} for t in task_names)
    slot_states: list[dict[str, Any] | None] = [None for _ in range(n_workers)]
    summary_by_task: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = []

    print("=" * 80)
    print("Parallel slot pool enabled")
    print("=" * 80)
    print(f"slots:          {n_workers}")
    print(f"gpu_ids:        {active_gpu_ids}")
    print(f"task_count:     {len(task_names)}")
    print(f"task_requeue:   {task_requeue_limit}")
    print(f"log_dir:        {log_dir}")
    print("=" * 80)

    def maybe_load_summary(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def queue_retry_or_record_failure(
        task_name: str,
        attempt: int,
        returncode: int,
        log_path: Path,
        summary: dict[str, Any] | None,
        extra_error: str = "",
    ) -> None:
        failed_eps = failed_episode_indices_from_summary(summary)
        if attempt < task_requeue_limit:
            pending_queue.append({"task_name": task_name, "attempt": attempt + 1})
            print(
                f"[task-requeue] {task_name} -> attempt {attempt + 2}/{task_requeue_limit + 1} "
                f"(code={returncode}, failed_eps={len(failed_eps)})"
            )
            return

        if summary is not None:
            summary_by_task[task_name] = summary
        all_failures.append(
            {
                "task_name": task_name,
                "error_type": "TaskIncompleteAfterRetries" if returncode == 0 else "SubprocessError",
                "error": extra_error or f"returncode={returncode}, log={log_path}",
                "returncode": int(returncode),
                "attempts_used": int(attempt + 1),
                "log_path": str(log_path),
                "failed_episode_indices": failed_eps,
            }
        )
        print(
            f"[task-failed] {task_name} exhausted retries "
            f"(attempts={attempt + 1}, code={returncode}, failed_eps={len(failed_eps)})"
        )

    def launch_task(slot_idx: int, task_item: dict[str, Any]) -> None:
        task_name = str(task_item["task_name"])
        attempt = int(task_item.get("attempt", 0))
        gpu_id = active_gpu_ids[slot_idx]
        raw_data_dir = (raw_root / task_name / raw_setting).resolve()
        task_output_dir = (output_root / task_name).resolve()
        summary_path = task_output_dir / "summary.json"
        task_log = log_dir / f"{task_name}.attempt{attempt + 1}.log"
        skip_existing_for_this_run = bool(skip_existing or attempt > 0)

        cmd = build_single_task_command(
            script_path=script_path,
            task_name=task_name,
            task_config=task_config,
            raw_data_dir=raw_data_dir,
            task_output_dir=task_output_dir,
            episode_start=episode_start,
            episode_end=episode_end,
            force_replay=force_replay,
            skip_existing=skip_existing_for_this_run,
            gpu_error_retries=gpu_error_retries,
            clear_cache_freq=clear_cache_freq,
            fallback_subprocess_retries=fallback_subprocess_retries,
            disable_gpu_subprocess_fallback=disable_gpu_subprocess_fallback,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("PYTHONUNBUFFERED", "1")

        log_handle = None
        try:
            log_handle = task_log.open("w", encoding="utf-8")
            log_handle.write(f"# cmd: {' '.join(cmd)}\n")
            log_handle.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n")
            log_handle.write(f"# attempt={attempt + 1}/{task_requeue_limit + 1}\n\n")
            log_handle.flush()
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            slot_states[slot_idx] = {
                "task_name": task_name,
                "attempt": attempt,
                "gpu_id": gpu_id,
                "proc": proc,
                "start_time": time.time(),
                "summary_path": summary_path,
                "log_path": task_log,
                "log_handle": log_handle,
            }
            print(
                f"[slot-{slot_idx}|gpu-{gpu_id}] launch {task_name} "
                f"attempt={attempt + 1}/{task_requeue_limit + 1} pid={proc.pid}"
            )
        except Exception as exc:
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass
            queue_retry_or_record_failure(
                task_name=task_name,
                attempt=attempt,
                returncode=-1,
                log_path=task_log,
                summary=None,
                extra_error=f"launch_failed: {type(exc).__name__}: {exc}",
            )

    poll_interval_sec = 1.0
    while pending_queue or any(state is not None for state in slot_states):
        made_progress = False

        # 1) Check finished subprocesses.
        for slot_idx, state in enumerate(slot_states):
            if state is None:
                continue
            proc = state["proc"]
            returncode = proc.poll()
            if returncode is None:
                continue

            made_progress = True
            task_name = str(state["task_name"])
            attempt = int(state["attempt"])
            gpu_id = str(state["gpu_id"])
            summary_path = Path(state["summary_path"])
            log_path = Path(state["log_path"])
            log_handle = state["log_handle"]
            elapsed = round(time.time() - float(state["start_time"]), 3)

            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

            summary = maybe_load_summary(summary_path)
            if summary is not None:
                summary_by_task[task_name] = summary

            failed_eps = failed_episode_indices_from_summary(summary)
            task_completed_cleanly = (
                int(returncode) == 0
                and summary is not None
                and int(summary.get("episodes_failed", 0)) == 0
            )

            if task_completed_cleanly:
                print(
                    f"[slot-{slot_idx}|gpu-{gpu_id}] done {task_name} "
                    f"in {elapsed}s (pid={proc.pid})"
                )
            else:
                queue_retry_or_record_failure(
                    task_name=task_name,
                    attempt=attempt,
                    returncode=int(returncode),
                    log_path=log_path,
                    summary=summary,
                    extra_error=(
                        f"returncode={returncode}, failed_eps={len(failed_eps)}, "
                        f"log={log_path}"
                    ),
                )

            slot_states[slot_idx] = None

        # 2) Fill idle slots from waiting queue tail/head.
        for slot_idx in range(n_workers):
            if slot_states[slot_idx] is not None or not pending_queue:
                continue
            task_item = pending_queue.popleft()
            launch_task(slot_idx, task_item)
            made_progress = True

        if not made_progress:
            time.sleep(poll_interval_sec)

    ordered_summaries = [summary_by_task[t] for t in task_names if t in summary_by_task]
    return ordered_summaries, all_failures


def main() -> None:
    args = parse_args()
    repo_root = ensure_repo_root()
    script_path = Path(__file__).resolve()
    task_names = resolve_task_list(args)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if args.raw_data_dir and len(task_names) != 1:
        raise ValueError("--raw_data_dir only supports single task mode")
    if args.output_dir and len(task_names) != 1:
        raise ValueError("--output_dir only supports single task mode")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    all_task_summaries: list[dict[str, Any]] = []
    global_failures: list[dict[str, Any]] = []

    use_parallel = (
        args.parallel_workers > 1
        and len(task_names) > 1
        and not args.raw_data_dir
        and not args.output_dir
    )

    if use_parallel:
        log_dir = Path(args.log_dir).resolve() if args.log_dir else (output_root / "_logs")
        try:
            summaries, failures = run_parallel_launcher(
                script_path=script_path,
                task_names=task_names,
                task_config=args.task_config,
                raw_root=Path(args.raw_root).resolve(),
                raw_setting=args.raw_setting,
                output_root=output_root,
                episode_start=args.episode_start,
                episode_end=args.episode_end,
                force_replay=args.force_replay,
                skip_existing=args.skip_existing,
                parallel_workers=args.parallel_workers,
                gpu_ids=gpu_ids,
                log_dir=log_dir,
                gpu_error_retries=args.gpu_error_retries,
                clear_cache_freq=args.clear_cache_freq,
                fallback_subprocess_retries=args.fallback_subprocess_retries,
                disable_gpu_subprocess_fallback=args.disable_gpu_subprocess_fallback,
                task_requeue_limit=args.task_requeue_limit,
            )
            all_task_summaries.extend(summaries)
            global_failures.extend(failures)
        except Exception as exc:
            raise RuntimeError(f"Parallel launcher failed: {exc}") from exc
    else:
        for task_idx, task_name in enumerate(task_names):
            print(f"\n\n########## [{task_idx + 1}/{len(task_names)}] task={task_name} ##########")
            try:
                if args.raw_data_dir:
                    raw_data_dir = Path(args.raw_data_dir).resolve()
                else:
                    raw_data_dir = (Path(args.raw_root).resolve() / task_name / args.raw_setting).resolve()

                if args.output_dir:
                    task_output_dir = Path(args.output_dir).resolve()
                else:
                    task_output_dir = (output_root / task_name).resolve()

                summary = run_single_task(
                    script_path=script_path,
                    repo_root=repo_root,
                    task_name=task_name,
                    task_config=args.task_config,
                    raw_data_dir=raw_data_dir,
                    task_output_dir=task_output_dir,
                    episode_start=args.episode_start,
                    episode_end=args.episode_end,
                    force_replay=args.force_replay,
                    skip_existing=args.skip_existing,
                    gpu_error_retries=args.gpu_error_retries,
                    clear_cache_freq=args.clear_cache_freq,
                    fallback_subprocess_retries=args.fallback_subprocess_retries,
                    disable_gpu_subprocess_fallback=args.disable_gpu_subprocess_fallback,
                )
                all_task_summaries.append(summary)
            except Exception as exc:
                summary_path = (output_root / task_name / "summary.json").resolve()
                partial_summary = None
                if summary_path.exists():
                    try:
                        with summary_path.open("r", encoding="utf-8") as f:
                            payload = json.load(f)
                        if isinstance(payload, dict):
                            partial_summary = payload
                            all_task_summaries.append(payload)
                    except Exception:
                        partial_summary = None
                global_failures.append(
                    {
                        "task_name": task_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "summary_path": str(summary_path) if summary_path.exists() else "",
                        "failed_episode_indices": failed_episode_indices_from_summary(partial_summary),
                    }
                )
                print(f"[TASK FAILED] {task_name}: {type(exc).__name__}: {exc}")

    total_success_tasks = sum(1 for s in all_task_summaries if s.get("episodes_failed", 0) == 0)
    global_summary = {
        "task_count_total": len(task_names),
        "task_count_completed": len(all_task_summaries),
        "task_count_no_failures": total_success_tasks,
        "task_count_failed_to_run": len(global_failures),
        "task_failures": global_failures,
        "elapsed_sec": round(time.time() - total_start, 3),
        "task_summaries": all_task_summaries,
    }
    global_summary["episode_avg_speed_mps_by_task"] = {
        s.get("task_name", f"task_{i}"): s.get("episode_avg_speed_mps", {})
        for i, s in enumerate(all_task_summaries)
    }
    retry_list_path, retry_line_count = write_failed_episode_retry_list(
        output_root=output_root,
        task_names=task_names,
        task_summaries=all_task_summaries,
        global_failures=global_failures,
    )
    global_summary["failed_episode_retry_list_path"] = str(retry_list_path)
    global_summary["failed_episode_retry_task_count"] = int(retry_line_count)

    global_summary_path = output_root / "summary_all_tasks.json"
    with global_summary_path.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("All tasks finished.")
    print(f"Global summary: {global_summary_path}")
    print(f"Failed-episode retry list: {retry_list_path} (tasks={retry_line_count})")
    print(
        f"Tasks completed: {len(all_task_summaries)}/{len(task_names)} | "
        f"failed_to_run={len(global_failures)}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
