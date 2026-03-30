import ast
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from collections import deque

import numpy as np
import cv2 as cv
import yaml


from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from PUMA.dataloader.gr00t_lerobot.history_flow_utils import (
    build_flow_cache_key,
    build_flow_cache_path,
    compute_flow_rgb_farneback,
    load_flow_cache,
    parse_hw_size,
    sample_history_offsets,
    save_flow_cache,
)
from PUMA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        history_k: int = 0,
        history_stride: int = 1,
        history_image_size: Optional[Sequence[int]] = None,
        history_flow_compute_size: Optional[Sequence[int]] = None,
        history_flow_cache: Optional[dict] = None,
        host="127.0.0.1",
        port=5694,
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.history_k = max(0, int(history_k))
        self.history_stride = max(1, int(history_stride))
        self.history_enabled = self.history_k > 0
        self.history_image_size = self._parse_history_image_size(history_image_size)
        self.history_flow_compute_size = self._parse_history_image_size(
            history_flow_compute_size if history_flow_compute_size is not None else self.history_image_size
        )
        (
            self.history_flow_cache_enabled,
            self.history_flow_cache_read,
            self.history_flow_cache_write,
            self.history_flow_cache_dirname,
            self.history_flow_cache_version,
            self.history_flow_cache_root,
        ) = self._parse_flow_cache_config(history_flow_cache)
        if self.history_flow_cache_root is not None:
            self.history_flow_cache_root.mkdir(parents=True, exist_ok=True)

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        history_buffer_len = self.history_k * self.history_stride
        self.history_frame_buffer = deque(maxlen=max(1, history_buffer_len))
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.history_frame_buffer.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        if self.history_enabled:
            history_images = self._build_history_images(current_frame=images[0], step=step)
            if history_images:
                example["history_images"] = history_images
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]
        self._push_history_frame(images)
        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    @staticmethod
    def normalize_state(state: dict[str, np.ndarray], state_norm_stats: Dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        mask = [True, True, True, True, True, True, True, True, True, True, True, True, False, False]
        mask = np.array(mask, dtype=bool)
        state_high, state_low = np.array(state_norm_stats["max"]), np.array(state_norm_stats["min"])
        normalized_state = np.where(
            mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(~mask, (normalized_state > 0.5).astype(normalized_state.dtype), normalized_state)
        return normalized_state

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        framework_cfg = model_config.get("framework", {})
        action_model_cfg = framework_cfg.get("action_model", {})

        if "future_action_window_size" in action_model_cfg:
            return int(action_model_cfg["future_action_window_size"]) + 1
        if "num_actions_chunk" in action_model_cfg:
            return int(action_model_cfg["num_actions_chunk"])
        if "action_horizon" in action_model_cfg:
            return int(action_model_cfg["action_horizon"])

        available_keys = sorted(action_model_cfg.keys()) if isinstance(action_model_cfg, dict) else []
        raise KeyError(
            "Unable to infer action chunk size from checkpoint config. "
            "Expected one of ['future_action_window_size', 'num_actions_chunk', 'action_horizon'], "
            f"but got keys: {available_keys}."
        )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _parse_history_image_size(history_image_size: Optional[Sequence[int]]) -> tuple[int, int]:
        default_size = (128, 128)
        if history_image_size is None:
            return default_size
        if isinstance(history_image_size, str):
            try:
                history_image_size = ast.literal_eval(history_image_size)
            except (ValueError, SyntaxError):
                return default_size
        if isinstance(history_image_size, (list, tuple)) and len(history_image_size) == 2:
            try:
                h = int(history_image_size[0])
                w = int(history_image_size[1])
            except (TypeError, ValueError):
                return default_size
            if h > 0 and w > 0:
                return (h, w)
        return default_size

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @classmethod
    def _parse_flow_cache_config(cls, cache_cfg: Optional[dict]):
        cache_cfg = cache_cfg or {}
        enabled = cls._as_bool(cache_cfg.get("enabled", False), False)
        cache_read = cls._as_bool(cache_cfg.get("read", True), True)
        cache_write = cls._as_bool(cache_cfg.get("write", True), True)
        cache_dirname = str(cache_cfg.get("dirname", "history_flow_cache"))
        cache_version = str(cache_cfg.get("version", "v1"))
        cache_root = cache_cfg.get("root_dir", None)
        cache_root_path = Path(cache_root).expanduser() if cache_root else None
        return enabled, cache_read, cache_write, cache_dirname, cache_version, cache_root_path

    def _build_history_images(self, current_frame: Optional[np.ndarray], step: int) -> list[np.ndarray]:
        if not self.history_enabled:
            return []
        return self._build_history_flow_images(current_frame=current_frame, step=step)

    def _get_or_compute_flow_cached(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        step: int,
        pair_index: int,
    ) -> np.ndarray:
        cache_path = None
        if self.history_flow_cache_enabled and self.history_flow_cache_root is not None:
            cache_key = build_flow_cache_key(
                dataset_path=self.history_flow_cache_root,
                dataset_name="robotwin_eval",
                trajectory_id=self.task_description or "unknown_task",
                step=int(step),
                prev_offset=int(pair_index),
                curr_offset=int(pair_index + 1),
                video_key="head_camera",
                compute_size=self.history_flow_compute_size,
                version=self.history_flow_cache_version,
            )
            cache_path = build_flow_cache_path(
                cache_root=self.history_flow_cache_root,
                cache_dirname=self.history_flow_cache_dirname,
                cache_key=cache_key,
            )
            if self.history_flow_cache_read:
                cached_flow = load_flow_cache(cache_path)
                if cached_flow is not None:
                    return cached_flow

        flow_rgb = compute_flow_rgb_farneback(
            prev_rgb=prev_frame,
            curr_rgb=curr_frame,
            compute_size=self.history_flow_compute_size,
        )
        if (
            self.history_flow_cache_enabled
            and self.history_flow_cache_write
            and cache_path is not None
        ):
            save_flow_cache(cache_path, flow_rgb)
        return flow_rgb

    def _build_history_flow_images(
        self,
        current_frame: Optional[np.ndarray],
        step: int,
    ) -> list[np.ndarray]:
        if current_frame is None:
            return []

        current_hist = cv.resize(current_frame, self.history_flow_compute_size, interpolation=cv.INTER_AREA)
        history_frames = list(self.history_frame_buffer)
        total_frames = len(history_frames)

        sampled_history = []
        if total_frames == 0:
            sampled_history = [current_hist for _ in range(self.history_k)]
        else:
            history_offsets = sample_history_offsets(self.history_k, self.history_stride)
            for offset in history_offsets:
                index = total_frames + offset
                index = max(0, min(index, total_frames - 1))
                sampled_history.append(history_frames[index])

        sampled_frames = sampled_history + [current_hist]
        flow_images = []
        for i in range(len(sampled_frames) - 1):
            flow_rgb = self._get_or_compute_flow_cached(
                prev_frame=sampled_frames[i],
                curr_frame=sampled_frames[i + 1],
                step=step,
                pair_index=i,
            )
            flow_rgb = cv.resize(flow_rgb, self.history_image_size, interpolation=cv.INTER_AREA)
            flow_images.append(flow_rgb)
        return flow_images

    def _push_history_frame(self, images: Sequence[np.ndarray]) -> None:
        if not self.history_enabled or len(images) == 0:
            return
        history_frame = cv.resize(images[0], self.history_flow_compute_size, interpolation=cv.INTER_AREA)
        self.history_frame_buffer.append(history_frame)
        self.num_image_history = len(self.history_frame_buffer)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                unnorm_key = next(iter(norm_stats.keys()))

        if unnorm_key not in norm_stats:
            unnorm_key = next(iter(norm_stats.keys()))

        return unnorm_key


def get_model(usr_args):
    def _read_yaml_config(config_path: Path) -> Optional[dict]:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception:
            return None
        return config if isinstance(config, dict) else None

    def _collect_ckpt_config_candidates(ckpt_path: Path) -> list[Path]:
        candidates: list[Path] = []
        ckpt_parent = ckpt_path.parent
        literal_base = Path(f"{ckpt_path}/../config")
        # Prefer config.full.yaml / .yml (full merged training config) when present.
        candidates.extend(
            [
                ckpt_parent / "config.full.yaml",
                ckpt_parent / "config.full.yml",
                literal_base,
                Path(f"{literal_base}.yaml"),
                Path(f"{literal_base}.yml"),
            ]
        )

        run_dir = ckpt_parent.parent if ckpt_parent.name == "checkpoints" else ckpt_parent
        candidates.extend(
            [
                run_dir / "config.full.yaml",
                run_dir / "config.full.yml",
                run_dir / "config",
                run_dir / "config.yaml",
                run_dir / "config.yml",
                ckpt_parent / "config",
                ckpt_parent / "config.yaml",
                ckpt_parent / "config.yml",
            ]
        )

        dedup: list[Path] = []
        seen = set()
        for path in candidates:
            path_key = str(path.expanduser().resolve(strict=False))
            if path_key in seen:
                continue
            seen.add(path_key)
            dedup.append(path.expanduser().resolve(strict=False))
        return dedup

    def _load_history_overrides_from_ckpt(policy_ckpt_path_value: Any) -> tuple[dict, Optional[Path]]:
        if not policy_ckpt_path_value:
            return {}, None

        ckpt_path = Path(str(policy_ckpt_path_value)).expanduser()
        for cfg_path in _collect_ckpt_config_candidates(ckpt_path):
            if not cfg_path.is_file():
                continue

            cfg = _read_yaml_config(cfg_path)
            if cfg is None:
                continue

            datasets_cfg = cfg.get("datasets", {})
            if not isinstance(datasets_cfg, dict):
                continue
            vla_data_cfg = datasets_cfg.get("vla_data", {})
            if not isinstance(vla_data_cfg, dict):
                continue

            history_overrides = {}
            for key in ("history_k", "history_stride", "history_mode", "history_image_size"):
                if key in vla_data_cfg:
                    history_overrides[key] = vla_data_cfg[key]

            history_flow_cfg = vla_data_cfg.get("history_flow")
            if isinstance(history_flow_cfg, dict):
                history_overrides["history_flow"] = history_flow_cfg

            if history_overrides:
                return history_overrides, cfg_path

        return {}, None

    def safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    history_k_raw = usr_args.get("history_k", usr_args.get("HISTORY_K", 0))
    history_stride_raw = usr_args.get("history_stride", usr_args.get("HISTORY_STRIDE", 1))
    history_image_size = usr_args.get("history_image_size", usr_args.get("HISTORY_IMAGE_SIZE", None))
    history_flow_cfg = usr_args.get("history_flow", {}) or {}
    if not isinstance(history_flow_cfg, dict):
        history_flow_cfg = {}

    ckpt_history_overrides, ckpt_history_source = _load_history_overrides_from_ckpt(policy_ckpt_path)
    if ckpt_history_overrides:
        history_k_raw = ckpt_history_overrides.get("history_k", history_k_raw)
        history_stride_raw = ckpt_history_overrides.get("history_stride", history_stride_raw)
        history_image_size = ckpt_history_overrides.get("history_image_size", history_image_size)

        ckpt_history_flow_cfg = ckpt_history_overrides.get("history_flow")
        if isinstance(ckpt_history_flow_cfg, dict):
            merged_flow_cfg = dict(history_flow_cfg)
            merged_flow_cfg.update(ckpt_history_flow_cfg)
            if isinstance(history_flow_cfg.get("cache"), dict) and isinstance(ckpt_history_flow_cfg.get("cache"), dict):
                merged_cache_cfg = dict(history_flow_cfg["cache"])
                merged_cache_cfg.update(ckpt_history_flow_cfg["cache"])
                merged_flow_cfg["cache"] = merged_cache_cfg
            history_flow_cfg = merged_flow_cfg

    history_k = safe_int(history_k_raw, 0)
    history_stride = safe_int(history_stride_raw, 1)
    parsed_history_image_size = ModelClient._parse_history_image_size(history_image_size)

    history_flow_compute_size = history_flow_cfg.get(
        "compute_size",
        usr_args.get("history_flow_compute_size", usr_args.get("FLOW_COMPUTE_SIZE", None)),
    )
    history_flow_compute_size = parse_hw_size(history_flow_compute_size, default_size=parsed_history_image_size)
    history_flow_cache = history_flow_cfg.get("cache", {})
    if not isinstance(history_flow_cache, dict):
        history_flow_cache = {}

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    print(
        "[HistoryConfig] source={}, history_k={}, history_stride={}, "
        "history_image_size={}, history_flow_compute_size={}, history_flow_cache={}".format(
            ckpt_history_source if ckpt_history_source is not None else "deploy_policy",
            history_k,
            history_stride,
            parsed_history_image_size,
            history_flow_compute_size,
            history_flow_cache,
        )
    )

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        history_k=history_k,
        history_stride=history_stride,
        history_image_size=parsed_history_image_size,
        history_flow_compute_size=history_flow_compute_size,
        history_flow_cache=history_flow_cache,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
