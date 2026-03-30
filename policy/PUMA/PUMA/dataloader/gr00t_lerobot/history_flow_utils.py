import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image


def sample_history_offsets(history_k: int, history_stride: int) -> list[int]:
    history_k = max(0, int(history_k))
    stride = max(1, int(history_stride))
    return [-(history_k - i) * stride for i in range(history_k)]


def parse_hw_size(
    size_value: Optional[Union[Sequence[int], str]],
    default_size: Tuple[int, int] = (128, 128),
) -> Tuple[int, int]:
    if size_value is None:
        return default_size
    if isinstance(size_value, str):
        try:
            size_value = json.loads(size_value)
        except Exception:
            return default_size
    if isinstance(size_value, (list, tuple)) and len(size_value) == 2:
        try:
            h = int(size_value[0])
            w = int(size_value[1])
        except (TypeError, ValueError):
            return default_size
        if h > 0 and w > 0:
            return (h, w)
    return default_size


def _ensure_rgb_uint8(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0 + 0.5).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    # OpenCV HSV hue range is [0, 179].
    hsv[..., 0] = np.mod(ang * 0.5, 180.0).astype(np.uint8)
    hsv[..., 1] = 255

    # Use robust normalization to avoid occasional large-motion domination.
    max_mag = float(np.percentile(mag, 99.0))
    if max_mag < 1e-6:
        hsv[..., 2] = 0
    else:
        hsv[..., 2] = np.clip((mag / max_mag) * 255.0, 0, 255).astype(np.uint8)

    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)


def compute_flow_rgb_farneback(
    prev_rgb: Union[np.ndarray, Image.Image],
    curr_rgb: Union[np.ndarray, Image.Image],
    compute_size: Tuple[int, int],
    farneback_cfg: Optional[dict] = None,
) -> np.ndarray:
    prev = _ensure_rgb_uint8(prev_rgb)
    curr = _ensure_rgb_uint8(curr_rgb)

    h, w = compute_size
    prev = cv.resize(prev, (w, h), interpolation=cv.INTER_AREA)
    curr = cv.resize(curr, (w, h), interpolation=cv.INTER_AREA)

    prev_gray = cv.cvtColor(prev, cv.COLOR_RGB2GRAY)
    curr_gray = cv.cvtColor(curr, cv.COLOR_RGB2GRAY)

    cfg = farneback_cfg or {}
    flow = cv.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=float(cfg.get("pyr_scale", 0.5)),
        levels=int(cfg.get("levels", 3)),
        winsize=int(cfg.get("winsize", 15)),
        iterations=int(cfg.get("iterations", 3)),
        poly_n=int(cfg.get("poly_n", 5)),
        poly_sigma=float(cfg.get("poly_sigma", 1.2)),
        flags=int(cfg.get("flags", 0)),
    )
    return _flow_to_rgb(flow)


def build_flow_cache_key(
    dataset_path: Union[str, Path],
    dataset_name: str,
    trajectory_id: Union[str, int],
    step: Union[str, int],
    prev_offset: int,
    curr_offset: int,
    video_key: str,
    compute_size: Tuple[int, int],
    version: str = "v1",
) -> str:
    payload = {
        "version": version,
        "dataset_path": str(dataset_path),
        "dataset_name": str(dataset_name),
        "trajectory_id": str(trajectory_id),
        "step": int(step),
        "prev_offset": int(prev_offset),
        "curr_offset": int(curr_offset),
        "video_key": str(video_key),
        "compute_h": int(compute_size[0]),
        "compute_w": int(compute_size[1]),
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_flow_cache_path(cache_root: Union[str, Path], cache_dirname: str, cache_key: str) -> Path:
    root = Path(cache_root)
    return root / cache_dirname / cache_key[:2] / f"{cache_key}.npz"


def load_flow_cache(cache_path: Union[str, Path]) -> Optional[np.ndarray]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "flow_rgb" not in data:
                return None
            flow_rgb = data["flow_rgb"]
    except Exception:
        return None
    if flow_rgb.dtype != np.uint8:
        flow_rgb = flow_rgb.astype(np.uint8)
    return flow_rgb


def save_flow_cache(cache_path: Union[str, Path], flow_rgb: np.ndarray) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    flow_rgb = np.asarray(flow_rgb, dtype=np.uint8)
    with open(cache_path, "wb") as f:
        np.savez(f, flow_rgb=flow_rgb)
