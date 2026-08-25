"""Image rotation with high-quality Pillow resampling."""

from __future__ import annotations

import numpy as np
from PIL import Image

# Pillow rejects LANCZOS/BOX/HAMMING for mode "F" on rotate/transform;
# BICUBIC is the highest-quality filter it allows for float32 channels.
_ROTATE_RESAMPLE = Image.Resampling.BICUBIC


def _centre_crop_or_pad(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Centre-crop to ``target_w``×``target_h``, padding with zeros if needed."""
    h, w = arr.shape[:2]
    pad_top = max(0, (target_h - h) // 2)
    pad_bottom = max(0, target_h - h - pad_top)
    pad_left = max(0, (target_w - w) // 2)
    pad_right = max(0, target_w - w - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        if arr.ndim == 2:
            arr = np.pad(
                arr,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                constant_values=0.0,
            )
        else:
            arr = np.pad(
                arr,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                constant_values=0.0,
            )
        h, w = arr.shape[:2]

    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return np.asarray(arr[y0 : y0 + target_h, x0 : x0 + target_w, ...], dtype=np.float32)


def _rotate_channel(
    channel: np.ndarray,
    angle_deg: float,
    *,
    expand: bool,
    center: tuple[float, float] | None,
) -> np.ndarray:
    """Rotate one float32 channel with Pillow (bicubic; see module note)."""
    src = np.asarray(channel, dtype=np.float32)
    image = Image.fromarray(src, mode="F")
    kwargs: dict = {
        "resample": _ROTATE_RESAMPLE,
        "expand": expand,
        "fillcolor": 0.0,
    }
    if center is not None:
        kwargs["center"] = (float(center[0]), float(center[1]))
    rotated = image.rotate(float(angle_deg), **kwargs)
    return np.asarray(rotated, dtype=np.float32)


def rotate_image(
    data: np.ndarray,
    angle_deg: float,
    *,
    expand: bool = True,
    crop_to_original: bool = False,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Rotate linear image data by ``angle_deg`` degrees (positive = CCW).

    Uses Pillow bicubic resampling per channel (Lanczos is unsupported for
    mode ``F`` on rotate). ``center`` is ``(x, y)`` in pixel coordinates
    (column, row); ``None`` uses the geometric centre.

    ``expand=True`` (default) grows the canvas so the full rotated rectangle
    fits. ``crop_to_original=True`` centre-crops (or pads) the result back to
    the input width and height.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for rotation: {arr.shape}")

    orig_h, orig_w = arr.shape[:2]
    if float(angle_deg) % 360.0 == 0.0:
        out = arr.copy()
    elif arr.ndim == 2:
        out = _rotate_channel(arr, angle_deg, expand=expand, center=center)
    else:
        channels = [
            _rotate_channel(arr[..., c], angle_deg, expand=expand, center=center)
            for c in range(3)
        ]
        h = max(ch.shape[0] for ch in channels)
        w = max(ch.shape[1] for ch in channels)
        channels = [
            ch if ch.shape == (h, w) else _centre_crop_or_pad(ch, w, h)
            for ch in channels
        ]
        out = np.stack(channels, axis=-1)

    if crop_to_original and (out.shape[0] != orig_h or out.shape[1] != orig_w):
        out = _centre_crop_or_pad(out, orig_w, orig_h)
    return out
