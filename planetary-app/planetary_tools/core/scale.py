"""Image scaling with Lanczos-3 resampling."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _resize_channel(channel: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one float32 channel with Pillow Lanczos-3."""
    src = np.asarray(channel, dtype=np.float32)
    if src.shape[1] == width and src.shape[0] == height:
        return src
    image = Image.fromarray(src, mode="F")
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)


def scale_image(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale linear image data to ``width``×``height`` using Lanczos-3."""
    if width < 1 or height < 1:
        raise ValueError("Width and height must be at least 1 pixel.")

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return _resize_channel(arr, width, height)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        channels = [
            _resize_channel(arr[..., c], width, height) for c in range(3)
        ]
        return np.stack(channels, axis=-1)
    raise ValueError(f"Unsupported image shape for scaling: {arr.shape}")