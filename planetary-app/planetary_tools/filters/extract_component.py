"""Extract a single colour component as greyscale."""

from __future__ import annotations

from typing import Literal

import numpy as np

from planetary_tools.core.colour import linear_luminance, rgb_to_oklab

ComponentId = Literal[
    "luminance",
    "oklab_l",
    "average",
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "yellow",
]

COMPONENT_LABELS: dict[ComponentId, str] = {
    "luminance": "Luminance (BT.709)",
    "oklab_l": "OKLab luminance",
    "average": "Average (R+G+B)/3",
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "cyan": "Cyan",
    "magenta": "Magenta",
    "yellow": "Yellow",
}

COMPONENT_ORDER: tuple[ComponentId, ...] = (
    "luminance",
    "oklab_l",
    "average",
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "yellow",
)


def _as_rgb(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if is_grayscale or arr.ndim == 2:
        if arr.ndim == 3:
            g = arr[..., 0]
        else:
            g = arr
        return np.stack([g, g, g], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unsupported image shape for component extract: {arr.shape}")


def extract_component_plane(
    data: np.ndarray,
    is_grayscale: bool,
    component: str,
) -> np.ndarray:
    """Return a single-channel (H, W) linear greyscale plane for ``component``.

    RGB channels are taken from linear document RGB. CMY are the mean of the
    two primaries that make that secondary colour (Cyan = (G+B)/2,
    Magenta = (R+B)/2, Yellow = (R+G)/2) — approximate luminance through a
    CMY bandpass from an RGB stack.
    """
    rgb = _as_rgb(data, is_grayscale)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    key = str(component).lower()

    if key in ("luminance", "bt709", "rec709"):
        plane = linear_luminance(rgb)
    elif key in ("oklab_l", "oklab", "l"):
        plane = rgb_to_oklab(rgb)[..., 0]
    elif key in ("average", "mean", "avg"):
        plane = (r + g + b) / 3.0
    elif key in ("red", "r"):
        plane = r
    elif key in ("green", "g"):
        plane = g
    elif key in ("blue", "b"):
        plane = b
    elif key in ("cyan", "c"):
        plane = 0.5 * (g + b)
    elif key in ("magenta", "m"):
        plane = 0.5 * (r + b)
    elif key in ("yellow", "y"):
        plane = 0.5 * (r + g)
    else:
        raise ValueError(f"Unknown component: {component!r}")

    return np.asarray(plane, dtype=np.float32)


def extract_component(
    data: np.ndarray,
    is_grayscale: bool,
    component: str,
    *,
    as_rgb: bool = False,
) -> np.ndarray:
    """Extract a component as greyscale.

    By default returns a 2-D (H, W) plane. With ``as_rgb=True``, returns
    (H, W, 3) with R=G=B for preview paths that expect three channels.
    """
    plane = extract_component_plane(data, is_grayscale, component)
    if as_rgb:
        return np.stack([plane, plane, plane], axis=-1)
    return plane
