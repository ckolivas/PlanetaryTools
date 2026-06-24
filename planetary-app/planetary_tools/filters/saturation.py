"""OKLab saturation and vibrance adjustments."""

from __future__ import annotations

import numpy as np

from planetary_tools.core.colour import oklab_to_rgb, rgb_to_oklab

# Typical OKLab chroma span used to weight vibrance (1.0 = 100%).
_OKLAB_CHROMA_REF = 0.4


def _as_rgb(data: np.ndarray) -> np.ndarray:
    rgb = np.asarray(data, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    return rgb


def apply_saturation_vibrance(
    data: np.ndarray,
    saturation: float = 1.0,
    vibrance: float = 1.0,
) -> np.ndarray:
    """Adjust chroma in OKLab; 1.0 leaves each control at 100%."""
    if abs(saturation - 1.0) < 1e-6 and abs(vibrance - 1.0) < 1e-6:
        return np.asarray(data, dtype=np.float32)

    rgb = _as_rgb(data)
    lab = rgb_to_oklab(rgb)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    if abs(saturation - 1.0) >= 1e-6:
        a = a * saturation
        b = b * saturation

    if abs(vibrance - 1.0) >= 1e-6:
        chroma = np.hypot(a, b)
        weight = 1.0 - np.clip(chroma / _OKLAB_CHROMA_REF, 0.0, 1.0)
        factor = 1.0 + (vibrance - 1.0) * weight
        a = a * factor
        b = b * factor

    out_lab = np.stack([L, a, b], axis=-1)
    return oklab_to_rgb(out_lab, clamp=False)