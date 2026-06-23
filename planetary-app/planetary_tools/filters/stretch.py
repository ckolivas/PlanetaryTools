"""Stretch Contrast OKLab — equivalent of the GIMP plug-in."""

from __future__ import annotations

import numpy as np

from planetary_tools.core.color import clamp01, rgb_to_oklab_L


def stretch_contrast_oklab(data: np.ndarray) -> np.ndarray:
    """Stretch OKLab L to full range via proportional RGB scaling."""
    was_gray = data.ndim == 2
    rgb = clamp01(data)
    if was_gray:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)

    L = rgb_to_oklab_L(rgb)
    L_min = float(L.min())
    L_max = float(L.max())
    L_range = L_max - L_min
    if L_range < 1e-6:
        return data.copy()

    scale = 1.0 / L_range
    V = np.max(rgb, axis=-1)
    L_new = (L - L_min) * scale

    with np.errstate(divide="ignore", invalid="ignore"):
        f_desired = np.where(L > 1e-7, L_new / L, 0.0)
        f_max = np.where(V > 1e-7, 1.0 / V, f_desired)
    f = np.minimum(f_desired, f_max)
    peak = V * f
    out_max = float(np.max(peak))
    if out_max < 1e-7:
        return data.copy()

    renorm = 1.0 / out_max
    f = f * renorm
    out = clamp01(rgb * f[..., None])

    if was_gray:
        return out[..., 0]
    return out