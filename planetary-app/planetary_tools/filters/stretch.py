"""Stretch Contrast OKLab — equivalent of the GIMP plug-in."""

from __future__ import annotations

import numpy as np

from planetary_tools.core.colour import clamp01, rgb_to_oklab_L, scale_rgb_by_oklab_L


def stretch_contrast_oklab(data: np.ndarray, amount_pct: float = 100.0) -> np.ndarray:
    """Stretch OKLab L via proportional RGB scaling to a target peak level.

    amount_pct is the peak level to stretch (or contract) to: 100 = full
    range, lower values cap the result's peak, 0 = black.
    """
    target = min(max(float(amount_pct) / 100.0, 0.0), 1.0)
    if target <= 0.0:
        return np.zeros_like(data)
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

    scale = target / L_range
    L_new = (L - L_min) * scale
    out = scale_rgb_by_oklab_L(rgb, L, L_new, peak=target)

    if was_gray:
        return out[..., 0]
    return out