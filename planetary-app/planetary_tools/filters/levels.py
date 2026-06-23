"""Per-channel levels on OKLab luminance and linear RGB."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from planetary_tools.core.color import oklab_to_rgb, rgb_to_oklab

LEVEL_CHANNELS = ("L", "R", "G", "B")
LEVEL_KEYS = ("in_min", "in_max", "gamma", "out_min", "out_max")


def identity_levels() -> dict[str, float]:
    return {
        "in_min": 0.0,
        "in_max": 1.0,
        "gamma": 1.0,
        "out_min": 0.0,
        "out_max": 1.0,
    }


def default_levels_params() -> dict[str, Any]:
    return {ch: identity_levels() for ch in LEVEL_CHANNELS}


def is_identity_levels(levels: dict[str, float]) -> bool:
    ref = identity_levels()
    return all(abs(float(levels.get(k, ref[k])) - ref[k]) < 1e-6 for k in LEVEL_KEYS)


def normalize_levels_params(params: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Merge stored params with identity defaults for each channel."""
    raw = params.get("channels", params)
    out: dict[str, dict[str, float]] = {}
    for ch in LEVEL_CHANNELS:
        src = raw.get(ch, {}) if isinstance(raw, dict) else {}
        merged = identity_levels()
        if isinstance(src, dict):
            for key in LEVEL_KEYS:
                if key in src:
                    merged[key] = float(src[key])
        out[ch] = merged
    return out


def apply_levels_curve(
    values: np.ndarray,
    levels: dict[str, float],
) -> np.ndarray:
    """Map input range through gamma to output range."""
    if is_identity_levels(levels):
        return np.asarray(values, dtype=np.float32)

    in_min = float(levels["in_min"])
    in_max = float(levels["in_max"])
    gamma = max(float(levels["gamma"]), 1e-6)
    out_min = float(levels["out_min"])
    out_max = float(levels["out_max"])

    span = in_max - in_min
    if abs(span) < 1e-10:
        return np.full_like(values, out_min, dtype=np.float32)

    norm = (np.asarray(values, dtype=np.float64) - in_min) / span
    norm = np.clip(norm, 0.0, 1.0)
    if abs(gamma - 1.0) >= 1e-6:
        norm = np.power(norm, 1.0 / gamma)
    out = out_min + norm * (out_max - out_min)
    return out.astype(np.float32)


def apply_levels(data: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply stored per-channel levels to OKLab L and linear R, G, B."""
    channels = normalize_levels_params(params)
    if all(is_identity_levels(channels[ch]) for ch in LEVEL_CHANNELS):
        return np.asarray(data, dtype=np.float32)

    # Always work on a copy — preview passes the same buffer each update.
    rgb = np.array(data, dtype=np.float32, copy=True)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)

    if not is_identity_levels(channels["L"]):
        lab = rgb_to_oklab(rgb)
        lab[..., 0] = apply_levels_curve(lab[..., 0], channels["L"])
        rgb = oklab_to_rgb(lab, clamp=False)

    for idx, ch in enumerate(("R", "G", "B")):
        if not is_identity_levels(channels[ch]):
            rgb[..., idx] = apply_levels_curve(rgb[..., idx], channels[ch])

    return rgb.astype(np.float32)


def levels_params_for_preset(params: dict[str, Any]) -> dict[str, Any]:
    """Return serialisable channel levels for preset storage."""
    return {"channels": deepcopy(normalize_levels_params(params))}