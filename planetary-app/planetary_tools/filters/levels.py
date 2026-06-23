"""Per-channel levels on OKLab luminance and linear RGB."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from planetary_tools.core.color import oklab_to_rgb, rgb_to_oklab

LEVEL_CHANNELS = ("L", "R", "G", "B")
LEVEL_KEYS = ("in_min", "in_max", "gamma", "out_min", "out_max")

# GIMP "Auto Input Levels": 2nd / 98th histogram percentiles per channel.
_AUTO_INPUT_LOW_PCT = 2.0
_AUTO_INPUT_HIGH_PCT = 98.0


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


def _is_identity_output(levels: dict[str, float]) -> bool:
    return (
        abs(float(levels["out_min"])) < 1e-6
        and abs(float(levels["out_max"]) - 1.0) < 1e-6
    )


def _is_identity_input(levels: dict[str, float]) -> bool:
    return (
        abs(float(levels["in_min"])) < 1e-6
        and abs(float(levels["in_max"]) - 1.0) < 1e-6
        and abs(float(levels["gamma"]) - 1.0) < 1e-6
    )


def is_identity_levels(levels: dict[str, float]) -> bool:
    return _is_identity_output(levels) and _is_identity_input(levels)


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


def _apply_output_curve(
    values: np.ndarray,
    levels: dict[str, float],
) -> np.ndarray:
    """Map linear values through the output range first."""
    if _is_identity_output(levels):
        return np.asarray(values, dtype=np.float32)

    out_min = float(levels["out_min"])
    out_max = float(levels["out_max"])
    out_span = out_max - out_min
    v = np.asarray(values, dtype=np.float64)
    return (out_min + np.clip(v, 0.0, 1.0) * out_span).astype(np.float32)


def _apply_input_curve(
    values: np.ndarray,
    levels: dict[str, float],
) -> np.ndarray:
    """Remap values through input black/white points and gamma."""
    if _is_identity_input(levels):
        return np.asarray(values, dtype=np.float32)

    in_min = float(levels["in_min"])
    in_max = float(levels["in_max"])
    gamma = max(float(levels["gamma"]), 1e-6)
    out_min = float(levels["out_min"])

    span = in_max - in_min
    if abs(span) < 1e-10:
        return np.full_like(values, out_min, dtype=np.float32)

    norm = (np.asarray(values, dtype=np.float64) - in_min) / span
    norm = np.clip(norm, 0.0, 1.0)
    if abs(gamma - 1.0) >= 1e-6:
        norm = np.power(norm, 1.0 / gamma)
    return norm.astype(np.float32)


def apply_levels_curve(
    values: np.ndarray,
    levels: dict[str, float],
) -> np.ndarray:
    """Apply output mapping, then input mapping and gamma."""
    if is_identity_levels(levels):
        return np.asarray(values, dtype=np.float32)
    out = _apply_output_curve(values, levels)
    return _apply_input_curve(out, levels)


def _apply_channel_output(
    rgb: np.ndarray,
    channels: dict[str, dict[str, float]],
) -> np.ndarray:
    """Output pass for OKLab L and linear RGB."""
    if not _is_identity_output(channels["L"]):
        lab = rgb_to_oklab(rgb)
        lab[..., 0] = _apply_output_curve(lab[..., 0], channels["L"])
        rgb = oklab_to_rgb(lab, clamp=False)

    for idx, ch in enumerate(("R", "G", "B")):
        if not _is_identity_output(channels[ch]):
            rgb[..., idx] = _apply_output_curve(rgb[..., idx], channels[ch])

    return rgb


def _apply_channel_input(
    rgb: np.ndarray,
    channels: dict[str, dict[str, float]],
) -> np.ndarray:
    """Input pass for OKLab L and linear RGB."""
    if not _is_identity_input(channels["L"]):
        lab = rgb_to_oklab(rgb)
        lab[..., 0] = _apply_input_curve(lab[..., 0], channels["L"])
        rgb = oklab_to_rgb(lab, clamp=False)

    for idx, ch in enumerate(("R", "G", "B")):
        if not _is_identity_input(channels[ch]):
            rgb[..., idx] = _apply_input_curve(rgb[..., idx], channels[ch])

    return rgb


def apply_levels(data: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply stored per-channel levels to OKLab L and linear R, G, B."""
    channels = normalize_levels_params(params)
    if all(is_identity_levels(channels[ch]) for ch in LEVEL_CHANNELS):
        return np.asarray(data, dtype=np.float32)

    # Always work on a copy — preview passes the same buffer each update.
    rgb = np.array(data, dtype=np.float32, copy=True)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)

    rgb = _apply_channel_output(rgb, channels)
    rgb = _apply_channel_input(rgb, channels)
    return rgb.astype(np.float32)


def levels_params_for_preset(params: dict[str, Any]) -> dict[str, Any]:
    """Return serialisable channel levels for preset storage."""
    return {"channels": deepcopy(normalize_levels_params(params))}


def _channel_values(data: np.ndarray, channel: str) -> np.ndarray:
    """Flatten per-channel sample values used for auto-balance histograms."""
    rgb = np.asarray(data, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if channel == "L":
        return rgb_to_oklab(rgb)[..., 0].ravel()
    idx = {"R": 0, "G": 1, "B": 2}[channel]
    return rgb[..., idx].ravel()


def auto_input_levels_for_channel(values: np.ndarray) -> dict[str, float]:
    """GIMP-style auto input levels for one channel (output 0–1, gamma 1)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return identity_levels()

    lo = float(np.percentile(flat, _AUTO_INPUT_LOW_PCT))
    hi = float(np.percentile(flat, _AUTO_INPUT_HIGH_PCT))
    if hi - lo < 1e-10:
        return identity_levels()

    return {
        "in_min": lo,
        "in_max": hi,
        "gamma": 1.0,
        "out_min": 0.0,
        "out_max": 1.0,
    }


_RGB_LEVEL_CHANNELS = ("R", "G", "B")


def _lowest_rgb_input_max(channels: dict[str, dict[str, float]]) -> float:
    """Lowest auto-balance input maximum across RGB channels."""
    return float(
        min(float(channels[ch]["in_max"]) for ch in _RGB_LEVEL_CHANNELS)
    )


def _set_rgb_output_max(
    channels: dict[str, dict[str, float]],
    out_max: float,
) -> None:
    """Set the same output maximum on all auto-balanced RGB channels."""
    for ch in _RGB_LEVEL_CHANNELS:
        if is_identity_levels(channels[ch]):
            continue
        channels[ch]["out_max"] = out_max


def _clamp_auto_balance_to_peak(
    channels: dict[str, dict[str, float]],
    data: np.ndarray,
    *,
    is_grayscale: bool,
) -> None:
    """Set RGB output maximum to the lowest auto-balance RGB input maximum."""
    del data, is_grayscale
    _set_rgb_output_max(channels, _lowest_rgb_input_max(channels))


def auto_balance_levels(
    data: np.ndarray,
    *,
    is_grayscale: bool,
) -> dict[str, dict[str, float]]:
    """GIMP auto input levels on RGB; output max set to lowest RGB input maximum."""
    channels = {ch: identity_levels() for ch in LEVEL_CHANNELS}
    for ch in _RGB_LEVEL_CHANNELS:
        channels[ch] = auto_input_levels_for_channel(_channel_values(data, ch))
    if all(is_identity_levels(channels[ch]) for ch in _RGB_LEVEL_CHANNELS):
        return channels
    _clamp_auto_balance_to_peak(channels, data, is_grayscale=is_grayscale)
    return channels