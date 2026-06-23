"""Per-channel level measurement and clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BrightnessInfo:
    """Min/max across discrete channel values, expressed as percentages."""

    min_pct: float
    max_pct: float
    would_clip: bool

    def format_line(self, prefix: str = "") -> str:
        if self.min_pct < -1e-4:
            note = "  (clipped)"
        elif self.max_pct > 100.0 + 1e-4:
            note = "  (clipping)"
        else:
            note = ""
        return f"{prefix}Min: {self.min_pct:.1f}%   Max: {self.max_pct:.1f}%{note}"


def _channel_array(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if is_grayscale:
        return arr if arr.ndim == 2 else arr[..., 0]
    return arr


def channel_range(data: np.ndarray, is_grayscale: bool) -> tuple[float, float]:
    """Return (min%, max%) over all discrete channel values."""
    ch = _channel_array(data, is_grayscale)
    lo, hi = float(ch.min()), float(ch.max())
    return lo * 100.0, hi * 100.0


def would_clip_low(data: np.ndarray, is_grayscale: bool) -> bool:
    ch = _channel_array(data, is_grayscale)
    return float(ch.min()) < -1e-6


def would_clip_high(data: np.ndarray, is_grayscale: bool) -> bool:
    ch = _channel_array(data, is_grayscale)
    return float(ch.max()) > 1.0 + 1e-6


def would_clip_channels(data: np.ndarray, is_grayscale: bool) -> bool:
    """True when any discrete channel value falls outside [0, 1]."""
    return would_clip_low(data, is_grayscale) or would_clip_high(data, is_grayscale)


def measure_brightness(data: np.ndarray, is_grayscale: bool) -> BrightnessInfo:
    lo, hi = channel_range(data, is_grayscale)
    return BrightnessInfo(lo, hi, would_clip_channels(data, is_grayscale))


def brightness_increase_pct(
    input_data: np.ndarray,
    output_data: np.ndarray,
    is_grayscale: bool,
) -> float | None:
    """Peak channel increase (%) from input to output; None when input peak is ~0."""
    _, in_max = channel_range(input_data, is_grayscale)
    _, out_max = channel_range(output_data, is_grayscale)
    if in_max < 1e-6:
        return None
    return (out_max / in_max - 1.0) * 100.0


def clip_black_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Floor channel values below 0% to 0%; leave maximum unchanged."""
    out = np.asarray(data, dtype=np.float32)
    if float(out.min()) < 0.0:
        out = np.maximum(out, 0.0)
    return out.astype(np.float32)


def clamp_high_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Scale all levels so the brightest channel value becomes 100%."""
    out = np.asarray(data, dtype=np.float32)
    peak = float(_channel_array(out, is_grayscale).max())
    if peak > 1.0 + 1e-6:
        out = out / peak
    return out.astype(np.float32)


def clamp_range_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Scale all levels so the darkest channel becomes 0% and brightest 100%."""
    out = np.asarray(data, dtype=np.float32)
    ch = _channel_array(out, is_grayscale)
    lo, hi = float(ch.min()), float(ch.max())
    span = hi - lo
    if hi > 1.0 + 1e-6 and span > 1e-6:
        out = (out - lo) / span
    return out.astype(np.float32)


def apply_channel_post_process(
    data: np.ndarray,
    is_grayscale: bool,
    *,
    clip_black: bool,
    clamp_high: bool,
    clamp_low: bool = False,
) -> np.ndarray:
    """Apply clip-black flooring and/or highlight clamping to 100%."""
    out = np.asarray(data, dtype=np.float32)
    if clip_black and would_clip_low(out, is_grayscale):
        out = clip_black_channels(out, is_grayscale)
    if clamp_high and would_clip_high(out, is_grayscale):
        if clamp_low:
            out = clamp_range_channels(out, is_grayscale)
        else:
            out = clamp_high_channels(out, is_grayscale)
    return out.astype(np.float32)


def clamp_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Floor below 0% and scale peak to 100%."""
    return apply_channel_post_process(
        data, is_grayscale, clip_black=True, clamp_high=True
    )


# Backwards-compatible aliases used elsewhere in the codebase.
brightness_range = channel_range
would_clip_brightness = would_clip_channels
rescale_brightness_levels = clamp_channels