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
    arr = np.asarray(data)
    # Image buffers already have enough precision for min/max. Convert only
    # other dtypes, retaining the historical float64 interpretation of them.
    if arr.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        arr = np.asarray(arr, dtype=np.float64)
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
    ch = _channel_array(data, is_grayscale)
    lo, hi = float(ch.min()), float(ch.max())
    return BrightnessInfo(lo * 100.0, hi * 100.0, lo < -1e-6 or hi > 1.0 + 1e-6)


def brightness_increase_pct(
    input_data: np.ndarray,
    output_data: np.ndarray,
    is_grayscale: bool,
) -> float | None:
    """Peak channel increase (%) from input to output; None when input peak is ~0."""
    in_max = float(_channel_array(input_data, is_grayscale).max()) * 100.0
    out_max = float(_channel_array(output_data, is_grayscale).max()) * 100.0
    if in_max < 1e-6:
        return None
    return (out_max / in_max - 1.0) * 100.0


def _clip_black_inplace(out: np.ndarray) -> None:
    if float(out.min()) < 0.0:
        np.maximum(out, 0.0, out=out)


def _clamp_inplace(out: np.ndarray, is_grayscale: bool, *, low: bool) -> None:
    ch = _channel_array(out, is_grayscale)
    hi = float(ch.max())
    if hi > 1.0 + 1e-6:
        if low:
            lo = float(ch.min())
            span = hi - lo
            if span > 1e-6:
                np.subtract(out, lo, out=out)
                np.divide(out, span, out=out)
        else:
            np.divide(out, hi, out=out)


def clip_black_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Floor channel values below 0% to 0%; leave maximum unchanged."""
    out = np.array(data, dtype=np.float32, copy=True)
    _clip_black_inplace(out)
    return out


def clamp_high_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Scale all levels so the brightest channel value becomes 100%."""
    out = np.array(data, dtype=np.float32, copy=True)
    _clamp_inplace(out, is_grayscale, low=False)
    return out


def clamp_range_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Scale all levels so the darkest channel becomes 0% and brightest 100%."""
    out = np.array(data, dtype=np.float32, copy=True)
    _clamp_inplace(out, is_grayscale, low=True)
    return out


def apply_channel_post_process(
    data: np.ndarray,
    is_grayscale: bool,
    *,
    clip_black: bool,
    clamp_high: bool,
    clamp_low: bool = False,
) -> np.ndarray:
    """Apply clip-black flooring and/or highlight clamping to 100%."""
    # Own one result buffer, including for a no-op; never mutate the source
    # shared by a document, preview worker or undo snapshot.
    out = np.array(data, dtype=np.float32, copy=True)
    if clip_black and would_clip_low(out, is_grayscale):
        _clip_black_inplace(out)
    if clamp_high:
        _clamp_inplace(out, is_grayscale, low=clamp_low)
    return out


def clamp_channels(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Floor below 0% and scale peak to 100%."""
    return apply_channel_post_process(
        data, is_grayscale, clip_black=True, clamp_high=True
    )


# Backwards-compatible aliases used elsewhere in the codebase.
brightness_range = channel_range
would_clip_brightness = would_clip_channels
rescale_brightness_levels = clamp_channels
