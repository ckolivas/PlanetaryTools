"""Fine-scale grain / noise estimation for enhance-filter readouts.

Grain is estimated as the MAD of a fine high-pass residual in low-structure
regions of the *subject* (not the black sky). Near-black background is
excluded first; among remaining pixels the lowest-contrast subset is used.

Before measuring, the image is cropped to the subject bounding box (with a
small pad) so wide fields with large empty margins score like tight crops of
the same disk. Large subjects are then Lanczos-downsampled for speed — not
pixel-strided, which aliases residual energy.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from planetary_tools.core.colour import linear_luminance
from planetary_tools.core.scale import scale_image

# Fraction of lowest local-contrast *signal* pixels treated as flat.
_FLAT_QUANTILE = 0.35
_LOCAL_WIN = 7
_HP_SIGMA = 1.0
_MIN_SAMPLES = 64
_MAX_SIDE = 512
# Padding around the subject bbox as a fraction of bbox size (each side).
_BBOX_PAD_FRAC = 0.05
_BBOX_PAD_MIN = 2

# Ignore background below this fraction of the image's bright peak.
# Planetary stacks are often mostly black sky, which would otherwise dominate
# any "flat region" mask and force the grain reading to ~0.
_SIGNAL_PEAK_FRACTION = 0.05
_SIGNAL_ABS_FLOOR = 1e-4

# Multiplies the raw MAD residual into a more readable absolute score.
# Tweak after testing on real planetary stacks (higher → larger numbers).
GRAIN_DISPLAY_SCALE = 1000.0


def _luminance(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if is_grayscale or arr.ndim == 2:
        return arr if arr.ndim == 2 else arr[..., 0]
    return linear_luminance(arr).astype(np.float64)


def _crop_to_subject(lum: np.ndarray) -> np.ndarray:
    """Crop to the bright-subject bounding box so empty FOV does not bias grain."""
    if lum.size == 0:
        return lum
    peak = float(np.percentile(lum, 99.0))
    floor = max(_SIGNAL_PEAK_FRACTION * peak, _SIGNAL_ABS_FLOOR)
    ys, xs = np.where(lum >= floor)
    if ys.size < _MIN_SAMPLES:
        return lum
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    pad_y = max(_BBOX_PAD_MIN, int(round(h * _BBOX_PAD_FRAC)))
    pad_x = max(_BBOX_PAD_MIN, int(round(w * _BBOX_PAD_FRAC)))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(lum.shape[0] - 1, y1 + pad_y)
    x1 = min(lum.shape[1] - 1, x1 + pad_x)
    return lum[y0 : y1 + 1, x0 : x1 + 1]


def _downsample_for_speed(lum: np.ndarray) -> np.ndarray:
    """Lanczos-downsample so the longest side is at most ``_MAX_SIDE``."""
    h, w = lum.shape[:2]
    longest = max(h, w)
    if longest <= _MAX_SIDE:
        return np.asarray(lum, dtype=np.float64)
    scale = _MAX_SIDE / float(longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = scale_image(np.asarray(lum, dtype=np.float32), new_w, new_h)
    return np.asarray(resized, dtype=np.float64)


def _prepare_luminance(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Subject crop then optional downsample for a framing-robust grain sample."""
    lum = _luminance(data, is_grayscale)
    lum = _crop_to_subject(lum)
    return _downsample_for_speed(lum)


def _local_std(lum: np.ndarray, size: int = _LOCAL_WIN) -> np.ndarray:
    mean = uniform_filter(lum, size=size, mode="reflect")
    mean_sq = uniform_filter(lum * lum, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def _fine_residual(lum: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(lum, _HP_SIGMA, mode="reflect")
    return lum - blurred


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _signal_mask(lum: np.ndarray) -> np.ndarray:
    """Pixels bright enough to be subject rather than empty background."""
    peak = float(np.percentile(lum, 99.0))
    floor = max(_SIGNAL_PEAK_FRACTION * peak, _SIGNAL_ABS_FLOOR)
    return lum >= floor


def _grain_sample_mask(lum: np.ndarray) -> np.ndarray:
    """Low-structure pixels on the subject (excludes black sky)."""
    signal = _signal_mask(lum)
    n_signal = int(signal.sum())
    if n_signal < _MIN_SAMPLES:
        # Degenerate image: fall back to whole frame.
        return np.ones(lum.shape, dtype=bool)

    local = _local_std(lum)
    # Rank local contrast only among signal pixels.
    thr = float(np.quantile(local[signal], _FLAT_QUANTILE))
    mask = signal & (local <= thr)
    if int(mask.sum()) < _MIN_SAMPLES:
        return signal
    return mask


def flat_region_grain_level(
    data: np.ndarray,
    is_grayscale: bool,
) -> float | None:
    """Peak-normalized MAD of fine residual in subject flat regions.

    Returns MAD / peak luminance so a global contrast stretch (which scales
    residual and peak together) does not change the grain score. ``None`` if
    the sample is unusable or the peak is effectively zero.
    """
    lum = _prepare_luminance(data, is_grayscale)
    if lum.size == 0:
        return None
    peak = float(np.max(lum))
    if peak < 1e-12:
        return None
    samples = _fine_residual(lum)[_grain_sample_mask(lum)]
    if samples.size < _MIN_SAMPLES:
        return None
    return _mad(samples) / peak


def absolute_grain(
    data: np.ndarray,
    is_grayscale: bool,
) -> float | None:
    """Grain score for UI display (peak-normalized MAD × GRAIN_DISPLAY_SCALE)."""
    level = flat_region_grain_level(data, is_grayscale)
    if level is None:
        return None
    return level * GRAIN_DISPLAY_SCALE
