"""Fine-scale grain / noise estimation for enhance-filter readouts.

Grain is estimated from the fine high-pass residual in low-structure regions
of the *subject* (not the black sky):

* **Bulk noise** — MAD of residual (robust to outliers)
* **Sparse speckles** — excess of the 99th-percentile |residual| over what a
  Gaussian with that MAD would produce (heavy tails after sharpening)
* **Mid-scale speckles** — MAD of a band-pass residual (≈2–5 px), scaled so it
  can raise the score when medium wavelet boost creates coarse salt

Before measuring, the image is cropped to the subject bounding box (with a
small pad) so wide fields score like tight crops. Large subjects are
Lanczos-downsampled for speed (not pixel-strided).
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
# Band-pass for few-pixel / coarse speckles: blur(lo) − blur(hi).
_BP_SIGMA_LO = 1.0
_BP_SIGMA_HI = 3.0
_MIN_SAMPLES = 64
_MAX_SIDE = 512
# Padding around the subject bbox as a fraction of bbox size (each side).
_BBOX_PAD_FRAC = 0.05
_BBOX_PAD_MIN = 2

# Ignore background below this fraction of the image's bright peak.
_SIGNAL_PEAK_FRACTION = 0.05
_SIGNAL_ABS_FLOOR = 1e-4

# For unit MAD of a Gaussian, |x| p99 ≈ 2.576/0.6745 ≈ 3.82 · MAD.
_GAUSS_P99_OVER_MAD = 3.8
# How strongly sparse fine-scale tails raise the score beyond MAD.
_EXCESS_TAIL_WEIGHT = 0.25
# Weight for mid-scale (band-pass) MAD relative to fine MAD.
_BANDPASS_MAD_WEIGHT = 0.35

# Multiplies the raw level into a more readable absolute score.
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


def _highpass_residual(lum: np.ndarray, sigma: float) -> np.ndarray:
    return lum - gaussian_filter(lum, sigma, mode="reflect")


def _bandpass_residual(lum: np.ndarray, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    """Mid-scale residual (few-pixel speckles), blur(lo) − blur(hi)."""
    lo = gaussian_filter(lum, sigma_lo, mode="reflect")
    hi = gaussian_filter(lum, sigma_hi, mode="reflect")
    return lo - hi


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _p99_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(np.abs(values), 99.0))


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
        return np.ones(lum.shape, dtype=bool)

    local = _local_std(lum)
    thr = float(np.quantile(local[signal], _FLAT_QUANTILE))
    mask = signal & (local <= thr)
    if int(mask.sum()) < _MIN_SAMPLES:
        return signal
    return mask


def _hybrid_grain_level(lum: np.ndarray, peak: float) -> float | None:
    """Peak-normalized hybrid grain from fine MAD, fine tails, and band-pass MAD."""
    mask = _grain_sample_mask(lum)
    fine = _highpass_residual(lum, _HP_SIGMA)[mask]
    if fine.size < _MIN_SAMPLES:
        return None

    mad_fine = _mad(fine) / peak
    p99_fine = _p99_abs(fine) / peak
    # Sparse speckles: p99 grows much faster than MAD under sharpening.
    excess_tail = max(0.0, p99_fine - _GAUSS_P99_OVER_MAD * mad_fine)
    fine_score = mad_fine + _EXCESS_TAIL_WEIGHT * excess_tail

    band = _bandpass_residual(lum, _BP_SIGMA_LO, _BP_SIGMA_HI)[mask]
    mad_band = _mad(band) / peak
    band_score = _BANDPASS_MAD_WEIGHT * mad_band

    return max(fine_score, band_score)


def flat_region_grain_level(
    data: np.ndarray,
    is_grayscale: bool,
) -> float | None:
    """Peak-normalized hybrid grain level in subject flat regions.

    Combines robust fine residual MAD with heavy-tail (p99) excess and a
    mid-scale band-pass MAD term so coarse speckles after sharpening are not
    under-reported. Stretch-invariant via / peak. ``None`` if unusable.
    """
    lum = _prepare_luminance(data, is_grayscale)
    if lum.size == 0:
        return None
    peak = float(np.max(lum))
    if peak < 1e-12:
        return None
    return _hybrid_grain_level(lum, peak)


def absolute_grain(
    data: np.ndarray,
    is_grayscale: bool,
) -> float | None:
    """Grain score for UI display (hybrid level × GRAIN_DISPLAY_SCALE)."""
    level = flat_region_grain_level(data, is_grayscale)
    if level is None:
        return None
    return level * GRAIN_DISPLAY_SCALE
