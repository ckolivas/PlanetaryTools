"""Noise estimation for enhance-filter readouts.

Pipeline
--------
1. Crop to subject bounding box (wide FOV ≈ tight crop of the same disk).
2. Lanczos-downsample for speed (not pixel stride — that aliases residual).
3. Estimate a **texture / blur scale** (PSF proxy) from multi-scale DoG energy
   on the subject — softer stacks peak at coarser scales.
4. Score residual energy at scales derived from that PSF proxy:
   * fine high-pass MAD (bulk noise)
   * excess p99 tail beyond a Gaussian expectation (sparse speckles)
   * mid-scale band-pass MAD (coarser salt), lightly weighted

Peak-normalized so global stretch does not inflate the score.

Calibration sources for PSF↔scale mapping: good.png, poor.png, blue.png
(unsharpened stacks). Already-sharpened samples (e.g. smooth/noisy) were not
used to set the mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from planetary_tools.core.colour import linear_luminance
from planetary_tools.core.scale import scale_image

# Fraction of lowest local-contrast *signal* pixels treated as flat.
_FLAT_QUANTILE = 0.35
_LOCAL_WIN = 7
_MIN_SAMPLES = 64
_MAX_SIDE = 512
_BBOX_PAD_FRAC = 0.05
_BBOX_PAD_MIN = 2

_SIGNAL_PEAK_FRACTION = 0.05
_SIGNAL_ABS_FLOOR = 1e-4

# Multi-scale DoG probes for texture / blur scale estimation (pixels).
_TEXTURE_SIGMAS = (0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
_TEXTURE_DOG_RATIO = 1.6
_TEXTURE_SCALE_MIN = 0.8
_TEXTURE_SCALE_MAX = 6.5

# Map texture scale s → residual filter sigmas (calibrated on good/poor/blue).
# Softer / coarser source content → larger s → residual probes coarser speckles.
_HP_FROM_S = 0.35
_HP_MIN, _HP_MAX = 0.6, 2.5
_BP_LO_FROM_S = 0.55
_BP_HI_FROM_S = 1.5
_BP_LO_MIN, _BP_HI_MAX = 0.9, 10.0
_BP_MIN_WIDTH = 1.0

# For unit MAD of a Gaussian, |x| p99 ≈ 3.8 · MAD.
_GAUSS_P99_OVER_MAD = 3.8
# Linear excess of p99 over the Gaussian expectation (sparse speckles).
_EXCESS_TAIL_WEIGHT = 0.45
# When p99/MAD exceeds this, multiply score further — soft stacks that look
# smooth (low MAD) but explode into coarse salt when sharpened (blue.png).
_TAIL_RATIO_REF = 6.0
_TAIL_RATIO_BOOST = 0.55
# Texture scale above this is treated as a soft / large-PSF source (blue≈4.3;
# good/poor≈3.9). Soft sources get mid-scale salt weighted much more heavily.
_TEXTURE_SOFT_REF = 4.15
_TEXTURE_SOFT_BOOST = 0.50  # multiplies fine-score for soft sources
_BANDPASS_MAD_WEIGHT = 0.18  # light band MAD for normal (sharp) sources
# Soft-source band-pass: catch coarse wavelet salt that fine residual misses.
_SOFT_BAND_MAD_WEIGHT = 0.20
_SOFT_BAND_MAD_FROM_S = 0.90  # extra × max(0, s − soft_ref)
_SOFT_BAND_EXCESS_WEIGHT = 0.25
_SOFT_BAND_EXCESS_FROM_S = 1.20

NOISE_DISPLAY_SCALE = 1000.0


def _luminance(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if is_grayscale or arr.ndim == 2:
        return arr if arr.ndim == 2 else arr[..., 0]
    return linear_luminance(arr).astype(np.float64)


def _crop_to_subject(lum: np.ndarray) -> np.ndarray:
    """Crop to the bright-subject bounding box so empty FOV does not bias the noise score."""
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
    """Subject crop then optional downsample for a framing-robust sample."""
    lum = _luminance(data, is_grayscale)
    lum = _crop_to_subject(lum)
    return _downsample_for_speed(lum)


def _local_std(lum: np.ndarray, size: int = _LOCAL_WIN) -> np.ndarray:
    mean = uniform_filter(lum, size=size, mode="reflect")
    mean_sq = uniform_filter(lum * lum, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


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
    peak = float(np.percentile(lum, 99.0))
    floor = max(_SIGNAL_PEAK_FRACTION * peak, _SIGNAL_ABS_FLOOR)
    return lum >= floor


def _noise_sample_mask(lum: np.ndarray) -> np.ndarray:
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


def estimate_texture_scale(
    data: np.ndarray,
    is_grayscale: bool,
) -> float:
    """Estimate characteristic subject texture / blur scale in prepared pixels.

    Energy-weighted mean of multi-scale DoG responses on the signal mask.
    Larger values indicate softer stacks / coarser residual structure (PSF proxy).
    """
    lum = _prepare_luminance(data, is_grayscale)
    return _estimate_texture_scale_from_lum(lum)


def _estimate_texture_scale_from_lum(lum: np.ndarray) -> float:
    mask = _signal_mask(lum)
    if int(mask.sum()) < _MIN_SAMPLES:
        mask = np.ones(lum.shape, dtype=bool)

    sigmas = np.asarray(_TEXTURE_SIGMAS, dtype=np.float64)
    energies = np.empty(sigmas.shape, dtype=np.float64)
    for i, s in enumerate(sigmas):
        a = gaussian_filter(lum, float(s), mode="reflect")
        b = gaussian_filter(lum, float(s * _TEXTURE_DOG_RATIO), mode="reflect")
        dog = (a - b) / max(float(s), 1e-6)
        v = dog[mask]
        med = float(np.median(v))
        energies[i] = float(np.median(np.abs(v - med)))

    total = float(energies.sum())
    if total < 1e-18:
        return 2.5  # neutral default
    weights = energies / total
    scale = float(np.sum(sigmas * weights))
    return float(np.clip(scale, _TEXTURE_SCALE_MIN, _TEXTURE_SCALE_MAX))


def _scales_from_texture(texture_scale: float) -> tuple[float, float, float]:
    """Return (hp_sigma, bandpass_lo, bandpass_hi) from texture scale s."""
    s = float(texture_scale)
    hp = float(np.clip(_HP_FROM_S * s, _HP_MIN, _HP_MAX))
    bp_lo = float(np.clip(_BP_LO_FROM_S * s, _BP_LO_MIN, _BP_HI_MAX))
    bp_hi = float(np.clip(_BP_HI_FROM_S * s, bp_lo + _BP_MIN_WIDTH, _BP_HI_MAX))
    return hp, bp_lo, bp_hi


def _hybrid_noise_level(
    lum: np.ndarray,
    peak: float,
    texture_scale: float,
) -> float | None:
    hp, bp_lo, bp_hi = _scales_from_texture(texture_scale)
    mask = _noise_sample_mask(lum)
    fine = (lum - gaussian_filter(lum, hp, mode="reflect"))[mask]
    if fine.size < _MIN_SAMPLES:
        return None

    mad_fine = _mad(fine) / peak
    p99_fine = _p99_abs(fine) / peak
    excess_tail = max(0.0, p99_fine - _GAUSS_P99_OVER_MAD * mad_fine)
    fine_score = mad_fine + _EXCESS_TAIL_WEIGHT * excess_tail

    # Heavy-tailed residual (high p99/MAD): coarse speckles after sharpening a
    # soft stack — MAD alone under-reports these (blue.png failure mode).
    tail_ratio = p99_fine / (mad_fine + 1e-18)
    fine_score *= 1.0 + _TAIL_RATIO_BOOST * max(0.0, tail_ratio - _TAIL_RATIO_REF)

    soft = max(0.0, float(texture_scale) - _TEXTURE_SOFT_REF)
    # Soft / large-PSF sources get a mild permanent boost on the fine score.
    fine_score *= 1.0 + _TEXTURE_SOFT_BOOST * soft

    band = (
        gaussian_filter(lum, bp_lo, mode="reflect")
        - gaussian_filter(lum, bp_hi, mode="reflect")
    )[mask]
    mad_band = _mad(band) / peak
    p99_band = _p99_abs(band) / peak
    band_excess = max(0.0, p99_band - _GAUSS_P99_OVER_MAD * mad_band)

    if soft > 0.0:
        # Soft stacks: mid-scale salt (esp. after medium/coarse wavelet) must
        # dominate — this is what looks "coarsely speckled" on blue.png.
        band_score = (
            (_SOFT_BAND_MAD_WEIGHT + _SOFT_BAND_MAD_FROM_S * soft) * mad_band
            + (_SOFT_BAND_EXCESS_WEIGHT + _SOFT_BAND_EXCESS_FROM_S * soft) * band_excess
        )
    else:
        # Sharp sources: light band MAD only (avoid scoring real structure).
        band_score = _BANDPASS_MAD_WEIGHT * mad_band

    return max(fine_score, band_score)


def flat_region_noise_level(
    data: np.ndarray,
    is_grayscale: bool,
    *,
    texture_scale: float | None = None,
) -> float | None:
    """Peak-normalized hybrid noise level in subject flat regions.

    If ``texture_scale`` is omitted it is estimated from ``data``. Pass a
    scale estimated from the *unsharpened source* when scoring sharpened
    trials (e.g. auto search) so residual probes stay matched to the PSF.
    """
    lum = _prepare_luminance(data, is_grayscale)
    if lum.size == 0:
        return None
    peak = float(np.max(lum))
    if peak < 1e-12:
        return None
    if texture_scale is None:
        texture_scale = _estimate_texture_scale_from_lum(lum)
    return _hybrid_noise_level(lum, peak, texture_scale)


def absolute_noise(
    data: np.ndarray,
    is_grayscale: bool,
    *,
    texture_scale: float | None = None,
) -> float | None:
    """Noise score for UI display (hybrid level × NOISE_DISPLAY_SCALE)."""
    level = flat_region_noise_level(
        data, is_grayscale, texture_scale=texture_scale
    )
    if level is None:
        return None
    return level * NOISE_DISPLAY_SCALE
