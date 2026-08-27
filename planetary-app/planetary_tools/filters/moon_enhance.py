"""Moon Enhance — brighten faint moons in a wide field without touching the planet.

Emulates the behaviour of WaveSharp's BackGround Enhance (BCE): detect compact
faint sources outside the planet's disk/ring envelope and drive each one toward
a target display brightness, leaving the planet and empty sky untouched.

Pipeline (calibrated on widefield.png — Saturn ~0.1% of FOV, moons at linear
luminance ~1e-4..1e-3 vs planet peak ~0.33):

1. Planet mask: seed at ``planet_floor × L.max()`` (the max, not a percentile —
   on a wide field p99 of luminance is sky), keep the largest connected
   component, and exclude everything within ``margin`` px of it (Euclidean
   distance transform). Auto margin is half the planet's equivalent radius
   (min 20 px). A full radius swallowed inner moons on close-ups
   (3moons.png: ~105 px vs moons at ~60 px from the rings).
2. Background: median filter at ×4 downsample, upsampled — removes sky
   gradients and planet glow so moons appear as compact positive residual.
3. Detection: local SNR against a local median/MAD noise map (sky noise is
   strongly non-uniform; a global MAD floods the detector with false peaks),
   non-maximum suppression, and an FWHM band that rejects hot pixels and
   extended glow. Capped at ``max_moons`` by descending SNR.
4. Gain: per moon ``g = clip(target_L / peak_L, 1, max_gain)`` blended through
   a Gaussian sized from the moon's measured FWHM. Equal gain on all channels
   preserves hue. Planet pixels inside the exclusion zone are bit-identical.

Detection (steps 1-3, ~2.5 s at 1600×3088) is cached on an image fingerprint +
detection params so brightness-only preview updates are instant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from planetary_tools.core.colour import linear_luminance
from planetary_tools.core.scale import scale_image

# Planet seed threshold as a fraction of the luminance maximum.
DEFAULT_PLANET_FLOOR = 0.05
# Auto exclusion margin = this fraction of the planet's equivalent radius.
# Half a radius covers limb/ring glow without swallowing inner moons just
# outside the seed (3moons.png; a full radius chose ~100 px when 50 px
# still captured them). Leftover glow knots on the exclusion rim are
# rejected in detection, not by inflating the pad (widefield.png).
_AUTO_MARGIN_RADIUS_FRAC = 0.5
_AUTO_MARGIN_MIN_PX = 20.0

# Background / noise maps are computed at this downsample factor. Full-res
# median filtering at these kernel sizes takes minutes; downsampled ~seconds.
_DOWNSAMPLE = 4
_BG_KERNEL_SMALL = 17
_NOISE_KERNEL_SMALL = 17

_RESIDUAL_SMOOTH_SIGMA = 1.5
_NMS_WINDOW = 15
# Local noise floor as a fraction of the luminance peak. Stacks with a
# clipped-black sky have local MAD ≈ 0, and dividing by it turns interpolation
# ripple into astronomical SNRs that shadow real moons in the NMS (4moons.png).
_SIGMA_FLOOR_FRAC = 1e-6
# Accepted moon FWHM band in full-res pixels (below: hot pixels; above: glow).
_FWHM_MIN = 2.0
_FWHM_MAX = 30.0
_FWHM_MEASURE_HALF = 16
# Moons are round; overexposed ring ansae and glow knots are stretched along
# the ring plane. Reject sources whose x/y half-max widths differ this much.
_MAX_ELONGATION = 2.0
# Extended residual on the exclusion rim is leftover planet glow, not a moon.
# Compact moons (FWHM ~4–7 px on 3moons.png / 4moons.png) still pass.
_GLOW_HUG_PX = 8.0
_GLOW_HUG_MIN_FWHM = 10.0
_FWHM_TO_SIGMA = 1.0 / 2.355
_MIN_MASK_SIGMA = 1.0

DEFAULT_MAX_GAIN = 300.0

_MIN_PLANET_AREA = 16


@dataclass(frozen=True)
class MoonDetection:
    y: int
    x: int
    snr: float
    peak_l: float
    fwhm: float


def _luminance(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if is_grayscale:
        return arr[..., 0]
    return linear_luminance(arr)


def _resample(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(scale_image(np.asarray(arr, dtype=np.float32), width, height))


def _planet_seed(
    lum: np.ndarray, planet_floor: float
) -> tuple[np.ndarray, float]:
    """Largest bright connected component and its equivalent-area radius."""
    empty = np.zeros(lum.shape, dtype=bool)
    peak = float(lum.max())
    if peak <= 0.0:
        return empty, 0.0
    seed = lum >= planet_floor * peak
    labels, count = ndimage.label(seed)
    if count == 0:
        return empty, 0.0
    sizes = ndimage.sum(seed, labels, range(1, count + 1))
    planet = labels == (int(np.argmax(sizes)) + 1)
    area = float(planet.sum())
    if area < _MIN_PLANET_AREA:
        return empty, 0.0
    return planet, float(np.sqrt(area / np.pi))


def _auto_margin_px(r_eq: float) -> float:
    raw = max(_AUTO_MARGIN_MIN_PX, _AUTO_MARGIN_RADIUS_FRAC * float(r_eq))
    # Planet-margin control is integers with step 10.
    return float(max(_AUTO_MARGIN_MIN_PX, 10.0 * round(raw / 10.0)))


def auto_planet_margin(
    data: np.ndarray,
    is_grayscale: bool = False,
    planet_floor: float = DEFAULT_PLANET_FLOOR,
) -> float:
    """Exclusion distance used when ``planet_margin`` is 0 (Auto)."""
    lum = _luminance(data, is_grayscale)
    _planet, r_eq = _planet_seed(lum, planet_floor)
    if r_eq <= 0.0:
        return _AUTO_MARGIN_MIN_PX
    return _auto_margin_px(r_eq)


def _planet_exclusion(
    lum: np.ndarray, planet_floor: float, margin: float
) -> tuple[np.ndarray, float]:
    """Boolean exclusion zone around the planet and its equivalent radius."""
    planet, r_eq = _planet_seed(lum, planet_floor)
    if r_eq <= 0.0:
        return np.zeros(lum.shape, dtype=bool), 0.0
    if margin <= 0.0:
        margin = _auto_margin_px(r_eq)
    distance = ndimage.distance_transform_edt(~planet)
    return distance <= margin, r_eq


def _measure_fwhm(residual: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Half-max widths through the peak along x and y, in pixels."""
    h, w = residual.shape
    half = _FWHM_MEASURE_HALF
    peak = residual[y, x]
    if peak <= 0.0:
        return 0.0, 0.0
    row = residual[y, max(0, x - half) : min(w, x + half + 1)]
    col = residual[max(0, y - half) : min(h, y + half + 1), x]
    fx = float((row > 0.5 * peak).sum())
    fy = float((col > 0.5 * peak).sum())
    return fx, fy


def detect_moons(
    lum: np.ndarray,
    *,
    sensitivity: float,
    planet_margin: float,
    max_moons: int,
    planet_floor: float = DEFAULT_PLANET_FLOOR,
) -> tuple[list[MoonDetection], np.ndarray]:
    """Detect compact faint sources outside the planet exclusion zone.

    Returns the detections and the boolean exclusion zone, inside which the
    filter must leave pixels bit-identical.
    """
    h, w = lum.shape
    exclusion, _r_eq = _planet_exclusion(lum, planet_floor, planet_margin)
    if bool(exclusion.all()):
        return [], exclusion
    # Distance outside the exclusion: glow knots peak on this rim.
    outside = ndimage.distance_transform_edt(~exclusion)

    small_w = max(1, w // _DOWNSAMPLE)
    small_h = max(1, h // _DOWNSAMPLE)
    small = _resample(lum, small_w, small_h)
    bg = _resample(ndimage.median_filter(small, size=_BG_KERNEL_SMALL), w, h)

    residual = np.clip(lum - bg, 0.0, None)
    residual[exclusion] = 0.0
    smooth = ndimage.gaussian_filter(residual, _RESIDUAL_SMOOTH_SIGMA)

    smooth_small = _resample(smooth, small_w, small_h)
    med_small = ndimage.median_filter(smooth_small, size=_NOISE_KERNEL_SMALL)
    mad_small = ndimage.median_filter(
        np.abs(smooth_small - med_small), size=_NOISE_KERNEL_SMALL
    )
    med = _resample(med_small, w, h)
    sigma = 1.4826 * _resample(mad_small, w, h)
    sigma_floor = max(_SIGMA_FLOOR_FRAC * float(lum.max()), 1e-12)
    snr = (smooth - med) / np.maximum(sigma, sigma_floor)
    snr[exclusion] = 0.0

    peaks = (snr == ndimage.maximum_filter(snr, size=_NMS_WINDOW)) & (
        snr > sensitivity
    )
    ys, xs = np.nonzero(peaks)
    detections: list[MoonDetection] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        fx, fy = _measure_fwhm(smooth, y, x)
        fwhm = 0.5 * (fx + fy)
        if not (_FWHM_MIN <= fwhm <= _FWHM_MAX):
            continue
        if max(fx, fy) > _MAX_ELONGATION * max(min(fx, fy), 1.0):
            continue
        if fwhm >= _GLOW_HUG_MIN_FWHM and float(outside[y, x]) <= _GLOW_HUG_PX:
            continue
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - 2), min(w, x + 3)
        peak_l = float(lum[y0:y1, x0:x1].max())
        detections.append(
            MoonDetection(y=y, x=x, snr=float(snr[y, x]), peak_l=peak_l, fwhm=fwhm)
        )
    detections.sort(key=lambda d: d.snr, reverse=True)
    return detections[: max(0, int(max_moons))], exclusion


# Detection dominates runtime (~2.5 s at 1600×3088); brightness-only changes
# should not re-run it. Keyed on a strided-sample fingerprint of the input plus
# the detection params. Preview and stats threads may race — worst case is one
# duplicate computation.
_DETECT_CACHE: dict[tuple, tuple[list[MoonDetection], np.ndarray]] = {}
_DETECT_CACHE_MAX = 4


def _fingerprint(arr: np.ndarray) -> tuple:
    sample = np.ascontiguousarray(arr[::61, ::61])
    return (arr.shape, hash(sample.tobytes()))


def _cached_detections(
    lum: np.ndarray,
    sensitivity: float,
    planet_margin: float,
    max_moons: int,
    planet_floor: float,
) -> tuple[list[MoonDetection], np.ndarray]:
    key = (
        _fingerprint(lum),
        round(float(sensitivity), 4),
        round(float(planet_margin), 4),
        int(max_moons),
        round(float(planet_floor), 6),
    )
    hit = _DETECT_CACHE.get(key)
    if hit is not None:
        return hit
    result = detect_moons(
        lum,
        sensitivity=sensitivity,
        planet_margin=planet_margin,
        max_moons=max_moons,
        planet_floor=planet_floor,
    )
    while len(_DETECT_CACHE) >= _DETECT_CACHE_MAX:
        _DETECT_CACHE.pop(next(iter(_DETECT_CACHE)))
    _DETECT_CACHE[key] = result
    return result


def _gain_field(
    shape: tuple[int, int],
    detections: list[MoonDetection],
    target_l: float,
    radius_scale: float,
    max_gain: float,
) -> np.ndarray:
    h, w = shape
    gain = np.ones((h, w), dtype=np.float32)
    for det in detections:
        g = target_l / max(det.peak_l, 1e-12)
        g = float(np.clip(g, 1.0, max_gain))
        if g <= 1.0:
            continue
        sigma = max(_MIN_MASK_SIGMA, det.fwhm * _FWHM_TO_SIGMA * radius_scale)
        radius = int(np.ceil(3.5 * sigma))
        y0, y1 = max(0, det.y - radius), min(h, det.y + radius + 1)
        x0, x1 = max(0, det.x - radius), min(w, det.x + radius + 1)
        yy = np.arange(y0, y1, dtype=np.float32) - det.y
        xx = np.arange(x0, x1, dtype=np.float32) - det.x
        r2 = yy[:, None] ** 2 + xx[None, :] ** 2
        local = 1.0 + (g - 1.0) * np.exp(-r2 / (2.0 * sigma * sigma))
        np.maximum(gain[y0:y1, x0:x1], local, out=gain[y0:y1, x0:x1])
    return gain


def moon_enhance(
    data: np.ndarray,
    is_grayscale: bool,
    brightness_pct: float,
    sensitivity: float,
    radius_scale: float,
    planet_margin: float,
    max_moons: int,
    planet_floor: float = DEFAULT_PLANET_FLOOR,
    max_gain: float = DEFAULT_MAX_GAIN,
) -> np.ndarray:
    """Brighten detected moons toward ``brightness_pct`` of full scale."""
    arr = np.asarray(data, dtype=np.float32)
    lum = _luminance(arr, is_grayscale)
    detections, exclusion = _cached_detections(
        lum, sensitivity, planet_margin, max_moons, planet_floor
    )
    target_l = float(brightness_pct) / 100.0
    if not detections or target_l <= 0.0:
        return arr.copy()
    gain = _gain_field(lum.shape, detections, target_l, radius_scale, max_gain)
    # Hard guarantee: gain tails from moons near the margin must not lift the
    # planet or its glow.
    gain[exclusion] = 1.0
    if arr.ndim == 2:
        return arr * gain
    return arr * gain[..., None]
