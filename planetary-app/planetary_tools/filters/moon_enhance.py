"""Moon Enhance — brighten faint moons in a wide field without touching the planet.

Emulates the behaviour of WaveSharp's BackGround Enhance (BCE): detect compact
faint sources outside the planet's disk/ring envelope and drive each one toward
a target display brightness, leaving the planet and empty sky untouched.

Pipeline (calibrated on widefield.png — Saturn ~0.1% of FOV, moons at linear
luminance ~1e-4..1e-3 vs planet peak ~0.33):

1. Planet mask: seed at ``planet_floor × L.max()`` (the max, not a percentile —
   on a wide field p99 of luminance is sky), keep the largest connected
   component, and exclude everything within ``margin`` px of it (Euclidean
   distance transform). Auto follows the connected disk/ring glow locally,
   with a small safety pad and a size-based limit (half the equivalent radius,
   min 20 px), rather than spreading the widest halo around the whole planet.
   Beyond the 20 px core, a compact round moon in Auto's outer glow fringe is
   still detected and only its own disk is opened from the exclusion.
2. Background: median filter at ×4 downsample, upsampled — removes sky
   gradients and planet glow so moons appear as compact positive residual.
3. Detection: local SNR against a local median/MAD noise map (sky noise is
   strongly non-uniform; a global MAD floods the detector with false peaks),
   non-maximum suppression, and an FWHM band that rejects hot pixels and
   extended glow. Elongated sources are rejected unless strong enough to be a
   trailed moon; peaks on the image border are ignored. Capped at
   ``max_moons`` by descending SNR.
4. Gain: per moon ``g = clip(target_L / peak_L, 1, max_gain)`` blended through
   a Gaussian sized from the moon's measured FWHM. Equal gain on all channels
   preserves hue. Planet pixels inside the exclusion zone are bit-identical.
   Optionally use the brightest detected moon's peak for every gain, retaining
   the relative peak brightness of the moons instead of equalising them.
   Pseudo-relative mode compresses peak ratios with a square root so faint
   moons receive more gain while keeping the brightness ordering.

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
# Conservative upper bound / fallback for the measured auto margin. A fixed
# half-radius margin alone hides close moons around sharply processed planets.
_AUTO_MARGIN_RADIUS_FRAC = 0.5
_AUTO_MARGIN_MIN_PX = 20.0
_AUTO_MARGIN_SKY_WIDTH_PX = 15
_AUTO_MARGIN_MIN_SAMPLES = 64
# Protect the substantial connected glow, not every trace of diffuse sky haze.
# Fainter residuals are handled by background subtraction and source-shape tests.
_AUTO_MARGIN_GLOW_FLOOR = 2e-3  # fraction of the image peak
_AUTO_MARGIN_OPEN_SIZE = 13  # remove compact sources before following the halo

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
# Moons are round; overexposed ring ansae, glow knots and sky noise blobs are
# stretched. Reject sources whose x/y half-max widths differ this much...
_MAX_ELONGATION = 2.0
# ...unless they are far stronger than any elongated noise (SNR ≤ ~11 on the
# sample images): a moon that moved during a derotated stack is trailed, e.g.
# the right moon on 2moons.png at SNR 43 with 14×6 px half-max widths.
_TRAILED_MIN_SNR = 20.0
# Smoothing and NMS reflect at the image edge, so stacking-edge artefacts form
# peaks there; a moon this close to the frame is cut off anyway.
_BORDER_PX = int(np.ceil(3.5 * _RESIDUAL_SMOOTH_SIGMA))
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
    """Size-based upper bound, also used when the glow cannot be measured."""
    raw = max(_AUTO_MARGIN_MIN_PX, _AUTO_MARGIN_RADIUS_FRAC * float(r_eq))
    # Preserve the previous conservative bound; measured margins use whole pixels.
    return float(max(_AUTO_MARGIN_MIN_PX, 10.0 * round(raw / 10.0)))


def _auto_exclusion(
    lum: np.ndarray, planet: np.ndarray, distance: np.ndarray, r_eq: float,
) -> np.ndarray:
    """Follow connected, broad glow instead of using one margin on every limb.

    A ringed planet has very different halo widths above the rings and around
    the disk. Opening removes compact moons; propagation keeps only glow
    attached to the planet. Sky statistics and a peak-relative floor prevent
    diffuse background haze from swallowing nearby sources. A tight crop
    without enough sky retains the conservative distance-based fallback.
    """
    limit = _auto_margin_px(r_eq)
    fallback = distance <= limit
    if limit <= _AUTO_MARGIN_MIN_PX:
        return fallback
    sky = lum[(distance >= limit) & (distance < limit + _AUTO_MARGIN_SKY_WIDTH_PX)]
    if sky.size < _AUTO_MARGIN_MIN_SAMPLES:
        return fallback
    sky_median, sky_high = np.percentile(sky, [50, 90])
    threshold = max(
        float(sky_high + 3.0 * (sky_high - sky_median)),
        _AUTO_MARGIN_GLOW_FLOOR * float(lum.max()),
    )
    broad = ndimage.grey_opening(lum, size=_AUTO_MARGIN_OPEN_SIZE)
    halo = ndimage.binary_propagation(
        planet, mask=planet | ((broad > threshold) & fallback),
    )
    halo = ndimage.binary_fill_holes(halo)
    pad = float(np.ceil(3.5 * _RESIDUAL_SMOOTH_SIGMA))
    local = ndimage.distance_transform_edt(~halo) <= pad
    return ((distance <= _AUTO_MARGIN_MIN_PX) | local) & fallback


def auto_planet_margin(
    data: np.ndarray,
    is_grayscale: bool = False,
    planet_floor: float = DEFAULT_PLANET_FLOOR,
) -> float:
    """Largest local Auto padding, for display or a conservative manual start."""
    lum = _luminance(data, is_grayscale)
    planet, r_eq = _planet_seed(lum, planet_floor)
    if r_eq <= 0.0:
        return _AUTO_MARGIN_MIN_PX
    distance = ndimage.distance_transform_edt(~planet)
    exclusion = _auto_exclusion(lum, planet, distance, r_eq)
    # In a tight crop the mask may cover the entire image before reaching the
    # fallback distance. Keep reporting the conservative fallback in that case.
    if exclusion.all():
        return _auto_margin_px(r_eq)
    return float(np.ceil(distance[exclusion].max()))


def _planet_exclusion(
    lum: np.ndarray, planet_floor: float, margin: float
) -> tuple[np.ndarray, float]:
    """Boolean exclusion zone around the planet and its equivalent radius."""
    planet, r_eq = _planet_seed(lum, planet_floor)
    if r_eq <= 0.0:
        return np.zeros(lum.shape, dtype=bool), 0.0
    distance = ndimage.distance_transform_edt(~planet)
    if margin <= 0.0:
        return _auto_exclusion(lum, planet, distance, r_eq), r_eq
    return distance <= margin, r_eq


def _exclusion_zones(
    lum: np.ndarray, planet_floor: float, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exclusion zone, and the Auto fringe of it where compact moons may still be found.

    Auto's glow-following mask ends at a brightness threshold, so a moon embedded
    in the faint outer glow can sit right on its edge (Enceladus below Saturn's
    rings on EnceladusTest.png). Beyond the fixed minimum core, such a moon is
    kept if it is compact against the local glow. Manual margins are exact.
    """
    planet, r_eq = _planet_seed(lum, planet_floor)
    none = np.zeros(lum.shape, dtype=bool)
    if r_eq <= 0.0:
        return none, none
    distance = ndimage.distance_transform_edt(~planet)
    if margin > 0.0:
        return distance <= margin, none
    exclusion = _auto_exclusion(lum, planet, distance, r_eq)
    return exclusion, exclusion & (distance > _AUTO_MARGIN_MIN_PX)


def _measure_compact_fwhm(lum: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Half-max widths of the compact part of a source, above the local glow.

    A white top-hat removes structure broader than the Auto glow opening, so a
    glow gradient does not inflate a fringe moon's width as it does the residual.
    """
    h, w = lum.shape
    reach = _FWHM_MEASURE_HALF + _AUTO_MARGIN_OPEN_SIZE
    y0, x0 = max(0, y - reach), max(0, x - reach)
    window = lum[y0 : min(h, y + reach + 1), x0 : min(w, x + reach + 1)]
    compact = window - ndimage.grey_opening(window, size=_AUTO_MARGIN_OPEN_SIZE)
    compact = ndimage.gaussian_filter(compact, _RESIDUAL_SMOOTH_SIGMA)
    return _measure_fwhm(compact, y - y0, x - x0)


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
    exclusion, fringe = _exclusion_zones(lum, planet_floor, planet_margin)
    if bool((exclusion & ~fringe).all()):
        return [], exclusion
    # Distance outside the exclusion: glow knots peak on this rim.
    outside = ndimage.distance_transform_edt(~exclusion)

    small_w = max(1, w // _DOWNSAMPLE)
    small_h = max(1, h // _DOWNSAMPLE)
    small = _resample(lum, small_w, small_h)
    bg = _resample(ndimage.median_filter(small, size=_BG_KERNEL_SMALL), w, h)

    residual = np.clip(lum - bg, 0.0, None)
    # Measure the original residual: cutting a halo at the exclusion rim
    # before smoothing creates artificial compact peaks and distorts FWHM.
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

    peaks = (snr == ndimage.maximum_filter(snr, size=_NMS_WINDOW)) & (
        (snr > sensitivity) & (~exclusion | fringe)
    )
    peaks[:_BORDER_PX] = peaks[-_BORDER_PX:] = False
    peaks[:, :_BORDER_PX] = peaks[:, -_BORDER_PX:] = False
    ys, xs = np.nonzero(peaks)
    detections: list[MoonDetection] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if fringe[y, x]:
            # Inside the glow only compact, round sources qualify: no trailed
            # exemption, and nothing as broad as the glow-hug rejection width.
            fx, fy = _measure_compact_fwhm(lum, y, x)
            fwhm = 0.5 * (fx + fy)
            if not (_FWHM_MIN <= fwhm < _GLOW_HUG_MIN_FWHM):
                continue
            if max(fx, fy) > _MAX_ELONGATION * max(min(fx, fy), 1.0):
                continue
        else:
            fx, fy = _measure_fwhm(smooth, y, x)
            fwhm = 0.5 * (fx + fy)
            if not (_FWHM_MIN <= fwhm <= _FWHM_MAX):
                continue
            if (max(fx, fy) > _MAX_ELONGATION * max(min(fx, fy), 1.0)
                    and float(snr[y, x]) < _TRAILED_MIN_SNR):
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
    detections = detections[: max(0, int(max_moons))]
    # Open the exclusion only over each kept fringe moon's own disk, so the
    # surrounding glow stays bit-identical.
    carved = [d for d in detections if fringe[d.y, d.x]]
    if carved:
        exclusion = exclusion.copy()
        for d in carved:
            r = int(np.ceil(d.fwhm))
            y0, y1 = max(0, d.y - r), min(h, d.y + r + 1)
            x0, x1 = max(0, d.x - r), min(w, d.x + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            exclusion[y0:y1, x0:x1] &= (yy - d.y) ** 2 + (xx - d.x) ** 2 > d.fwhm ** 2
    return detections, exclusion


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
    retain_relative_brightness: bool = False,
    pseudo_relative_brightness: bool = False,
) -> np.ndarray:
    h, w = shape
    gain = np.ones((h, w), dtype=np.float32)
    brightest_peak = max((det.peak_l for det in detections), default=0.0)
    for det in detections:
        peak = det.peak_l
        if retain_relative_brightness:
            # target * sqrt(moon / brightest), expressed as a gain denominator.
            # The core guard also ignores stale preset flags when relative is off.
            peak = (np.sqrt(max(det.peak_l, 0.0) * brightest_peak)
                    if pseudo_relative_brightness else brightest_peak)
        g = target_l / max(peak, 1e-12)
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
    retain_relative_brightness: bool = False,
    pseudo_relative_brightness: bool = False,
) -> np.ndarray:
    """Brighten detected moons toward ``brightness_pct`` of full scale.

    With ``retain_relative_brightness``, set a common gain from the brightest
    detected moon, preserving relative peak luminance. Otherwise each moon has
    its own gain. ``pseudo_relative_brightness`` compresses those ratios with
    a square root, only when relative mode is enabled. Gains never dim a moon
    and are limited by ``max_gain``.
    """
    arr = np.asarray(data, dtype=np.float32)
    lum = _luminance(arr, is_grayscale)
    detections, exclusion = _cached_detections(
        lum, sensitivity, planet_margin, max_moons, planet_floor
    )
    target_l = float(brightness_pct) / 100.0
    if not detections or target_l <= 0.0:
        return arr.copy()
    gain = _gain_field(
        lum.shape, detections, target_l, radius_scale, max_gain,
        retain_relative_brightness, pseudo_relative_brightness,
    )
    # Hard guarantee: gain tails from moons near the margin must not lift the
    # planet or its glow.
    gain[exclusion] = 1.0
    # The target is luminance, so a coloured moon's strongest channel could
    # pass 100%. Cap the common gain there instead of clipping, keeping hue;
    # pixels already at or above 100% are left as they are.
    channel_max = arr if arr.ndim == 2 else arr.max(axis=-1)
    with np.errstate(divide="ignore"):
        np.minimum(gain, np.maximum(1.0 / channel_max, 1.0), out=gain)
    if arr.ndim == 2:
        return arr * gain
    return arr * gain[..., None]
