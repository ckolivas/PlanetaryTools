"""Moon Enhance — brighten faint moons in a wide field without touching the planet.

Emulates the behaviour of WaveSharp's BackGround Enhance (BCE): detect compact
faint sources outside the planet's disk/ring envelope and drive each one toward
a target display brightness, leaving the planet and empty sky untouched.

Pipeline (calibrated on widefield.png — Saturn ~0.1% of FOV, moons at linear
luminance ~1e-4..1e-3 vs planet peak ~0.33):

1. Planet mask: seed at ``planet_floor × L.max()`` (the max, not a percentile —
   on a wide field p99 of luminance is sky), keep the largest connected
   component plus any large piece split from it by a dark band (the ring
   shadow), and exclude everything within ``margin`` px of it (Euclidean
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

Selected moons (positions the user clicked, snapped to the local peak by
``locate_selected_moon``) come first and are always kept, even when below
Sensitivity, rejected by shape or inside the exclusion zone; the remaining
``max_moons`` places go to the strongest detections.

Detection (steps 1-3, ~2.5 s at 1600×3088) is cached on an image fingerprint +
detection params so brightness-only preview updates are instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
# A selected point within this distance of a detection is that detection.
_SELECT_MATCH_PX = 8.0
# A click searches this far for the moon it means, in image pixels.
SELECT_SEARCH_PX = 20.0
# Opening removes structure narrower than this when locating a clicked moon,
# leaving the broad planet glow; most moons are far narrower.
_SELECT_OPEN_SIZE = 31
# A clicked peak must stand this far above the local noise of the glow-free
# map; low, as a selected moon may be too faint for the detector...
_SELECT_MIN_SNR = 5.0
# ...and make up this much of its own brightness. Moons on the sample images
# are 0.5-1.0 of it; sky noise 0.25-0.33 and glow knots near the rings ~0.06.
_SELECT_MIN_FRACTION = 0.4
# A peak clicked within this distance needs only this fraction: a faint moon
# on strong glow, chosen deliberately. It must be moon-sized, as glow knots
# near the planet are not (moons measure up to 11 px across on the samples,
# 14 trailed; such knots 15 or more).
_SELECT_SQUARE_PX = 6.0
_SELECT_SQUARE_MIN_FRACTION = 0.2
_SELECT_SQUARE_MAX_FWHM = 12.0
# A moon that moved during a derotated stack is trailed (14×6 px on 2moons.png).
_SELECT_MAX_ELONGATION = 3.0
# The window's compact-residual spread is about this fraction of the pixel
# step noise on clean sky (0.15-0.36 on the sample captures).
_SELECT_STEP_NOISE_RATIO = 0.3
# Noise peaks reach 1.3-1.9 times the window's median peak; moons 4 or more.
# Skipped when the sky is clipped flat and there are too few peaks to judge.
_SELECT_PEAK_CONTRAST = 3.0
_SELECT_MIN_SKY_PEAKS = 25
# A peak beside brighter glow must rise this fraction of its height above
# the dip between them.
_SELECT_MIN_PROMINENCE = 0.3
_FWHM_TO_SIGMA = 1.0 / 2.355
_MIN_MASK_SIGMA = 1.0

DEFAULT_MAX_GAIN = 300.0

_MIN_PLANET_AREA = 16
# A dark band can split the seed: Saturn's ring shadow cuts the globe below the
# rings off from the rings (1moon.png, 32% of the largest piece), and its limb
# was then detected as moons. Keep every piece this large relative to the
# largest; even Ganymede is ~0.15% of Jupiter's disk area.
_PLANET_PIECE_FRAC = 0.05


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
    """Bright planet components and their combined equivalent-area radius."""
    empty = np.zeros(lum.shape, dtype=bool)
    peak = float(lum.max())
    if peak <= 0.0:
        return empty, 0.0
    seed = lum >= planet_floor * peak
    labels, count = ndimage.label(seed)
    if count == 0:
        return empty, 0.0
    sizes = np.asarray(ndimage.sum(seed, labels, range(1, count + 1)))
    keep = np.flatnonzero(sizes >= _PLANET_PIECE_FRAC * sizes.max()) + 1
    planet = np.isin(labels, keep)
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


def _selection_peak_is_compact(
    compact: np.ndarray, iy: int, ix: int, x0: int, y0: int, w: int, h: int,
) -> bool:
    """Whether a peak is a hill of its own, no bigger than a moon.

    Glow just outside a ring or limb is a long band. Its FWHM, measured
    across and along the band through the peak, can look moon-sized, but
    the area above half its peak spreads far along the band, and the noise
    bumps along it barely rise above the band. A moon beside the band is cut
    off from it at the dip between them, so its own hill is measured above
    half its peak or that dip, whichever is higher.
    """
    peak = float(compact[iy, ix])

    def hill(level: float) -> np.ndarray:
        labels, _ = ndimage.label(compact > level)
        return labels == labels[iy, ix]

    low, high = 0.5 * peak, peak
    blob = hill(low)
    if compact[blob].max() > peak:
        # Joined to higher ground: find the dip that separates them.
        for _ in range(12):
            mid = 0.5 * (low + high)
            if compact[hill(mid)].max() > peak:
                low = mid
            else:
                high = mid
        if high > (1.0 - _SELECT_MIN_PROMINENCE) * peak:
            return False  # A bump on brighter glow, not a source of its own.
        blob = hill(high)
    if blob.sum() > np.pi * (_FWHM_MAX / 2.0) ** 2:
        return False
    rows, cols = np.nonzero(blob)
    wh, ww = compact.shape
    # Running off the window means it spreads beyond the search; the image
    # border itself is no evidence of that.
    return not ((rows.min() == 0 and y0 > 0) or (cols.min() == 0 and x0 > 0)
                or (rows.max() == wh - 1 and y0 + wh < h)
                or (cols.max() == ww - 1 and x0 + ww < w))


def locate_selected_moon(
    data: np.ndarray,
    is_grayscale: bool,
    x: float,
    y: float,
    search: float = SELECT_SEARCH_PX,
    planet_floor: float = DEFAULT_PLANET_FLOOR,
    sensitivity: float | None = None,
    planet_margin: float = 0.0,
    max_moons: int = 0,
    known: Iterable[tuple[float, float]] = (),
) -> tuple[int, int] | None:
    """The (x, y) peak of the moon nearest a clicked point, if there is one.

    Broad glow is removed first, so a click beside the planet does not find
    the brighter glow toward the limb. Of the moon-sized peaks within
    ``search`` that stand out from the local noise and from the glow under
    them, the strongest is taken; ring edges are elongated. ``None`` when
    there is no such peak, or it is on the planet itself.

    Given ``sensitivity``, ``planet_margin`` and ``max_moons``, the moons
    the automatic detection picks with those settings count too: a moon
    brightened automatically can always be selected, even on glow too strong
    for the checks above. ``known`` positions (moons another tool picks
    itself) count the same way. Of all these, the one nearest the click is
    taken.
    """
    lum = _luminance(data, is_grayscale)
    h, w = lum.shape
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    reach = int(np.ceil(search)) + _SELECT_OPEN_SIZE
    y0, x0 = max(0, cy - reach), max(0, cx - reach)
    window = lum[y0 : min(h, cy + reach + 1), x0 : min(w, cx + reach + 1)]
    smooth = ndimage.gaussian_filter(window, _RESIDUAL_SMOOTH_SIGMA)
    compact = smooth - ndimage.grey_opening(smooth, size=_SELECT_OPEN_SIZE)
    noise = 1.4826 * float(np.median(np.abs(compact - np.median(compact))))
    # Ring or limb glow can fill much of the window and inflate that spread.
    # Neighbouring-pixel differences barely see smooth glow, so they cap it.
    steps = np.concatenate([np.diff(window, axis=0).ravel(), np.diff(window, axis=1).ravel()])
    if steps.size:
        step_noise = 1.4826 * float(np.median(np.abs(steps))) / np.sqrt(2.0)
        noise = min(noise, _SELECT_STEP_NOISE_RATIO * step_noise)
    floor = max(_SELECT_MIN_SNR * noise, _SIGMA_FLOOR_FRAC * float(lum.max()))
    maxima = (compact == ndimage.maximum_filter(compact, size=5)) & (compact > 0)
    if np.count_nonzero(maxima) >= _SELECT_MIN_SKY_PEAKS:
        # The residual is never negative, so its spread understates how high
        # noise peaks reach; a moon stands well above the typical one.
        floor = max(floor, _SELECT_PEAK_CONTRAST * float(np.median(compact[maxima])))
    yy, xx = np.ogrid[y0 : y0 + window.shape[0], x0 : x0 + window.shape[1]]
    near = (yy - cy) ** 2 + (xx - cx) ** 2 <= search ** 2
    peaks = maxima & near & (compact > floor)
    # The planet's own pieces, as the detector finds them: a bright moon can
    # pass the seed threshold without being part of the planet.
    planet = _planet_seed(lum, planet_floor)[0][y0 : y0 + window.shape[0],
                                                x0 : x0 + window.shape[1]]
    best = close = None
    for iy, ix in zip(*np.nonzero(peaks)):
        fx, fy = _measure_fwhm(compact, int(iy), int(ix))
        fwhm = 0.5 * (fx + fy)
        if not (_FWHM_MIN <= fwhm <= _FWHM_MAX):
            continue
        if max(fx, fy) > _SELECT_MAX_ELONGATION * max(min(fx, fy), 1.0):
            continue
        if planet[iy, ix]:
            continue
        # A ripple on the glow, not a source of its own; but a moon on
        # strong glow, clicked squarely, needs only to stand clear of it.
        fraction = compact[iy, ix] / max(float(smooth[iy, ix]), 1e-30)
        if fraction < _SELECT_MIN_FRACTION and not (
                fraction >= _SELECT_SQUARE_MIN_FRACTION
                and max(fx, fy) <= _SELECT_SQUARE_MAX_FWHM
                and np.hypot(x0 + ix - x, y0 + iy - y) <= _SELECT_SQUARE_PX):
            continue
        if not _selection_peak_is_compact(compact, int(iy), int(ix), x0, y0, w, h):
            continue  # A band of ring or limb glow, not a moon.
        if fraction < _SELECT_MIN_FRACTION:
            if close is None or compact[iy, ix] > compact[close]:
                close = (iy, ix)
        elif best is None or compact[iy, ix] > compact[best]:
            best = (iy, ix)
    candidates = [(x0 + int(c[1]), y0 + int(c[0])) for c in (best, close) if c is not None]
    if sensitivity is not None and max_moons > 0:
        detections, _ = _cached_detections(
            lum, sensitivity, planet_margin, max_moons, planet_floor
        )
        candidates += [(d.x, d.y) for d in detections
                       if np.hypot(d.x - x, d.y - y) <= search]
    candidates += [(int(round(kx)), int(round(ky))) for kx, ky in known
                   if np.hypot(kx - x, ky - y) <= search]
    if not candidates:
        return None
    return min(candidates, key=lambda c: np.hypot(c[0] - x, c[1] - y))


def _selected_detection(
    lum: np.ndarray,
    smooth: np.ndarray,
    snr: np.ndarray,
    exclusion: np.ndarray,
    detections: list[MoonDetection],
    x: float,
    y: float,
    planet: np.ndarray,
) -> MoonDetection | None:
    """The detection at a selected point, or one measured there if none."""
    h, w = lum.shape
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= ix < w and 0 <= iy < h):
        return None
    near = [d for d in detections if np.hypot(d.x - x, d.y - y) <= _SELECT_MATCH_PX]
    if near:
        return min(near, key=lambda d: np.hypot(d.x - x, d.y - y))
    if planet[iy, ix]:
        return None  # On the planet, not beside it.
    if exclusion[iy, ix]:
        fx, fy = _measure_compact_fwhm(lum, iy, ix)
    else:
        fx, fy = _measure_fwhm(smooth, iy, ix)
    fwhm = float(np.clip(0.5 * (fx + fy), _FWHM_MIN, _FWHM_MAX))
    y0, y1 = max(0, iy - 2), min(h, iy + 3)
    x0, x1 = max(0, ix - 2), min(w, ix + 3)
    return MoonDetection(y=iy, x=ix, snr=float(snr[iy, ix]),
                         peak_l=float(lum[y0:y1, x0:x1].max()), fwhm=fwhm)


def detect_moons(
    lum: np.ndarray,
    *,
    sensitivity: float,
    planet_margin: float,
    max_moons: int,
    planet_floor: float = DEFAULT_PLANET_FLOOR,
    selected: tuple[tuple[float, float], ...] = (),
) -> tuple[list[MoonDetection], np.ndarray]:
    """Detect compact faint sources outside the planet exclusion zone.

    ``selected`` (x, y) points are moons the user chose: they come first and
    are kept whatever their strength or place, the strongest detections
    filling the remaining ``max_moons`` places.

    Returns the detections and the boolean exclusion zone, inside which the
    filter must leave pixels bit-identical.
    """
    h, w = lum.shape
    exclusion, fringe = _exclusion_zones(lum, planet_floor, planet_margin)
    if bool((exclusion & ~fringe).all()) and not selected:
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
    chosen: list[MoonDetection] = []
    planet = _planet_seed(lum, planet_floor)[0] if selected else None
    for x, y in selected:
        d = _selected_detection(lum, smooth, snr, exclusion, detections,
                                float(x), float(y), planet)
        if d is not None and d not in chosen:
            chosen.append(d)
    rest = sorted((d for d in detections if d not in chosen),
                  key=lambda d: d.snr, reverse=True)
    detections = chosen + rest[: max(0, int(max_moons) - len(chosen))]
    # Open the exclusion only over each kept fringe or selected moon's own
    # disk, so the surrounding glow stays bit-identical.
    carved = [d for d in detections if exclusion[d.y, d.x]]
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
    selected: tuple[tuple[float, float], ...] = (),
) -> tuple[list[MoonDetection], np.ndarray]:
    key = (
        _fingerprint(lum),
        round(float(sensitivity), 4),
        round(float(planet_margin), 4),
        int(max_moons),
        round(float(planet_floor), 6),
        selected,
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
        selected=selected,
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
    selected_moons=(),
) -> np.ndarray:
    """Brighten detected moons toward ``brightness_pct`` of full scale.

    ``selected_moons`` are (x, y) points of moons to keep first; see
    ``detect_moons``.

    With ``retain_relative_brightness``, set a common gain from the brightest
    detected moon, preserving relative peak luminance. Otherwise each moon has
    its own gain. ``pseudo_relative_brightness`` compresses those ratios with
    a square root, only when relative mode is enabled. Gains never dim a moon
    and are limited by ``max_gain``.
    """
    arr = np.asarray(data, dtype=np.float32)
    lum = _luminance(arr, is_grayscale)
    selected = tuple((float(x), float(y)) for x, y in (selected_moons or ()))
    detections, exclusion = _cached_detections(
        lum, sensitivity, planet_margin, max_moons, planet_floor, selected
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
