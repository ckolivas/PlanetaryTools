"""Correct Optical Aberrations — use a small moon as the image's own PSF.

A moon such as Tethys, Dione or Rhea is 1–2.5 px across at typical planetary
image scales, so in the stack (or after linear sharpening) it is the effective
point spread function of the whole image: astigmatism, coma or residual
tracking blur show up as an elongated or lopsided moon. Measured on
2026-11-21 IR642 Saturn data from a scope with slight astigmatism, three moons
agreed to a correlation of 0.996+ and their PSF deconvolved the planet without
overshoot, so the moon blur represents the planet blur.

Method:

1. Moons: Auto combines every suitable moon (Moon Enhance's detector, then
   bright, compact, isolated, unclipped and off the planet's glow), weighted
   by suitability; or the moon nearest a point the user picked.
2. PSF: a stamp around each moon, a fitted quadratic background removed,
   neighbour moons masked and refilled by point reflection, centred on the
   peak (a coma tail would pull a centroid towards itself), edge-tapered,
   normalised and averaged. Angular harmonics of the core measure coma
   (m = 1), astigmatism (m = 2) and trefoil (m = 3), and whether the moons
   agree on them: off-axis coma varies across the field.
3. Astigmatism (switched off, ASTIGMATISM_ENABLED; kept for revisiting):
   an elliptical Gaussian fitted to the point-symmetric core
   gives σa, σb and the angle. The analytic inverse of the extra blur along
   the long axis, ``K = exp(2π² ((σa² − σt²) fu² + (σb² − σt²)₊ fv²))``,
   with σt = Target size × σb, softly capped at the Max boost G as
   ``1/((1 − 1/G)/K + 1/G)``: 1 at zero frequency, never below 1.
4. Coma and trefoil displace light rather than spread it, so in the spectrum
   they are a phase error. The phase between the measured PSF and the PSF
   with those harmonics removed is undone where the spectrum is measured
   reliably, above the largest scales, with the kernel kept within the
   moon's blur: light moves back and nothing is amplified.
5. Protect background blends the result back to the input on sky that holds
   only noise.

History: dividing a round target by the measured spectrum fell towards zero
wherever the faint moon was noisy, smoothing the speckle around it instead of
correcting it, and errors in the moon's halo, scaled up by a planet 1000×
brighter, left dark rings around the limb (0.69% of the peak on 2026-11-21 L3
stacks). On the 2026-11-21 IR642 stack the moon went from 13.6×9.1 px to
9.4×8.1 at 4× with ringing 0.005% of the peak; on the 2026-11-03
prefoil stack trefoil fell from 0.09-0.12 to 0.015-0.03 on every moon. Correct
the stack before wavelet sharpening: sharpening the long axis of an already
sharpened image deepens its wavelet ringing.

Only valid where the image so far is linear in the moon's light: measure
before Moon Enhance, denoise or clamping.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.fft import irfft2, rfft2
from scipy import ndimage, optimize

from planetary_tools.filters.moon_enhance import (
    DEFAULT_PLANET_FLOOR,
    MoonDetection,
    _cached_detections,
    _luminance,
)

# Astigmatism correction is switched off: even limited to the frequencies
# with signal, the boost left a worse result after sharpening than no
# correction (the noise it added cost more sharpening than the elongation
# it removed gave back). The code is kept, with its tests, to revisit; with
# this off, the moons' elongation is still measured (the ellipse sizes the
# sky mask) but not compared between moons for drift, and the filter takes
# no astigmatism, Target size or Max boost parameters.
ASTIGMATISM_ENABLED = False

DEFAULT_OPTICAL_CORRECTION_PARAMS = {
    # Pick the most suitable moon in each image.
    "auto": True,
    # Manual moon position in image pixels; negative = none chosen.
    "moon_x": -1.0,
    "moon_y": -1.0,
    # Measure each colour channel's PSF on the same moon.
    "per_channel": True,
    # Leave sky that holds only noise unchanged instead of moving its speckle.
    "protect_background": True,
    # Which aberrations to correct.
    "coma": True,
    "trefoil": True,
}
if ASTIGMATISM_ENABLED:
    DEFAULT_OPTICAL_CORRECTION_PARAMS.update({
        # Target PSF size as % of the measured minor axis profile. 100 = round only.
        "target_size": 100.0,
        # Largest boost the filter may give any spatial frequency.
        "max_boost": 2.0,
        "astigmatism": True,
    })

# Candidate detection threshold (local SNR) and count.
_DETECT_SENSITIVITY = 5.0
_DETECT_MAX = 40
# Auto selection: moons need this SNR; above the cap brightness stops helping
# and the most compact moon wins (a resolved disc such as Titan is larger).
_AUTO_MIN_SNR = 15.0
_AUTO_SNR_CAP = 200.0
_COMPACT_POWER = 4.0
# Candidates need this fraction of the strongest (capped) SNR, and at least
# this fraction of the typical candidate size.
_AUTO_RELATIVE_SNR = 0.2
_MIN_SIZE_FRAC = 0.5
# Narrowest plausible moon after sharpening; single pixels are noise.
_MIN_FWHM = 1.3
# Manual picks: nearest detection within this distance (moons drift between
# frames of a batch), else the brightest pixel within the local radius.
_PICK_SEARCH_PX = 30.0
_PICK_LOCAL_PX = 8
# Stamp radius from the measured FWHM.
_RADIUS_PER_FWHM = 2.5
_MIN_RADIUS = 6
_MAX_RADIUS = 64
_ANNULUS_PX = 4
# Neighbour moons brighter than this fraction of the chosen peak are masked.
_NEIGHBOUR_FRAC = 0.02
# Saturated/clipped moons are not PSFs.
_CLIPPED_FRAC = 0.98
# The core ellipse is fitted within this many long-axis half-maximum widths.
_FIT_REACH_PER_FWHM = 1.3
# Spectral noise floor for trusting the measured phase, in noise sigmas.
_NOISE_FLOOR_SIGMAS = 3.0
# Phase is left alone below this many cycles per long-axis FWHM (or per
# flare length, when the flare reaches further), where the
# moon's halo and any glow under it are least reliable. 0.35 keeps nearly all
# the trefoil correction (2026-11-03 prefoil: three-fold 0.09-0.11 -> 0.01-0.02)
# without extra rings; 0.7 halved it. A moon on the planet's glow still rings
# (2026-11-01 IR642: 0.5% of the planet peak), so Auto avoids such moons.
_PHASE_LOW_FREQ = 0.35
# Phase kernel radius as a fraction of the stamp radius.
_PHASE_KERNEL_REACH = 0.6
# Flare: how far the one-sided (odd) part of the moon reaches, where its
# smoothed strength falls below this fraction of its peak (and 4 noise
# sigmas). A narrow collimation tail on an SCT reaches 2.5-3 FWHM, on
# 2026-11-01 1205_8 IR642 20 px at FWHM 7.6: one-sided light 8-16 px out
# 16.8% -> 10.6% with the cutoff sized by the FWHM, 6.3% following the
# flare, the planet's own down-left glow moved back with it (limb change
# 2.9% -> 4.7% of its peak, no new rings). L3 stacks reach 1.3-2.1 FWHM.
# The phase cutoff follows the worst moon's flare out (from the core's edge;
# a longer kernel reach only added error: synthetic coma 3.2x -> 2.1x).
# Only with a significant coma or trefoil: otherwise the odd part is glow
# and noise.
_FLARE_FRACTION = 0.1
_FLARE_SMOOTH = 1.5
_FLARE_SIGMAS = 4.0
# Auto combines moons scoring at least this fraction of the best.
_COMBINE_MIN_SCORE = 0.25
# Moons disagreeing this much (resultant/total below it) on a significant
# aberration are reported: it probably varies across the field.
_AGREEMENT_WARN = 0.5
_SIGNIFICANT_AMOUNT = 0.03
# Elongation the moons share. Drift during a long stack (orbital motion, or
# field rotation on an alt-az mount) smears each moon by its own amount and
# direction, but not the planet; optics, ADC and dispersion elongate every
# moon and the planet alike. Smearing adds to a moon's covariance, raising its
# elongation by at most what it raises its size, so the shared part is the
# largest common covariance every moon could have been smeared from. Where it
# is this fraction of the combined elongation or more the combination is
# used as it is; at the lower one or less only the shared part is corrected,
# blended between. The 24 2026-11-21 and 2026-11-01 stacks (real
# astigmatism) keep 0.85-1.0 of the correction; the 2026-11-03 prefoil stack
# (none; its moons' elongations differ up to 2.4x) keeps 0.55.
_SHARED_KEEP = 0.9
_SHARED_ONLY = 0.7
# Below this elongation (fraction of the core size) there is nothing to share.
_SHARED_MIN_ELONGATION = 0.02
# The panel reports a correction cut below this fraction.
_SHARED_REPORT = 0.95
# Auto-select: background variation under the moon at this fraction of its
# peak halves the score.
_GLOW_PENALTY_FRAC = 0.1
# Protect background: signal is where the locally smoothed image stands this
# many noise sigmas above the sky (soft step between the two), widened by
# this many long-axis sigmas so the blend lies where the correction changes
# only noise.
_SIGNAL_LOW_SIGMA = 3.0
_SIGNAL_HIGH_SIGMA = 6.0
_MASK_REACH_PER_SIGMA = 3.0
# The astigmatism boost amplifies noise, so with Protect background each
# pixel gets the strongest boost whose noise gain stays under 1/8 of its
# local signal-to-noise (smoothed over a long-axis width, in log scale so the
# bright planet dominates nearby): full on the planet and a moon's core,
# gentle on its faint outer glow. One boost everywhere speckled the moons
# (2026-11-21 IR642 at 4x: glow noise 12-25 px out x3.3; adaptive x1.03 with
# the moon still 13.6x9.0 -> 9.6x8.2 px and ringing unchanged at 0.003%).
_BOOST_SNR_PER_GAIN = 8.0
_BOOST_LEVELS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
# Per pixel is not enough: on the planet every pixel has signal, but along
# the long axis its finest scales are already at the noise, where a boost
# adds only noise for later sharpening to raise further (2026-11-21 IR642:
# long-axis signal/noise power 8.7 at 0.2 cycles/px, 1.9 at 0.3, boosted 3x;
# noise score 0.26 -> 0.54, and Wavelet Auto had to cut fine from 22 to 5).
# Each frequency's boost K is cut to 1 + (K - 1) s/(s + K^2), s the image's
# signal/noise power there: the full boost only where the signal outweighs
# the noise it amplifies. Same IR642 stack: noise 0.37; after Wavelet Auto
# at the same noise, long/short-axis edge gradient 4.08/11.55 against
# 4.06/11.32 unlimited and 3.80/11.48 uncorrected.
# Noise is the mean power beyond this frequency; the power is smoothed over
# this many frequency bins.
_NOISE_FREQ = 0.4
_SPECTRUM_SMOOTH = 2.0


@dataclass(frozen=True)
class PsfMeasurement:
    """A measured PSF stamp and its shape, in the plane it was measured on."""

    x: float  # centroid, image pixels
    y: float
    psf: np.ndarray  # (2R+1, 2R+1), sum 1, centred
    fwhm_major: float
    fwhm_minor: float
    angle_deg: float  # major axis, degrees anticlockwise from image +x (y down)
    sigma_major: float  # fitted core Gaussian widths along the axes
    sigma_minor: float
    snr: float
    spectral_noise: float  # per-frequency noise in the normalised OTF
    pixel_noise: float = 0.0  # sky noise per pixel, in units of the normalised psf
    # Background variation under the stamp relative to the moon's peak: a
    # moon on the planet's glow is measured less reliably.
    background_slope: float = 0.0
    # Angular harmonic strengths of the core (one-sided, two-fold, three-fold).
    coma_amount: float = 0.0
    astigmatism_amount: float = 0.0
    trefoil_amount: float = 0.0
    # Complex harmonic coefficients (m -> strength and direction) and, for a
    # combination of moons, how well the moons agree (1 = same, 0 = cancel)
    # and the moons' own mean strength.
    harmonics: dict = field(default_factory=dict)
    # How far the one-sided flare reaches from the centre, px (0 = no
    # significant coma or trefoil).
    flare_radius: float = 0.0
    agreement: dict = field(default_factory=dict)
    moon_amounts: dict = field(default_factory=dict)
    # Moons measured: (x, y, radius) for the canvas markers.
    moons: tuple = ()
    # For a combination: the ellipse the astigmatism boost corrects
    # (sigma_major, sigma_minor, angle_deg), None = the measured one, and the
    # fraction of the combined elongation the moons share (1 = all of it).
    boost_ellipse: tuple | None = None
    shared_elongation: float = 1.0

    @property
    def radius(self) -> int:
        return self.psf.shape[0] // 2

    @property
    def elongation(self) -> float:
        return self.fwhm_major / max(self.fwhm_minor, 1e-6)


@dataclass(frozen=True)
class MoonChoice:
    x: float
    y: float
    snr: float
    fwhm: float
    auto: bool


def _plane(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    return np.asarray(_luminance(data, is_grayscale), dtype=np.float64)


def _candidates(lum: np.ndarray) -> list[MoonDetection]:
    detections, _ = _cached_detections(
        lum.astype(np.float32, copy=False), _DETECT_SENSITIVITY, 0.0, _DETECT_MAX,
        DEFAULT_PLANET_FLOOR,
    )
    return detections


def _background_plane(stamp: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares quadratic surface through ``use`` pixels, and the noise.

    Quadratic, not a plane: a moon close to the planet sits on curved glow
    (2026-11-01 IR642: 15-30% of the moon's peak), which a plane leaves as a
    lopsided pedestal that reads as coma.
    """
    n = stamp.shape[0]
    yy, xx = (np.mgrid[:n, :n].astype(np.float64) - n//2)/max(n//2, 1)
    if use.sum() < 12:
        level = float(np.median(stamp))
        return np.full_like(stamp, level), float(np.std(stamp))
    terms = (np.ones_like(xx), xx, yy, xx*xx, yy*yy, xx*yy)
    a = np.stack([t[use] for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(a, stamp[use], rcond=None)
    plane = sum(c*t for c, t in zip(coef, terms))
    resid = stamp[use] - plane[use]
    # MAD: robust to a faint unmasked star or glow knot in the annulus.
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return plane, max(noise, float(np.std(resid)) * 0.1)


def _half_max_fwhm(stamp: np.ndarray) -> float:
    peak = float(stamp.max())
    if peak <= 0:
        return 0.0
    return 2.0 * float(np.sqrt((stamp >= 0.5*peak).sum() / np.pi))


def _cut(plane: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray | None:
    h, w = plane.shape
    if cx - half < 0 or cy - half < 0 or cx + half >= w or cy + half >= h:
        return None
    return plane[cy-half:cy+half+1, cx-half:cx+half+1].copy()


def _axis_fwhm(psf: np.ndarray, angle: float) -> float:
    """Half-maximum width along a line through the centre at ``angle``."""
    r = psf.shape[0] // 2
    t = np.arange(-r, r + 0.001, 0.25)
    coords = np.vstack([r + t*np.sin(angle), r + t*np.cos(angle)])
    prof = ndimage.map_coordinates(psf, coords, order=1)
    peak = float(prof[len(t)//2])
    if peak <= 0:
        return 0.0
    above = prof >= 0.5*peak
    centre = len(t)//2
    left = centre
    while left > 0 and above[left - 1]:
        left -= 1
    right = centre
    while right < len(t) - 1 and above[right + 1]:
        right += 1

    def cross(i0, i1):
        p0, p1 = prof[i0], prof[i1]
        f = (0.5*peak - p0) / (p1 - p0) if p1 != p0 else 0.0
        return t[i0] + f*(t[i1] - t[i0])
    lo = cross(left - 1, left) if left > 0 else t[0]
    hi = cross(right + 1, right) if right < len(t) - 1 else t[-1]
    return float(hi - lo)


def measure_psf(
    plane: np.ndarray,
    x: float,
    y: float,
    *,
    neighbours: list[tuple[float, float, float]] = (),
    snr: float = 0.0,
) -> PsfMeasurement | None:
    """Measure the PSF of the moon nearest (x, y) on one image plane.

    ``neighbours`` are (x, y, fwhm) of other moons to mask out of the stamp.
    Returns None when no usable moon is there (off the edge, flat or empty).
    """
    plane = np.asarray(plane, dtype=np.float64)
    h, w = plane.shape
    cx, cy = int(round(x)), int(round(y))
    # First pass: a small stamp to size the PSF.
    half0 = 12
    fwhm = 0.0
    for _ in range(3):
        stamp = _cut(plane, cx, cy, half0 + _ANNULUS_PX)
        if stamp is None:
            break
        n = stamp.shape[0]
        yy, xx = np.mgrid[:n, :n] - n//2
        rr = np.hypot(xx, yy)
        bg, _ = _background_plane(stamp, rr > half0)
        core = (stamp - bg)[_ANNULUS_PX:-_ANNULUS_PX, _ANNULUS_PX:-_ANNULUS_PX]
        fwhm = _half_max_fwhm(core)
        want = int(np.clip(np.ceil(_RADIUS_PER_FWHM*fwhm) + 2, _MIN_RADIUS, _MAX_RADIUS))
        if want <= half0:
            break
        half0 = want
    radius = int(np.clip(np.ceil(_RADIUS_PER_FWHM*max(fwhm, 1.0)) + 2,
                         _MIN_RADIUS, _MAX_RADIUS))
    # Shrink to fit the image edge; too small means no useful stamp.
    edge = min(cx, cy, w - 1 - cx, h - 1 - cy) - _ANNULUS_PX - 2
    radius = min(radius, edge)
    if radius < _MIN_RADIUS - 2:
        return None
    outer = radius + _ANNULUS_PX
    big = _cut(plane, cx, cy, outer + 2)
    if big is None:
        return None
    n = big.shape[0]
    yy, xx = (np.mgrid[:n, :n] - n//2).astype(np.float64)
    rr = np.hypot(xx, yy)
    masked = np.zeros((n, n), bool)
    for nx, ny, nf in neighbours:
        dx, dy = nx - cx, ny - cy
        if np.hypot(dx, dy) < 1.0:
            continue
        # Never reach into the chosen moon's own core.
        reach = min(max(1.2*nf, 3.0), 0.6*float(np.hypot(dx, dy)))
        masked |= np.hypot(xx - dx, yy - dy) <= reach
    annulus = (rr > radius) & (rr <= outer) & ~masked
    bg, noise = _background_plane(big, annulus)
    sub = big - bg
    inner = rr <= radius
    background_range = float(bg[inner].max() - bg[inner].min())
    # Refill masked neighbours by point reflection through the moon.
    if masked.any():
        mirrored = sub[::-1, ::-1]
        sub = np.where(masked & ~masked[::-1, ::-1], mirrored, sub)
        sub[masked & masked[::-1, ::-1]] = 0.0
    # Centre on the peak, not the centroid: a coma tail pulls the centroid
    # towards itself and would hide most of the coma it should measure.
    my, mx = 0.0, 0.0
    near = np.hypot(xx, yy) <= max(3.0, 0.75*fwhm)
    smooth = ndimage.gaussian_filter(sub, 1.0)
    if not (np.clip(sub, 0, None)*near).any():
        return None
    iy, ix = np.unravel_index(int(np.argmax(np.where(near, smooth, -np.inf))), smooth.shape)
    if 0 < iy < n - 1 and 0 < ix < n - 1:
        def vertex(a, b, c):
            d = a - 2*b + c
            return 0.0 if d >= 0 else float(np.clip(0.5*(a - c)/d, -0.5, 0.5))
        my = iy - n//2 + vertex(smooth[iy-1, ix], smooth[iy, ix], smooth[iy+1, ix])
        mx = ix - n//2 + vertex(smooth[iy, ix-1], smooth[iy, ix], smooth[iy, ix+1])
    # Coma moves the core off the stamp centre a little; further is not a moon.
    if np.hypot(mx, my) > max(3.0, 0.25*radius):
        return None
    shifted = ndimage.shift(sub, (-my, -mx), order=3, mode="nearest")
    c = n//2
    psf = shifted[c-radius:c+radius+1, c-radius:c+radius+1]
    m = psf.shape[0]
    py, px = (np.mgrid[:m, :m] - radius).astype(np.float64)
    pr = np.hypot(px, py)
    # Cosine taper over the outer quarter keeps the OTF free of edge ripple.
    t0 = 0.75*radius
    taper = np.where(pr <= t0, 1.0,
                     np.where(pr >= radius, 0.0,
                              0.5*(1 + np.cos(np.pi*(pr - t0)/(radius - t0)))))
    psf = psf*taper
    flux = float(psf.sum())
    peak = float(psf.max())
    if flux <= 0 or peak <= 0 or peak <= 3*noise:
        return None
    psf_n = psf / flux
    # Per-pixel noise relative to flux, summed incoherently over the taper.
    spectral_noise = float(noise/flux*np.sqrt((taper**2).sum()))
    return _describe(
        psf_n, cx + mx, cy + my,
        snr=float(snr) if snr else float(peak/max(noise, 1e-12)),
        spectral_noise=spectral_noise, pixel_noise=float(noise/flux),
        background_slope=background_range/max(peak, 1e-12),
    )


def _describe(
    psf_n: np.ndarray, x: float, y: float, *, snr: float, spectral_noise: float,
    pixel_noise: float, background_slope: float,
) -> PsfMeasurement:
    """Shape of a centred, normalised PSF: widths, ellipse and harmonics."""
    radius = psf_n.shape[0]//2
    py, px = (np.mgrid[:psf_n.shape[0], :psf_n.shape[1]] - radius).astype(np.float64)
    pr = np.hypot(px, py)
    # Second moments of the core start the ellipse fit.
    wgt = np.clip(psf_n, 0, None) * (pr <= max(2.0, 1.2*_half_max_fwhm(psf_n)))
    tw = float(wgt.sum()) + 1e-30
    sxx = float((wgt*px*px).sum()/tw)
    syy = float((wgt*py*py).sum()/tw)
    sxy = float((wgt*px*py).sum()/tw)
    evals, evecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
    major = evecs[:, 1]
    angle = float(np.arctan2(major[1], major[0]))
    fw_major = _axis_fwhm(psf_n, angle)
    fw_minor = _axis_fwhm(psf_n, angle + np.pi/2)
    if fw_major < fw_minor:
        fw_major, fw_minor = fw_minor, fw_major
        angle += np.pi/2
    # Astigmatism is fitted on the point-symmetric part, so a coma tail or
    # trefoil lobes do not read as elongation.
    symmetric = 0.5*(psf_n + psf_n[::-1, ::-1])
    sig_major, sig_minor, angle = _fit_core_ellipse(symmetric, fw_major, fw_minor, angle)
    fw_major = _axis_fwhm(psf_n, angle)
    fw_minor = _axis_fwhm(psf_n, angle + np.pi/2)
    if fw_major < fw_minor:
        fw_major, fw_minor = fw_minor, fw_major
    vectors = _harmonic_vectors(psf_n, max(fw_minor, 1.0))
    flare = 0.0
    if max(abs(vectors[1]), abs(vectors[3])) >= _SIGNIFICANT_AMOUNT:
        flare = _flare_radius(psf_n, pixel_noise)
    deg = float(np.degrees(-angle)) % 180.0  # y is down: report anticlockwise
    if deg > 90.0:
        deg -= 180.0
    return PsfMeasurement(
        x=float(x), y=float(y), psf=psf_n,
        fwhm_major=fw_major, fwhm_minor=fw_minor, angle_deg=deg,
        sigma_major=float(sig_major), sigma_minor=float(sig_minor),
        snr=float(snr), spectral_noise=float(spectral_noise),
        pixel_noise=float(pixel_noise), background_slope=float(background_slope),
        coma_amount=abs(vectors[1]), astigmatism_amount=abs(vectors[2]),
        trefoil_amount=abs(vectors[3]), harmonics=vectors, flare_radius=flare,
        moons=((float(x), float(y), float(max(4.0, fw_major))),),
    )


def _flare_radius(psf: np.ndarray, pixel_noise: float) -> float:
    """How far the one-sided part of a centred PSF reaches: the radius past
    its peak where the smoothed odd part falls below a tenth of its peak."""
    r = psf.shape[0]//2
    odd = ndimage.gaussian_filter(0.5*(psf - psf[::-1, ::-1]), _FLARE_SMOOTH)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ring = np.floor(np.hypot(xx, yy)).astype(int)
    strength = ndimage.maximum(np.abs(odd), ring, index=np.arange(r))
    strength = np.asarray(strength, dtype=np.float64)
    # Pixel noise of the odd part (a difference of two pixels, halved) after smoothing.
    noise = pixel_noise/np.sqrt(2)/(2*np.sqrt(np.pi)*_FLARE_SMOOTH)
    peak = int(np.argmax(strength))
    floor = max(_FLARE_FRACTION*float(strength[peak]), _FLARE_SIGMAS*noise)
    below = np.nonzero(strength[peak:] < floor)[0]
    return float(peak + below[0]) if below.size else float(r)


def _harmonic_profiles(psf: np.ndarray, orders: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    """Complex angular harmonic profiles c_m(r) of a centred stamp."""
    r = psf.shape[0]//2
    radii = np.arange(0.0, r + 0.5, 0.5)
    count = max(64, int(np.ceil(4*np.pi*r)))
    theta = np.linspace(0.0, 2*np.pi, count, endpoint=False)
    rad, th = np.meshgrid(radii, theta, indexing="ij")
    values = ndimage.map_coordinates(psf, [r + rad*np.sin(th), r + rad*np.cos(th)],
                                     order=1, mode="constant")
    profiles = {m: (values*np.exp(-1j*m*th)).mean(axis=1) for m in orders}
    return radii, profiles


def _harmonic_vectors(psf: np.ndarray, fwhm: float) -> dict[int, complex]:
    """The m = 1, 2, 3 angular harmonics over the core (0.3-1.5 FWHM) relative
    to the round part (coma, astigmatism, trefoil): size and direction."""
    radii, prof = _harmonic_profiles(psf, (0, 1, 2, 3))
    use = (radii >= 0.3*fwhm) & (radii <= 1.5*fwhm)
    if not use.any():
        return {1: 0j, 2: 0j, 3: 0j}
    base = float((np.abs(prof[0][use])*radii[use]).sum()) + 1e-30
    return {m: complex((prof[m][use]*radii[use]).sum())/base for m in (1, 2, 3)}


def combine_measurements(
    measurements: list[PsfMeasurement], weights: list[float],
) -> PsfMeasurement | None:
    """Weighted mean of several moons' PSFs, described like a single moon.

    Uniform aberrations (collimation coma, trefoil, astigmatism) are kept and
    measured with more signal; where they vary across the field, moons on
    different sides partly cancel, and ``agreement`` reports it.
    """
    pairs = [(m, float(w)) for m, w in zip(measurements, weights) if m is not None and w > 0]
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0][0]
    radius = max(m.radius for m, _ in pairs)
    size = 2*radius + 1
    total = np.zeros((size, size))
    wsum = sum(w for _, w in pairs)
    for m, w in pairs:
        off = radius - m.radius
        total[off:off + m.psf.shape[0], off:off + m.psf.shape[1]] += w*m.psf
    psf = total/max(float(total.sum()), 1e-30)
    noise = float(np.sqrt(sum((w*m.spectral_noise)**2 for m, w in pairs)))/wsum
    pixel = float(np.sqrt(sum((w*m.pixel_noise)**2 for m, w in pairs)))/wsum
    best = max(pairs, key=lambda item: item[1])[0]
    out = _describe(psf, best.x, best.y, snr=best.snr, spectral_noise=noise,
                    pixel_noise=pixel,
                    background_slope=sum(w*m.background_slope for m, w in pairs)/wsum)
    agreement, amounts = {}, {}
    for order in (1, 2, 3):
        vec = sum(w*m.harmonics.get(order, 0j) for m, w in pairs)
        size_sum = sum(w*abs(m.harmonics.get(order, 0j)) for m, w in pairs) + 1e-30
        agreement[order] = float(abs(vec)/size_sum)
        amounts[order] = float(size_sum/wsum)
    boost, shared = (_shared_ellipse([m for m, _ in pairs], out)
                     if ASTIGMATISM_ENABLED else (None, 1.0))
    # The correction reaches as far as the worst moon's flare.
    flare = max([out.flare_radius] + [m.flare_radius for m, _ in pairs])
    return replace(out, agreement=agreement, moon_amounts=amounts, flare_radius=flare,
                   moons=tuple(mk for m, _ in pairs for mk in m.moons),
                   boost_ellipse=boost, shared_elongation=shared)


def _covariance(sigma_major: float, sigma_minor: float, angle_deg: float) -> tuple[float, complex]:
    """Core size (mean variance) and elongation (half the variance difference,
    at twice the image-orientation angle) of a fitted ellipse, in px²."""
    th = np.radians(-angle_deg)
    return (0.5*(sigma_major**2 + sigma_minor**2),
            0.5*(sigma_major**2 - sigma_minor**2)*np.exp(2j*th))


def _shared_ellipse(
    measurements: list[PsfMeasurement], combined: PsfMeasurement,
) -> tuple[tuple[float, float, float] | None, float]:
    """The elongation every moon shares, as the ellipse to correct, and its
    fraction of the combined elongation.

    A smear adds a covariance whose elongation is at most its size, so a
    moon (size s, elongation E) came from a shared core (sf, Ef) only if
    |E - Ef| <= s - sf. The shared core taken is the largest that fits every
    moon: the least smear that explains the differences between them.
    """
    size, elong = _covariance(combined.sigma_major, combined.sigma_minor, combined.angle_deg)
    if abs(elong) < _SHARED_MIN_ELONGATION*size:
        return None, 1.0
    moons = [_covariance(m.sigma_major, m.sigma_minor, m.angle_deg) for m in measurements]

    def excess(p):
        ef = complex(p[0], p[1])
        return max(abs(e - ef) - s for s, e in moons)

    fit = optimize.minimize(excess, [elong.real, elong.imag], method="Nelder-Mead",
                            options={"xatol": 1e-4*size, "fatol": 1e-5*size})
    shared = complex(fit.x[0], fit.x[1])
    ratio = min(abs(shared)/abs(elong), 1.0)
    if ratio >= _SHARED_KEEP:
        return None, ratio
    t = float(np.clip((ratio - _SHARED_ONLY)/(_SHARED_KEEP - _SHARED_ONLY), 0.0, 1.0))
    # Never more elongation than the combination itself shows.
    if abs(shared) > abs(elong):
        shared *= abs(elong)/abs(shared)
    target = t*elong + (1.0 - t)*shared
    spread = min(abs(target), 0.95*size)
    angle = float(np.degrees(-0.5*np.angle(target)))
    angle = (angle + 90.0) % 180.0 - 90.0
    return ((float(np.sqrt(size + spread)), float(np.sqrt(size - spread)), angle),
            float(abs(target)/abs(elong)))


def _without_harmonics(psf: np.ndarray, orders: tuple[int, ...]) -> np.ndarray:
    """The stamp with the chosen angular harmonics removed."""
    r = psf.shape[0]//2
    radii, prof = _harmonic_profiles(psf, orders)
    yy, xx = (np.mgrid[:psf.shape[0], :psf.shape[1]] - r).astype(np.float64)
    rad = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    out = psf.astype(np.float64).copy()
    for m in orders:
        c = np.interp(rad, radii, prof[m].real) + 1j*np.interp(rad, radii, prof[m].imag)
        out -= 2.0*np.real(c*np.exp(1j*m*phi))
    return out


def _fit_core_ellipse(
    psf: np.ndarray, fw_major: float, fw_minor: float, angle: float,
) -> tuple[float, float, float]:
    """Elliptical Gaussian (plus offset) fitted to the core: σa ≥ σb and angle.

    Uses every core pixel, so a noisy moon still gives stable widths; the
    offset absorbs a sharpened moon's negative ring and any residual sky.
    """
    r = psf.shape[0]//2
    yy, xx = (np.mgrid[:psf.shape[0], :psf.shape[1]] - r).astype(np.float64)
    sel = np.hypot(xx, yy) <= max(2.0, _FIT_REACH_PER_FWHM*fw_major)
    x, y, z = xx[sel], yy[sel], psf[sel]
    sa0 = max(fw_major/2.355, 0.4)
    sb0 = max(fw_minor/2.355, 0.4)

    def residual(q):
        amp, sa, sb, th, off = q
        u = x*np.cos(th) + y*np.sin(th)
        v = -x*np.sin(th) + y*np.cos(th)
        return amp*np.exp(-0.5*((u/sa)**2 + (v/sb)**2)) + off - z

    try:
        fit = optimize.least_squares(
            residual, [float(z.max()), sa0, sb0, angle, 0.0],
            bounds=([0.0, 0.3, 0.3, -np.inf, -np.inf], [np.inf, 4*r, 4*r, np.inf, np.inf]))
        _, sa, sb, th, _ = fit.x
    except (ValueError, np.linalg.LinAlgError):
        sa, sb, th = sa0, sb0, angle
    if sb > sa:
        sa, sb, th = sb, sa, th + np.pi/2
    th = (th + np.pi/2) % np.pi - np.pi/2
    return float(sa), float(sb), float(th)


def _neighbours_of(
    detections: list[MoonDetection], x: float, y: float, peak: float,
) -> list[tuple[float, float, float]]:
    return [(float(d.x), float(d.y), float(d.fwhm)) for d in detections
            if np.hypot(d.x - x, d.y - y) >= 2.0 and d.peak_l >= _NEIGHBOUR_FRAC*peak]


def auto_select_moons(
    data: np.ndarray, is_grayscale: bool,
) -> list[tuple[MoonChoice, float]]:
    """Suitable moons, best first, with their weights for combining.

    Bright enough, compact, isolated, unclipped and off the planet's glow;
    moons scoring under a quarter of the best are left out.
    """
    lum = _plane(data, is_grayscale)
    detections = _candidates(lum)
    top = float(lum.max())
    if not detections:
        return []
    # Faint blips (a quantisation step or noise knot) measure deceptively
    # compact, so only sources within reach of the strongest moon compete.
    strongest = max(d.snr for d in detections)
    floor = max(_AUTO_MIN_SNR, _AUTO_RELATIVE_SNR*min(strongest, _AUTO_SNR_CAP))
    measured = []
    for d in detections:
        if d.snr < floor or d.peak_l >= _CLIPPED_FRAC*top:
            continue
        m = measure_psf(lum, d.x, d.y, neighbours=_neighbours_of(detections, d.x, d.y, d.peak_l),
                        snr=d.snr)
        # A hot pixel or noise spike, not a moon blurred by the optics.
        if m is None or m.fwhm_minor < _MIN_FWHM:
            continue
        measured.append((d, m, float(np.sqrt(m.fwhm_major*m.fwhm_minor))))
    if not measured:
        return []
    # Every moon shares the PSF; far smaller than the typical source is noise.
    typical = float(np.median([size for *_, size in measured]))
    measured = [item for item in measured if item[2] >= _MIN_SIZE_FRAC*typical]
    compact = min(size for *_, size in measured)
    scored: list[tuple[float, MoonChoice]] = []
    for d, m, size in measured:
        # Bright neighbours overlapping the stamp cost accuracy even when masked.
        crowd = sum(1 for o in detections if o is not d
                    and np.hypot(o.x - d.x, o.y - d.y) < m.radius + o.fwhm
                    and o.peak_l >= 0.2*d.peak_l)
        # Brightness beyond the cap only breaks ties between equal sizes;
        # a resolved disc such as Titan loses to a smaller moon.
        # A moon on the planet's glow is measured less reliably.
        glow = 1.0/(1.0 + (m.background_slope/_GLOW_PENALTY_FRAC)**2)
        score = (min(d.snr, _AUTO_SNR_CAP)*(compact/size)**_COMPACT_POWER
                 * 0.5**crowd * glow * (1.0 + 0.02*np.log10(1.0 + d.snr)))
        scored.append((score, MoonChoice(m.x, m.y, d.snr, size, True)))
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][0]
    return [(choice, score/best) for score, choice in scored
            if score >= _COMBINE_MIN_SCORE*best]


def auto_select_moon(data: np.ndarray, is_grayscale: bool) -> MoonChoice | None:
    """The single most suitable moon."""
    moons = auto_select_moons(data, is_grayscale)
    return moons[0][0] if moons else None


def locate_moon(
    data: np.ndarray, is_grayscale: bool, x: float, y: float,
    search: float = _PICK_SEARCH_PX,
) -> MoonChoice | None:
    """The moon nearest a picked point: a detection, else the local peak."""
    lum = _plane(data, is_grayscale)
    h, w = lum.shape
    detections = _candidates(lum)
    near = [d for d in detections if np.hypot(d.x - x, d.y - y) <= search]
    if near:
        d = min(near, key=lambda d: (np.hypot(d.x - x, d.y - y)
                                     - 0.25*search*min(d.snr, 100.0)/100.0))
        return MoonChoice(float(d.x), float(d.y), d.snr, d.fwhm, False)
    r = _PICK_LOCAL_PX
    x0, x1 = int(max(0, x - r)), int(min(w, x + r + 1))
    y0, y1 = int(max(0, y - r)), int(min(h, y + r + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    win = lum[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    return MoonChoice(float(x0 + ix), float(y0 + iy), 0.0, 0.0, False)


def choose_moon(data: np.ndarray, is_grayscale: bool, params: dict) -> MoonChoice | None:
    if params.get("auto", True):
        return auto_select_moon(data, is_grayscale)
    x, y = float(params.get("moon_x", -1.0)), float(params.get("moon_y", -1.0))
    if x < 0 or y < 0:
        return None
    return locate_moon(data, is_grayscale, x, y)


def choose_moons(
    data: np.ndarray, is_grayscale: bool, params: dict,
) -> list[tuple[MoonChoice, float]]:
    """Auto: every suitable moon with its weight; manual: the picked moon."""
    if params.get("auto", True):
        return auto_select_moons(data, is_grayscale)
    choice = choose_moon(data, is_grayscale, params)
    return [(choice, 1.0)] if choice is not None else []


def measure_moons(
    data: np.ndarray, is_grayscale: bool, moons: list[tuple[MoonChoice, float]],
    plane: np.ndarray | None = None,
) -> PsfMeasurement | None:
    """The weighted combination of the moons' PSFs measured on ``plane``."""
    return combine_measurements(
        [measure_choice(data, is_grayscale, c, plane=plane) for c, _ in moons],
        [w for _, w in moons])


def elongation_differs(m: PsfMeasurement) -> bool:
    """The moons' elongations differ enough that the astigmatism boost was
    cut to the part they share (``m.shared_elongation``)."""
    return m.boost_ellipse is not None and m.shared_elongation < _SHARED_REPORT


def disagreements(m: PsfMeasurement) -> list[str]:
    """Aberrations the combined moons disagree on."""
    names = {1: "Coma/collimation", 2: "Astigmatism", 3: "Trefoil"}
    out = []
    for order, agree in sorted(m.agreement.items()):
        # Astigmatism differing between moons is handled by shared_elongation.
        if order == 2:
            continue
        # Significant on the moons themselves, but mostly cancelling.
        if agree < _AGREEMENT_WARN and m.moon_amounts.get(order, 0.0) > _SIGNIFICANT_AMOUNT:
            out.append(names[order])
    return out


def measure_choice(
    data: np.ndarray, is_grayscale: bool, choice: MoonChoice, plane: np.ndarray | None = None,
) -> PsfMeasurement | None:
    lum = _plane(data, is_grayscale)
    if plane is None:
        plane = lum
    detections = _candidates(lum)
    iy, ix = int(round(choice.y)), int(round(choice.x))
    h, w = lum.shape
    peak = float(lum[min(max(iy, 0), h - 1), min(max(ix, 0), w - 1)])
    return measure_psf(plane, choice.x, choice.y,
                       neighbours=_neighbours_of(detections, choice.x, choice.y, peak),
                       snr=choice.snr)


def _otf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    r = kernel.shape[0]//2
    k = np.zeros(shape, dtype=np.float64)
    k[:kernel.shape[0], :kernel.shape[1]] = kernel
    return rfft2(np.roll(k, (-r, -r), axis=(0, 1)))


def _phase_otf(
    m: PsfMeasurement, shape: tuple[int, int], orders: tuple[int, ...],
) -> np.ndarray:
    """Unit-magnitude transfer that removes the chosen one-sided (m = 1, coma)
    and three-fold (m = 3, trefoil) parts of the PSF.

    These odd aberrations displace light rather than spread it, so in the
    spectrum they are a phase error: correcting it moves the light back and
    amplifies nothing. Only where the moon's spectrum is measured reliably,
    and not at the largest scales, where the moon's faint halo and any glow
    under it would distort a planet 1000x brighter.
    """
    h = _otf(m.psf, shape)
    kept = _otf(_without_harmonics(m.psf, orders), shape)
    delta = np.angle(h*np.conj(kept))
    power = np.abs(h)**2
    weight = power/(power + (_NOISE_FLOOR_SIGMAS*m.spectral_noise)**2 + 1e-30)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    # A flare reaching further than the core is corrected out to its tip.
    f0 = _PHASE_LOW_FREQ/max(m.fwhm_major, m.flare_radius - 0.5*m.fwhm_major, 1.0)
    weight = weight*(1.0 - np.exp(-(fx*fx + fy*fy)/(f0*f0)))
    phase = np.exp(-1j*weight*delta)
    # Coma and trefoil move light within the PSF: the kernel cannot reach
    # further than the moon's blur, and anything beyond is measurement noise.
    kernel = irfft2(phase, shape)
    dy = (np.fft.fftfreq(shape[0])*shape[0])[:, None]
    dx = (np.fft.fftfreq(shape[1])*shape[1])[None, :]
    dist = np.hypot(dy, dx)
    reach = max(3.0, _PHASE_KERNEL_REACH*m.radius)
    window = np.where(dist <= 0.7*reach, 1.0, np.where(
        dist >= reach, 0.0, 0.5*(1.0 + np.cos(np.pi*(dist - 0.7*reach)/(0.3*reach)))))
    kernel *= window
    kernel /= kernel.sum()
    return rfft2(kernel)


def correction_otf(
    m: PsfMeasurement, shape: tuple[int, int], target_size: float, max_boost: float,
    *, astigmatism: bool = True, coma: bool = True, trefoil: bool = True,
) -> np.ndarray:
    """Transfer function correcting the chosen aberrations of the measured moon."""
    k: np.ndarray | complex = 1.0
    if astigmatism:
        k = _ellipse_otf(m, shape, target_size, max_boost)
    orders = tuple(order for order, on in ((1, coma), (3, trefoil)) if on)
    if orders:
        k = k*_phase_otf(m, shape, orders)
    if np.isscalar(k):
        return np.ones((shape[0], shape[1]//2 + 1), dtype=np.complex128)
    return k


def _ellipse_otf(
    m: PsfMeasurement, shape: tuple[int, int], target_size: float, max_boost: float,
) -> np.ndarray:
    """Real, even transfer function ≥ 1 that makes the fitted core round."""
    boost = float(max_boost)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    if boost <= 1.0:
        return np.ones((shape[0], shape[1]//2 + 1))
    # Only the elongation the moons share, when they differ (drift).
    sa, sb, deg = m.boost_ellipse or (m.sigma_major, m.sigma_minor, m.angle_deg)
    th = np.radians(-deg)  # back to image (y down) orientation
    fu = fx*np.cos(th) + fy*np.sin(th)
    fv = -fx*np.sin(th) + fy*np.cos(th)
    st2 = (max(float(target_size), 1.0)/100.0*sb)**2
    exponent = 2*np.pi**2*(max(sa**2 - st2, 0.0)*fu*fu
                           + max(sb**2 - st2, 0.0)*fv*fv)
    full = np.exp(np.minimum(exponent, 50.0))
    # Soft cap: 1 at zero frequency, follows the full inverse while it is
    # small and bends smoothly up to G; a sharp corner rings more for the
    # same correction.
    return 1.0/((1.0 - 1.0/boost)/full + 1.0/boost)


def _soft_step(x: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((x - low)/(high - low), 0.0, 1.0)
    return t*t*(3.0 - 2.0*t)


def _signal_maps(
    lum: np.ndarray, sigma_minor: float, sigma_major: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky mask (1 on the planet, its glow and moons; 0 on sky holding only
    noise) and the per-pixel signal-to-noise above the sky level.

    The noise is measured on the quieter half of the background-subtracted
    image (a 16-bit stack's sky is flat quantisation steps, where a MAD is 0).
    """
    lum = np.asarray(lum, dtype=np.float64)
    h, w = lum.shape
    small = ndimage.zoom(lum, 0.25, order=1)
    bg = ndimage.median_filter(small, size=17)
    bg = ndimage.zoom(bg, (h/small.shape[0], w/small.shape[1]), order=1)[:h, :w]
    resid = lum - bg
    s = max(1.0, float(sigma_minor))
    noise = resid - ndimage.gaussian_filter(resid, 2.0)
    quiet = np.abs(resid) <= np.percentile(np.abs(resid), 50)
    smooth = ndimage.gaussian_filter(resid, s)
    sigma = float(np.std(ndimage.gaussian_filter(noise, s)[quiet])) + 1e-12
    mask = _soft_step(smooth/sigma, _SIGNAL_LOW_SIGMA, _SIGNAL_HIGH_SIGMA)
    reach = max(1.0, _MASK_REACH_PER_SIGMA*float(sigma_major))
    mask = ndimage.maximum_filter(mask, size=2*int(np.ceil(reach)) + 1)
    mask = np.clip(ndimage.gaussian_filter(mask, max(1.0, reach/3.0)), 0.0, 1.0)
    pixel_sigma = float(np.std(noise[quiet])) + 1e-12
    # Signal for the boost is measured against the sky level, not the local
    # median: next to the planet that median is raised by its own glow, so the
    # bright gaps between rings and globe and the ring shadow read as empty,
    # got no boost beside a fully boosted limb, and sharpening turned the
    # seams into dark hooks (2026-11-21 IR642: after Wavelet Auto, limb
    # undershoot 1.24% of the peak, 0.62% with the sky level).
    sky = float(np.percentile(bg, 5))
    signal = ndimage.gaussian_filter(lum - sky, 1.0)
    return mask, signal/pixel_sigma


def spectral_snr(plane: np.ndarray, m: PsfMeasurement) -> np.ndarray:
    """Signal-to-noise power per spatial frequency of a plane, padded as
    ``correct_plane`` pads it (rfft2 layout)."""
    plane = np.asarray(plane, dtype=np.float64)
    pad = _plane_pad(plane.shape, m)
    return _spectral_snr(rfft2(np.pad(plane, pad, mode="reflect")))


def _plane_pad(shape: tuple[int, ...], m: PsfMeasurement) -> int:
    return min(max(2*m.radius, 16), max(1, min(shape) - 1))


def _spectral_snr(spectrum: np.ndarray) -> np.ndarray:
    power = np.abs(spectrum)**2
    h, w = power.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(2*(w - 1))[None, :]
    high = np.hypot(fx, fy) > _NOISE_FREQ
    noise = float(power[high].mean()) if high.any() else float(power.mean())
    smooth = ndimage.gaussian_filter(power, _SPECTRUM_SMOOTH, mode="wrap")
    return np.maximum(smooth/max(noise, 1e-30) - 1.0, 0.0)


def _resample_spectrum(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A per-frequency (rfft2 layout) grid at another padded image shape."""
    target = (shape[0], shape[1]//2 + 1)
    if grid.shape == target:
        return grid
    src_fy = np.fft.fftshift(np.fft.fftfreq(grid.shape[0]))
    src_fx = np.fft.rfftfreq(2*(grid.shape[1] - 1))
    fy = np.fft.fftfreq(shape[0])
    fx = np.fft.rfftfreq(shape[1])
    iy = np.interp(fy, src_fy, np.arange(src_fy.size))
    ix = np.interp(fx, src_fx, np.arange(src_fx.size))
    yy, xx = np.meshgrid(iy, ix, indexing="ij")
    return ndimage.map_coordinates(np.fft.fftshift(grid, axes=0), [yy, xx],
                                   order=1, mode="nearest")


def _limit_boost(k: np.ndarray, snr: np.ndarray | None) -> np.ndarray:
    """The boost where the image has signal to boost: 1 + (K - 1) s/(s + K²)."""
    if snr is None:
        return k
    return 1.0 + (k - 1.0)*snr/(snr + k*k)


def _adaptive_level_weights(
    snr: np.ndarray, levels: list[float], ellipse_for, sigma_major: float,
):
    """Per boost level, a function giving its weight map.

    Each pixel's allowed noise gain is its smoothed signal-to-noise / 8; it
    is placed between the two levels whose white-noise gain brackets that.
    """
    gains = []
    for level in levels:
        k = ellipse_for(level)
        gains.append(float(np.sqrt(np.mean(np.abs(k)**2))))
    smooth = np.exp(ndimage.gaussian_filter(
        np.log(np.maximum(snr, 1e-3)), max(1.0, sigma_major)))
    allowed = np.maximum(smooth/_BOOST_SNR_PER_GAIN, 1.0)
    index = np.interp(allowed, np.maximum.accumulate(np.array(gains)),
                      np.arange(len(levels), dtype=np.float64))

    def weight(i: int) -> np.ndarray:
        return np.clip(1.0 - np.abs(index - i), 0.0, 1.0)
    return weight


def background_mask(lum: np.ndarray, sigma_minor: float, sigma_major: float) -> np.ndarray:
    """1 on the planet, its glow and moons; 0 on sky that holds only noise."""
    return _signal_maps(lum, sigma_minor, sigma_major)[0]


def _apply_otf(plane: np.ndarray, k_fn, pad: int) -> np.ndarray:
    padded = np.pad(np.asarray(plane, dtype=np.float64), pad, mode="reflect")
    k = k_fn(padded.shape)
    out = irfft2(rfft2(padded)*k, padded.shape)
    return out[pad:-pad, pad:-pad]


class NoMoonError(ValueError):
    pass


def correct_optical_aberrations(
    data: np.ndarray,
    is_grayscale: bool,
    *,
    auto: bool = True,
    moon_x: float = -1.0,
    moon_y: float = -1.0,
    target_size: float = 100.0,
    max_boost: float = 2.0,
    per_channel: bool = True,
    protect_background: bool = True,
    astigmatism: bool = False,
    coma: bool = True,
    trefoil: bool = True,
) -> np.ndarray:
    """Correct the moons' aberrations in the image. ``astigmatism`` (with
    ``target_size`` and ``max_boost``) is ignored unless ASTIGMATISM_ENABLED,
    which the tests switch on to keep that code working."""
    astigmatism = astigmatism and ASTIGMATISM_ENABLED
    params = {"auto": auto, "moon_x": moon_x, "moon_y": moon_y}
    arr = np.asarray(data, dtype=np.float32)
    moons = choose_moons(arr, is_grayscale, params)
    if not moons:
        if auto:
            raise NoMoonError("No suitable moon found for Correct Optical Aberrations.")
        raise NoMoonError("No moon selected: click a moon or drag a box around it.")
    base = measure_moons(arr, is_grayscale, moons)
    if base is None:
        choice = moons[0][0]
        raise NoMoonError(
            f"Could not measure a PSF at ({choice.x:.0f}, {choice.y:.0f}).")

    mask = snr = None
    if protect_background:
        mask, snr = signal_maps(_plane(arr, is_grayscale), base)
    settings = dict(target_size=target_size, max_boost=max_boost,
                    astigmatism=astigmatism, coma=coma, trefoil=trefoil)

    if arr.ndim == 2:
        return correct_plane(arr, base, mask, snr, **settings)
    if is_grayscale:
        out = correct_plane(arr[..., 0], base, mask, snr, **settings)
        return np.repeat(out[..., None], arr.shape[2], axis=2)
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        m = base
        if per_channel:
            mc = measure_moons(arr, is_grayscale, moons, plane=arr[..., c])
            # A channel too faint to measure well falls back to luminance.
            if mc is not None and mc.spectral_noise <= 4*max(base.spectral_noise, 1e-4):
                m = mc
        out[..., c] = correct_plane(arr[..., c], m, mask, snr, **settings)
    return out


def signal_maps(lum: np.ndarray, m: PsfMeasurement) -> tuple[np.ndarray, np.ndarray]:
    """Protect-background maps for the image: sky mask and per-pixel SNR."""
    mask, snr = _signal_maps(lum, m.sigma_minor, m.sigma_major)
    return mask.astype(np.float32), snr


def correct_plane(
    plane: np.ndarray, m: PsfMeasurement, mask: np.ndarray | None, snr: np.ndarray | None,
    *, target_size: float = 100.0, max_boost: float = 1.0,
    astigmatism: bool = False, coma: bool = True, trefoil: bool = True,
    frequency_snr: np.ndarray | None = None,
) -> np.ndarray:
    """Correct one image plane; ``mask``/``snr`` None leaves noise unprotected.

    Coma and trefoil only move light, so they are corrected wherever there is
    signal; the astigmatism boost is graded by what each pixel's
    signal-to-noise can take, and at each spatial frequency by the plane's
    signal-to-noise there (``frequency_snr``, from ``spectral_snr`` of the
    whole plane when this is a cut-out of it; measured here otherwise).
    """
    plane = np.asarray(plane, dtype=np.float32)
    pad = _plane_pad(plane.shape, m)
    padded = np.pad(plane.astype(np.float64), pad, mode="reflect")
    spectrum = rfft2(padded)
    shape = padded.shape
    if mask is None:
        freq_snr = None  # unprotected: the full correction
    elif frequency_snr is None:
        freq_snr = _spectral_snr(spectrum)
    else:
        freq_snr = _resample_spectrum(frequency_snr, shape)
    others = correction_otf(m, shape, target_size, max_boost, astigmatism=False,
                            coma=coma, trefoil=trefoil)

    def ellipse_for(boost: float) -> np.ndarray:
        return _limit_boost(_ellipse_otf(m, shape, target_size, boost), freq_snr)

    def run(boost: float, astig: bool) -> np.ndarray:
        k = others*ellipse_for(boost) if astig else others
        return irfft2(spectrum*k, shape)[pad:-pad, pad:-pad]

    if mask is None:
        return run(max_boost, astigmatism).astype(np.float32)
    if not astigmatism or max_boost <= 1.0:
        corrected = run(max_boost, False)
    else:
        snr_pad = np.pad(snr, pad, mode="reflect")
        g = max(float(max_boost), 1.0)
        levels = [lv for lv in _BOOST_LEVELS if lv < g] + [g]
        weight = _adaptive_level_weights(snr_pad, levels, ellipse_for, m.sigma_major)
        corrected = np.zeros(plane.shape, dtype=np.float64)
        for i, level in enumerate(levels):
            w = weight(i)[pad:-pad, pad:-pad]
            if w.any():
                corrected += w*run(level, level > 1.0)
    return (plane + mask*(corrected - plane)).astype(np.float32)


def moon_preview(
    lum: np.ndarray, m: PsfMeasurement, maps: tuple[np.ndarray, np.ndarray] | None,
    frequency_snr: np.ndarray | None = None, **settings,
) -> tuple[np.ndarray, np.ndarray]:
    """The best moon cut from the image before and after correction, exactly
    as the image is corrected (for the dialog thumbnails). ``frequency_snr``
    is ``spectral_snr`` of ``lum``, measured here when not given."""
    lum = np.asarray(lum, dtype=np.float32)
    if maps is not None and frequency_snr is None and settings.get("astigmatism"):
        frequency_snr = spectral_snr(lum, m)
    h, w = lum.shape
    r = m.radius
    reach = 3*r
    cx, cy = int(round(m.x)), int(round(m.y))
    x0, x1 = max(0, cx - reach), min(w, cx + reach + 1)
    y0, y1 = max(0, cy - reach), min(h, cy + reach + 1)
    crop = lum[y0:y1, x0:x1]
    mask = snr = None
    if maps is not None:
        mask, snr = maps[0][y0:y1, x0:x1], maps[1][y0:y1, x0:x1]
    corrected = correct_plane(crop, m, mask, snr, frequency_snr=frequency_snr, **settings)
    sy, sx = cy - y0, cx - x0
    window = (slice(max(0, sy - r), sy + r + 1), slice(max(0, sx - r), sx + r + 1))
    before, after = crop[window], corrected[window]
    sky = float(np.median(np.r_[before[0], before[-1], before[:, 0], before[:, -1]]))
    return before - sky, after - sky
