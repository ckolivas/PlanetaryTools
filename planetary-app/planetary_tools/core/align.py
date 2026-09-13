"""Subpixel translation and local rigid registration of planetary images."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import least_squares

from planetary_tools.core.colour import linear_luminance

_MAX_SHIFT_PX = 5  # RGB channel correction, in original pixels.


def _luma(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return linear_luminance(arr[..., :3])
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported image shape for alignment: {arr.shape}")


def _registration_structure(lum: np.ndarray) -> np.ndarray:
    """Match limbs, rings and belts without sky noise or broad brightness bias."""
    arr = np.asarray(lum, dtype=np.float64)
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError("Alignment requires non-empty, finite image planes.")
    smooth = gaussian_filter(arr, 1.0)
    return smooth - gaussian_filter(smooth, max(4.0, 0.02 * min(arr.shape)))


def _best_shift(reference: np.ndarray, target: np.ndarray, max_shift: int) -> tuple[int, int]:
    """Return the best integer seed within an original-pixel search radius."""
    ref = reference.astype(np.float64) - reference.mean()
    tgt = target.astype(np.float64) - target.mean()
    if not np.any(ref) or not np.any(tgt):
        return 0, 0
    corr = np.fft.ifft2(np.fft.fft2(ref) * np.conj(np.fft.fft2(tgt))).real
    h, w = ref.shape
    ys = np.arange(-min(max_shift, (h - 1) // 2), min(max_shift, (h - 1) // 2) + 1)
    xs = np.arange(-min(max_shift, (w - 1) // 2), min(max_shift, (w - 1) // 2) + 1)
    iy, ix = np.unravel_index(np.argmax(corr[np.ix_(ys % h, xs % w)]), (len(ys), len(xs)))
    return int(ys[iy]), int(xs[ix])


def _refine_alignment(
    reference: np.ndarray,
    target: np.ndarray,
    initial: tuple[float, float, float],
    *,
    max_angle: float = 0.0,
    max_shift: float | None = None,
) -> tuple[float, float, float, float]:
    """Refine a structure-image match at native resolution using normalised luma.

    Parameters and return value use CCW degrees and the shift of the rotated
    target in (dy, dx) order. A zero angle bound fixes rotation. Only estimation
    resamples here; callers apply the final transform to the original pixels.
    """
    angle, dy, dx = initial
    # Limit work to the useful part of the reference, retaining a sky margin.
    magnitude = np.abs(reference)
    peak = float(magnitude.max())
    if peak < 1e-12 or float(np.max(np.abs(target))) < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    yy, xx = np.nonzero(magnitude > 0.01 * peak)
    y0, y1 = max(0, int(yy.min()) - 8), min(reference.shape[0], int(yy.max()) + 9)
    x0, x1 = max(0, int(xx.min()) - 8), min(reference.shape[1], int(xx.max()) + 9)
    stride = max(1, int(np.ceil(np.sqrt((y1 - y0) * (x1 - x0) / 100_000))))
    y, x = np.mgrid[y0:y1:stride, x0:x1:stride].astype(np.float64)
    values = reference[y0:y1:stride, x0:x1:stride].ravel().copy()
    values -= values.mean()
    norm = float(np.sqrt(np.sum(values * values)))
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    values /= norm
    coefficients = spline_filter(target, order=3)
    cy, cx = (np.asarray(reference.shape) - 1) / 2.0
    rotate = max_angle > 0.0

    def residual(params: np.ndarray) -> np.ndarray:
        theta, sy, sx = params if rotate else (angle, *params)
        c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        yt, xt = y - cy - sy, x - cx - sx
        samples = map_coordinates(
            coefficients, [c * yt + s * xt + cy, -s * yt + c * xt + cx],
            order=3, prefilter=False, mode="constant", cval=0.0,
        ).ravel()
        samples -= samples.mean()
        samples /= max(float(np.sqrt(np.sum(samples * samples))), 1e-12)
        return samples - values

    start = np.array([angle, dy, dx] if rotate else [dy, dx], dtype=np.float64)
    lower, upper = start - 3.0, start + 3.0
    if rotate:
        lower[0] = max(-max_angle, angle - 0.25)
        upper[0] = min(max_angle, angle + 0.25)
    if max_shift is not None:
        lower[-2:] = np.maximum(lower[-2:], -max_shift)
        upper[-2:] = np.minimum(upper[-2:], max_shift)
    fit = least_squares(
        residual, start, bounds=(lower, upper),
        x_scale=[0.1, 1.0, 1.0] if rotate else 1.0,
        ftol=1e-7, xtol=1e-5, gtol=1e-8, max_nfev=30,
    )
    theta, sy, sx = fit.x if rotate else (angle, *fit.x)
    score = float(np.clip(np.sum(values * (values + fit.fun)), -1.0, 1.0))
    return float(theta), float(sy), float(sx), score


def align_channel(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align one channel to a reference with a single cubic subpixel shift."""
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("align_channel requires two single-channel image planes.")
    return align_to_reference(reference, target)


def align_to_reference(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correct up to five pixels of translation, keeping colour planes together."""
    ref = np.asarray(reference, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if ref.shape[:2] != tgt.shape[:2]:
        raise ValueError(
            f"align_to_reference requires matching shapes ({ref.shape} vs {tgt.shape})."
        )
    ref_s = _registration_structure(_luma(ref))
    tgt_s = _registration_structure(_luma(tgt))
    dy, dx = _best_shift(ref_s, tgt_s, _MAX_SHIFT_PX)
    _, dy, dx, _score = _refine_alignment(
        ref_s, tgt_s, (0.0, dy, dx), max_shift=_MAX_SHIFT_PX,
    )
    if abs(dy) < 1e-5 and abs(dx) < 1e-5:
        return tgt
    # Resample spatial planes independently, including alpha if present.
    if tgt.ndim == 2:
        return ndi_shift(tgt, (dy, dx), order=3, mode="constant", cval=0.0)
    return np.stack([
        ndi_shift(tgt[..., c], (dy, dx), order=3, mode="constant", cval=0.0)
        for c in range(tgt.shape[2])
    ], axis=-1)
