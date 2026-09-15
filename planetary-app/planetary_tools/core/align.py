"""Subpixel translation and local rigid registration of planetary images."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    distance_transform_edt, gaussian_filter, gaussian_filter1d, map_coordinates,
    spline_filter,
)
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import least_squares

from planetary_tools.core.colour import linear_luminance

_MAX_SHIFT_PX = 5  # Local translation correction, in original pixels.


def _luma(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return linear_luminance(arr[..., :3])
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported image shape for alignment: {arr.shape}")


def _registration_structure(
    lum: np.ndarray, *, smoothing: float = 1.0, background: float | None = None,
) -> np.ndarray:
    """Match limbs, rings and belts without sky noise or broad brightness bias."""
    arr = np.asarray(lum, dtype=np.float64)
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError("Alignment requires non-empty, finite image planes.")
    smooth = gaussian_filter(arr, smoothing)
    if background is None:
        background = max(4.0, 0.02 * min(arr.shape))
    return smooth - gaussian_filter(smooth, background)


def _seeing_structure_pair(
    reference: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare planetary outlines on a shared scale, suppressing seeing detail.

    Size the band from the bright subject rather than its surrounding canvas.
    Percentile extents keep isolated moons/stars from setting that scale. Both
    images use the same band, including when the softer frame is the reference.
    This filtering is for estimation only, never for the exported pixels.
    """
    extents = []
    for plane in (reference, target):
        arr = np.asarray(plane, dtype=np.float64)
        if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
            raise ValueError("Alignment requires non-empty, finite image planes.")
        smooth = gaussian_filter(arr, 2.0)
        floor = float(np.percentile(smooth, 10))
        peak = float(smooth.max()) - floor
        if peak > 1e-12:
            yy, xx = np.nonzero(smooth > floor + 0.2 * peak)
            extents.append(min(np.diff(np.percentile(yy, [1, 99]))[0],
                               np.diff(np.percentile(xx, [1, 99]))[0]))
    sigma = max(2.0, 0.025 * max(extents, default=0.0))
    return tuple(_registration_structure(p, smoothing=sigma, background=4 * sigma)
                 for p in (reference, target))


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


def _vertical_limb_correction(reference: np.ndarray, target: np.ndarray) -> float:
    """Refine disk height after rigid alignment without following bright rings.

    A central strip through the largest inscribed disk avoids the ring tips and
    isolated moons. Match curvature across both limbs with one displacement and
    relative blur. Curvature suppresses diffuse glow and brightness ramps, while
    retaining the sharper disk edge. Normalising the two limbs together keeps a
    faint, defocused halo from getting the same weight as a resolved edge.
    Fall back to the rigid fit if the disk is incomplete or poorly constrained.
    """
    disks = []
    for plane in (reference, target):
        smooth = gaussian_filter(np.asarray(plane, dtype=np.float64), 2.0)
        floor = float(np.percentile(smooth, 10))
        peak = float(smooth.max()) - floor
        if peak < 1e-12:
            return 0.0
        distance = distance_transform_edt(smooth > floor + .06 * peak)
        cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
        disks.append((float(distance[cy, cx]), cy, cx))
    # Glow can expand this low-threshold mask and move its centre. Prefer the
    # more compact disk so a glowing reference does not put the limb windows
    # outside the sharper target's edges. Both are already in reference space.
    radius, cy, cx = min(disks)
    width = int(.4 * radius)
    h, w = smooth.shape
    # Both complete limb windows and their shift margin must be available.
    if (radius < 8 or cy - 1.4 * radius < 4 or cy + 1.4 * radius >= h - 4
            or cx - width < 0 or cx + width >= w):
        return 0.0
    profiles = [gaussian_filter1d(
        np.asarray(p[:, cx-width:cx+width+1], dtype=np.float64).mean(axis=1), 1.5,
    ) for p in (reference, target)]
    windows = []
    for side in (-1, 1):
        rows = np.arange(int(cy + side * radius - .4 * radius),
                         int(cy + side * radius + .4 * radius))
        # Require a rising upper limb and falling lower limb in both images.
        if any(-side * (p[rows[-1]] - p[rows[0]]) < .05 * np.ptp(p)
               or np.ptp(p[rows]) < 1e-12 for p in profiles):
            return 0.0

        windows.append(rows)
    rows = np.concatenate(windows)

    def residual(params):
        dy, variance = params
        # Signed variance allows either reference direction. Keep kernel support
        # fixed across the allowed blur range so derivatives stay continuous.
        a = gaussian_filter1d(profiles[0], np.sqrt(1 + max(variance, 0)),
                              order=2, radius=33)
        b = gaussian_filter1d(profiles[1], np.sqrt(1 + max(-variance, 0)),
                              order=2, radius=33)
        a = a[rows].copy()
        b = map_coordinates(b, [rows - dy], order=3, mode="nearest")
        a -= a.mean()
        b -= b.mean()
        a /= max(float(np.linalg.norm(a)), 1e-12)
        b /= max(float(np.linalg.norm(b)), 1e-12)
        return b - a

    fit = least_squares(residual, [0., 0.], bounds=([-3., -64.], [3., 64.]),
                        diff_step=1e-3, gtol=1e-10, max_nfev=60)
    correlation = 1 - float(np.dot(fit.fun, fit.fun)) / 2
    if (not fit.success or correlation < .9 or abs(fit.x[0]) >= 2.99
            or abs(fit.x[1]) >= 63.99 or np.linalg.norm(fit.jac[:, 0]) < 1e-3):
        return 0.0
    return float(fit.x[0])


def _refine_alignment(
    reference: np.ndarray,
    target: np.ndarray,
    initial: tuple[float, float, float],
    *,
    max_angle: float = 0.0,
    max_shift: float | None = None,
    angle_span: float = 0.25,
) -> tuple[float, float, float, float]:
    """Refine a structure-image match at native resolution using normalised luma.

    Parameters and return value use CCW degrees and the shift of the rotated
    target in (dy, dx) order. A zero angle bound fixes rotation. Only estimation
    resamples here; callers apply the final transform to the original pixels.
    ``angle_span`` bounds local refinement around the coarse rotation seed.
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
    # Include the zero sky in the spline prefilter as well as the sampler;
    # otherwise edge coefficients do not reconstruct an identity match.
    padding = 12
    coefficients = spline_filter(np.pad(target, padding), order=3)
    cy, cx = (np.asarray(reference.shape) - 1) / 2.0
    rotate = max_angle > 0.0

    def residual(params: np.ndarray) -> np.ndarray:
        theta, sy, sx = params if rotate else (angle, *params)
        c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        yt, xt = y - cy - sy, x - cx - sx
        samples = map_coordinates(
            coefficients, [c * yt + s * xt + cy + padding,
                           -s * yt + c * xt + cx + padding],
            # Ordinary constant mode jumps to zero just outside the last pixel,
            # which can trap numerical derivatives at integer shifts/angles.
            order=3, prefilter=False, mode="grid-constant", cval=0.0,
        ).ravel()
        samples -= samples.mean()
        samples /= max(float(np.sqrt(np.sum(samples * samples))), 1e-12)
        return samples - values

    start = np.array([angle, dy, dx] if rotate else [dy, dx], dtype=np.float64)
    lower, upper = start - 3.0, start + 3.0
    if rotate:
        lower[0] = max(-max_angle, angle - angle_span)
        upper[0] = min(max_angle, angle + angle_span)
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
    """Align a channel using Derotate/Align's masked, shift-only estimator.

    Estimate on copies with the default 25% perceptual brightness mask; render
    the original target once, keeping the reference channel's canvas size.
    """
    # field_derotate uses the shared refinement primitives in this module.
    from planetary_tools.core.field_derotate import (
        apply_rigid, estimate_rigid, mask_alignment_background,
    )

    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("align_channel requires two single-channel image planes.")
    if reference.shape != target.shape:
        raise ValueError(
            f"align_channel requires matching shapes ({reference.shape} vs {target.shape})."
        )
    match = estimate_rigid(
        mask_alignment_background(reference), mask_alignment_background(target),
        rotate=False,
    )
    return apply_rigid(target, match, expand=False)


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
