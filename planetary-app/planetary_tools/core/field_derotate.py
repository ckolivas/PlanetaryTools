"""Derotate/Align by luminance match to a reference image.

Rigid registration in the style of WaveSharp Align/Rotate: no astrometry.
Rotation is optional; shift-only matching skips the angle search. Angle and
shift both come from the planet’s luminance structure. Shift uses a
low-passed luma match so compact moons cannot pull the translation.
Optional subpixel lock uses the Align RGB 3× cross-correlation against the
chosen reference after the integer match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift

from planetary_tools.core.align import align_to_reference
from planetary_tools.core.colour import linear_luminance
from planetary_tools.core.rotate import (
    geometric_centre,
    paste_into_canvas,
    rotate_image,
)
from planetary_tools.core.scale import scale_image

ProgressCb = Callable[[int, int, str], None]

# Keep the short side near this many pixels for the angle sweep.
_SEARCH_SHORT = 256
_COARSE_STEP = 0.5
_FINE_SPAN = 1.0
_FINE_STEP = 0.05
_POLISH_SPAN = 0.1
_POLISH_STEP = 0.01
_SHIFT_SIGMA_FRAC = 0.03
_SHIFT_SIGMA_MIN = 6.0
_WEAK_SCORE = 0.15
_CORE_FRAC = 0.2


@dataclass(frozen=True)
class RigidMatch:
    angle_deg: float
    dy: float
    dx: float
    score: float
    status: str = "OK"


def luma(data: np.ndarray) -> np.ndarray:
    """BT.709 luminance, or the plane itself if already 2-D."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return linear_luminance(arr[..., :3])
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _wrap_180(angle: float) -> float:
    a = (float(angle) + 180.0) % 360.0 - 180.0
    if a <= -180.0:
        a += 360.0
    return a


def _search_downsample(arr: np.ndarray) -> np.ndarray:
    """Downsample for the angle sweep, keeping belts/rings resolvable."""
    h, w = arr.shape[:2]
    short = min(h, w)
    if short <= _SEARCH_SHORT:
        return np.asarray(arr, dtype=np.float32)
    factor = short / float(_SEARCH_SHORT)
    return scale_image(arr, max(1, int(round(w / factor))), max(1, int(round(h / factor))))


def _structure(lum: np.ndarray) -> np.ndarray:
    """High-pass luma so a round disk does not drown belts/rings in NCC."""
    sigma = max(2.0, 0.04 * min(lum.shape[:2]))
    return np.asarray(lum, dtype=np.float32) - gaussian_filter(lum, sigma=sigma)


def phase_correlation_shift(
    reference: np.ndarray, target: np.ndarray
) -> tuple[int, int, float]:
    """Zero-mean FFT cross-correlation. Returns ``(dy, dx, ncc_peak)``.

    ``(dy, dx)`` is the integer shift to apply to ``target`` to match
    ``reference`` (scipy.ndimage.shift convention). Any magnitude; wraps at
    half the frame.
    """
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError("phase_correlation_shift requires matching shapes.")
    a = ref - ref.mean()
    b = tgt - tgt.mean()
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na < 1e-12 or nb < 1e-12:
        return 0, 0, 0.0
    corr = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    corr /= na * nb
    peak_idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = int(peak_idx[0])
    dx = int(peak_idx[1])
    h, w = corr.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return dy, dx, float(corr[peak_idx])


def _rotate_luma(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Same-size rotation for the search loop (CCW, matches Pillow)."""
    if abs(float(angle_deg)) < 1e-12:
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(
        ndi_rotate(
            np.asarray(arr, dtype=np.float32),
            float(angle_deg),
            reshape=False,
            order=1,
            cval=0.0,
        ),
        dtype=np.float32,
    )


def _best_angle(
    ref: np.ndarray,
    tgt: np.ndarray,
    angles: np.ndarray,
) -> tuple[float, float, int, int]:
    """Return ``(angle, score, dy, dx)`` maximising luma NCC after rotation."""
    best_score = -np.inf
    best = (0.0, 0.0, 0, 0)
    second: tuple[float, float] | None = None
    for theta in angles:
        rotated = _rotate_luma(tgt, float(theta))
        dy, dx, score = phase_correlation_shift(ref, rotated)
        if score > best_score:
            if math.isfinite(best_score):
                second = (best[0], best_score)
            best_score = score
            best = (float(theta), score, dy, dx)
        elif second is None or score > second[1]:
            second = (float(theta), score)
    theta, score, dy, dx = best
    # Prefer the smaller |angle| if a ~180° rival is essentially as good.
    if second is not None:
        t2, s2 = second
        if abs(_wrap_180(t2 - theta)) > 150.0 and s2 > 0.95 * score:
            if abs(_wrap_180(t2)) < abs(_wrap_180(theta)):
                rotated = _rotate_luma(tgt, t2)
                dy, dx, score = phase_correlation_shift(ref, rotated)
                theta = t2
    return float(theta), float(score), int(dy), int(dx)


def _search_angle(
    ref: np.ndarray, tgt: np.ndarray, max_angle: float
) -> tuple[float, float, bool]:
    max_angle = abs(float(max_angle))
    coarse = np.arange(-max_angle, max_angle + 0.5 * _COARSE_STEP, _COARSE_STEP)
    theta, score, _dy, _dx = _best_angle(ref, tgt, coarse)
    fine = np.arange(theta - _FINE_SPAN, theta + _FINE_SPAN + 0.5 * _FINE_STEP, _FINE_STEP)
    theta, score, _dy, _dx = _best_angle(ref, tgt, fine)
    polish = np.arange(
        theta - _POLISH_SPAN, theta + _POLISH_SPAN + 0.5 * _POLISH_STEP, _POLISH_STEP
    )
    theta, score, _dy, _dx = _best_angle(ref, tgt, polish)
    theta = _wrap_180(theta)
    rival = _wrap_180(theta + 180.0)
    if abs(rival) < abs(theta):
        rot_r = _rotate_luma(tgt, rival)
        _dyr, _dxr, score_r = phase_correlation_shift(ref, rot_r)
        if score_r > 0.95 * score:
            theta, score = rival, score_r
    on_limit = abs(theta) >= max_angle - _COARSE_STEP
    return theta, score, on_limit


def _shift_sigma(shape: tuple[int, ...]) -> float:
    return max(_SHIFT_SIGMA_MIN, _SHIFT_SIGMA_FRAC * min(shape[0], shape[1]))


def _luma_centroid(lum: np.ndarray) -> tuple[float, float]:
    peak = float(lum.max())
    if peak <= 0.0:
        return geometric_centre(lum.shape)
    mask = lum >= _CORE_FRAC * peak
    if not bool(mask.any()):
        return geometric_centre(lum.shape)
    ys, xs = np.nonzero(mask)
    w = lum[ys, xs].astype(np.float64)
    wsum = float(w.sum())
    if wsum <= 0.0:
        return geometric_centre(lum.shape)
    return float(np.dot(w, xs) / wsum), float(np.dot(w, ys) / wsum)


def centre_pad_to(data: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Centre ``data`` on a black canvas of ``canvas_w``×``canvas_h``."""
    arr = np.asarray(data, dtype=np.float32)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if w == canvas_w and h == canvas_h:
        return arr
    if w > canvas_w or h > canvas_h:
        raise ValueError(
            f"Frame {w}×{h} is larger than canvas {canvas_w}×{canvas_h}."
        )
    return paste_into_canvas(arr, canvas_w, canvas_h, geometric_centre(arr.shape))


def pad_to_common(images: list[np.ndarray]) -> list[np.ndarray]:
    """Centre-pad every image onto a canvas of max width × max height."""
    if not images:
        return []
    canvas_w = max(int(im.shape[1]) for im in images)
    canvas_h = max(int(im.shape[0]) for im in images)
    return [centre_pad_to(im, canvas_w, canvas_h) for im in images]


def estimate_rigid(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    max_angle: float = 45.0,
    rotate: bool = True,
) -> RigidMatch:
    """Match ``target`` to ``reference`` by best luminance.

    When ``rotate`` is true, search a rigid rotation + translation.
    ``angle_deg`` is CCW (Pillow). ``(dy, dx)`` shifts the *rotated* target
    onto the reference (scipy.ndimage.shift convention, original-pixel units).
    When ``rotate`` is false, only the low-pass translation is estimated.
    Frames of different size are centre-padded to a shared canvas first.
    """
    reference, target = pad_to_common(
        [np.asarray(reference, dtype=np.float32), np.asarray(target, dtype=np.float32)]
    )
    ref_l = luma(reference)
    tgt_l = luma(target)

    sigma = _shift_sigma(ref_l.shape)
    if not rotate:
        dy, dx, shift_score = phase_correlation_shift(
            gaussian_filter(ref_l, sigma=sigma),
            gaussian_filter(tgt_l, sigma=sigma),
        )
        status = "OK" if shift_score >= _WEAK_SCORE else "Weak match"
        return RigidMatch(
            angle_deg=0.0,
            dy=float(dy),
            dx=float(dx),
            score=float(shift_score),
            status=status,
        )

    ref_s = _structure(_search_downsample(ref_l))
    tgt_s = _structure(_search_downsample(tgt_l))
    theta, score, on_limit = _search_angle(ref_s, tgt_s, max_angle)

    status = "OK"
    if score < _WEAK_SCORE:
        status = "Weak match"

    rotated = rotate_image(target, theta, expand=False, crop_to_original=False)
    rot_l = luma(rotated)
    # Planet-only shift (low-pass) — compact moons must not drive this.
    dy, dx, shift_score = phase_correlation_shift(
        gaussian_filter(ref_l, sigma=sigma),
        gaussian_filter(rot_l, sigma=sigma),
    )
    score = max(score, float(shift_score))
    if on_limit and status == "OK":
        status = "Hit search limit"

    return RigidMatch(
        angle_deg=float(theta),
        dy=float(dy),
        dx=float(dx),
        score=float(score),
        status=status,
    )


def apply_rigid(
    data: np.ndarray,
    match: RigidMatch,
) -> np.ndarray:
    """Rotate (expand) then shift. Output may be larger than the input."""
    rotated = rotate_image(data, match.angle_deg, expand=True, crop_to_original=False)
    if abs(match.dy) < 1e-6 and abs(match.dx) < 1e-6:
        return rotated
    if rotated.ndim == 2:
        shift = (match.dy, match.dx)
    else:
        shift = (match.dy, match.dx, 0.0)
    return np.asarray(
        ndi_shift(rotated, shift=shift, order=1, mode="constant", cval=0.0),
        dtype=np.float32,
    )


def reference_pivot(data: np.ndarray) -> tuple[float, float]:
    """Planet-dominated luma centroid — used to centre the common canvas."""
    return _luma_centroid(luma(data))


@dataclass
class DerotateFrameResult:
    path: Path
    output_path: Path
    match: RigidMatch


@dataclass
class DerotateSetResult:
    processed: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    canvas_size: tuple[int, int] | None = None
    frames: list[DerotateFrameResult] = field(default_factory=list)


def _is_identity_match(match: RigidMatch) -> bool:
    return (
        abs(match.angle_deg) < 1e-9
        and abs(match.dx) < 1e-9
        and abs(match.dy) < 1e-9
    )


def _reference_index(
    items: list[tuple[Path, np.ndarray, RigidMatch]],
    ref_index: int | None,
) -> int:
    if ref_index is not None and 0 <= ref_index < len(items):
        return ref_index
    for i, (_path, _data, match) in enumerate(items):
        if _is_identity_match(match):
            return i
    return 0


def derotate_set(
    items: list[tuple[Path, np.ndarray, RigidMatch]],
    output_dir: Path,
    *,
    suffix: str = "_derot",
    bit_depth: int = 32,
    subpixel: bool = False,
    ref_index: int | None = None,
    on_progress: ProgressCb | None = None,
) -> DerotateSetResult:
    """Apply matches, paste onto a common centred canvas, and save.

    ``items`` is ``(path, pixels, match)``. The first item whose match is the
    identity (or the first item) supplies the canvas-centre pivot.

    When ``subpixel`` is true, each non-reference canvas is locked to the
    reference with the Align RGB 3× cross-correlation after the integer
    rigid match.
    """
    from planetary_tools.core.document import ImageDocument
    from planetary_tools.io.loader import save_image

    result = DerotateSetResult()
    if not items:
        return result

    padded = pad_to_common([data for _path, data, _match in items])
    items = [
        (path, arr, match) for (path, _old, match), arr in zip(items, padded)
    ]

    applied: list[tuple[Path, np.ndarray, RigidMatch]] = []
    total = len(items)
    for i, (path, data, match) in enumerate(items):
        if on_progress:
            on_progress(i, total, f"Applying {path.name}")
        try:
            applied.append((path, apply_rigid(data, match), match))
        except Exception as exc:
            result.failed.append((str(path), str(exc)))

    if not applied:
        return result

    # Canvas: large enough that each pivot can sit at the centre.
    pivots: list[tuple[float, float]] = []
    for _path, arr, match in applied:
        if abs(match.angle_deg) < 1e-9 and abs(match.dx) < 1e-9 and abs(match.dy) < 1e-9:
            pivots.append(reference_pivot(arr))
        else:
            # After rot+shift the planet should sit where the reference planet
            # sits in a same-size frame; on an expanded frame it is near centre
            # plus the applied shift, which is already in the pixels. Use luma
            # centroid of the aligned frame.
            pivots.append(reference_pivot(arr))

    max_left = max(px for px, _py in pivots)
    max_right = max(arr.shape[1] - 1 - px for (_p, arr, _m), (px, _py) in zip(applied, pivots))
    max_top = max(py for _px, py in pivots)
    max_bottom = max(arr.shape[0] - 1 - py for (_p, arr, _m), (_px, py) in zip(applied, pivots))
    canvas_w = int(math.ceil(max_left + max_right + 1))
    canvas_h = int(math.ceil(max_top + max_bottom + 1))
    canvas_w = max(canvas_w, max(arr.shape[1] for _p, arr, _m in applied))
    canvas_h = max(canvas_h, max(arr.shape[0] for _p, arr, _m in applied))
    result.canvas_size = (canvas_w, canvas_h)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canvases: list[tuple[Path, np.ndarray, RigidMatch]] = []
    for (path, arr, match), pivot in zip(applied, pivots):
        canvases.append((path, paste_into_canvas(arr, canvas_w, canvas_h, pivot), match))

    if subpixel and len(canvases) >= 2:
        ref_path = items[_reference_index(items, ref_index)][0]
        ref_i = next((i for i, (path, _c, _m) in enumerate(canvases) if path == ref_path), 0)
        ref_canvas = canvases[ref_i][1]
        for i, (path, canvas, match) in enumerate(canvases):
            if i == ref_i:
                continue
            if on_progress:
                on_progress(i, total, f"Subpixel {path.name}")
            try:
                canvases[i] = (path, align_to_reference(ref_canvas, canvas), match)
            except Exception as exc:
                result.failed.append((str(path), str(exc)))

    failed_paths = {p for p, _exc in result.failed}
    for i, (path, canvas, match) in enumerate(canvases):
        if str(path) in failed_paths:
            continue
        if on_progress:
            on_progress(i, total, f"Saving {path.name}")
        try:
            out_path = output_dir / f"{path.stem}{suffix}{path.suffix}"
            is_gray = canvas.ndim == 2
            doc = ImageDocument(
                data=np.asarray(canvas, dtype=np.float32),
                path=out_path,
                is_grayscale=is_gray,
                modified=True,
                storage_bits=bit_depth,
            )
            if canvas.ndim == 3:
                doc.is_grayscale = False
            save_image(doc, out_path, bit_depth=bit_depth)
            result.processed += 1
            result.frames.append(
                DerotateFrameResult(path=path, output_path=out_path, match=match)
            )
        except Exception as exc:
            result.failed.append((str(path), str(exc)))

    return result


IDENTITY_MATCH = RigidMatch(angle_deg=0.0, dy=0.0, dx=0.0, score=1.0, status="Reference")
