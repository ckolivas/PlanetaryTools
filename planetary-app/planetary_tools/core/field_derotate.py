"""Derotate/Align by luminance match to a reference image.

Rigid registration in the style of WaveSharp Align/Rotate: no astrometry.
Rotation is optional; shift-only matching skips the angle search. Angle and
shift are refined together against native-resolution limbs, rings and belts.
All output frames share reference coordinates and are resampled only once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, rotate as ndi_rotate

from planetary_tools.core.align import _refine_alignment, _registration_structure
from planetary_tools.core.colour import linear_luminance
from planetary_tools.core.rotate import (
    geometric_centre,
    paste_into_canvas,
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
    max_angle = min(180.0, abs(float(max_angle)))
    coarse = np.unique(np.clip(
        np.r_[np.arange(-max_angle, max_angle, _COARSE_STEP), 0.0, max_angle],
        -max_angle, max_angle,
    ))
    theta, score, _dy, _dx = _best_angle(ref, tgt, coarse)
    fine = np.arange(
        max(-max_angle, theta - _FINE_SPAN),
        min(max_angle, theta + _FINE_SPAN) + 0.5 * _FINE_STEP, _FINE_STEP,
    )
    fine = np.clip(fine, -max_angle, max_angle)
    theta, score, _dy, _dx = _best_angle(ref, tgt, fine)
    polish = np.arange(
        max(-max_angle, theta - _POLISH_SPAN),
        min(max_angle, theta + _POLISH_SPAN) + 0.5 * _POLISH_STEP, _POLISH_STEP
    )
    theta, score, _dy, _dx = _best_angle(ref, tgt, np.clip(polish, -max_angle, max_angle))
    theta = _wrap_180(theta)
    rival = _wrap_180(theta + 180.0)
    if abs(rival) <= max_angle and abs(rival) < abs(theta):
        rot_r = _rotate_luma(tgt, rival)
        _dyr, _dxr, score_r = phase_correlation_shift(ref, rot_r)
        if score_r > 0.95 * score:
            theta, score = rival, score_r
    on_limit = abs(theta) >= max_angle - _COARSE_STEP
    return theta, score, on_limit


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
    Translation is fractional; when ``rotate`` is false its angle stays zero.
    Frames of different size are centre-padded to a shared canvas first.
    """
    reference, target = pad_to_common(
        [np.asarray(reference, dtype=np.float32), np.asarray(target, dtype=np.float32)]
    )
    ref_l = luma(reference)
    tgt_l = luma(target)

    ref_s = _registration_structure(ref_l)
    tgt_s = _registration_structure(tgt_l)
    if not np.any(ref_s) or not np.any(tgt_s):
        return RigidMatch(0.0, 0.0, 0.0, 0.0, "Weak match")
    if not math.isfinite(max_angle):
        raise ValueError("Maximum alignment angle must be finite.")
    max_angle = min(180.0, abs(float(max_angle)))
    rotate = rotate and max_angle > 0.0
    theta = 0.0
    if rotate:
        theta, _score, _on_limit = _search_angle(
            _structure(_search_downsample(ref_l)),
            _structure(_search_downsample(tgt_l)), max_angle,
        )
    dy, dx, _score = phase_correlation_shift(ref_s, _rotate_luma(tgt_s, theta))
    theta, dy, dx, score = _refine_alignment(
        ref_s, tgt_s, (theta, dy, dx), max_angle=max_angle if rotate else 0.0,
    )
    on_limit = rotate and abs(theta) >= max_angle - 0.01
    status = "Weak match" if score < _WEAK_SCORE else "Hit search limit" if on_limit else "OK"
    return RigidMatch(theta, dy, dx, score, status)


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array([[c, -s], [s, c]])


def _transformed_bounds(data: np.ndarray, match: RigidMatch) -> np.ndarray:
    h, w = data.shape[:2]
    centre = (np.array([h, w]) - 1) / 2.0
    corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]])
    return (corners - centre) @ _rotation_matrix(match.angle_deg).T + centre + [match.dy, match.dx]


def _render_rigid(
    data: np.ndarray, match: RigidMatch, shape: tuple[int, int], origin: np.ndarray,
) -> np.ndarray:
    """Sample directly from the original into one shared reference canvas."""
    if abs(match.angle_deg) < 1e-9 and all(
        abs(v - round(v)) < 1e-9 for v in (match.dy, match.dx)
    ):
        # Keep the reference and integer translations bit-exact.
        cx, cy = geometric_centre(data.shape)
        return paste_into_canvas(
            data, shape[1], shape[0], (cx, cy),
            (cx + round(match.dx) - origin[1], cy + round(match.dy) - origin[0]),
        )
    centre = (np.array(data.shape[:2]) - 1) / 2.0
    inverse = _rotation_matrix(match.angle_deg).T
    offset = centre + inverse @ (origin - centre - [match.dy, match.dx])

    def render(plane: np.ndarray) -> np.ndarray:
        return affine_transform(
            plane, inverse, offset=offset, output_shape=shape,
            order=3, mode="constant", cval=0.0, output=np.float32,
        )

    if data.ndim == 2:
        return render(data)
    return np.stack([render(data[..., c]) for c in range(data.shape[2])], axis=-1)


def apply_rigid(data: np.ndarray, match: RigidMatch) -> np.ndarray:
    """Apply rotation and translation once, expanding to retain the full frame."""
    arr = np.asarray(data, dtype=np.float32)
    bounds = _transformed_bounds(arr, match)
    origin = np.floor(np.minimum(bounds.min(axis=0), [0, 0]))
    end = np.ceil(np.maximum(bounds.max(axis=0), np.array(arr.shape[:2]) - 1))
    shape = tuple((end - origin + 1).astype(int))
    return _render_rigid(arr, match, shape, origin)


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
    """Save each original through one transform onto a shared reference canvas.

    ``subpixel=False`` rounds translations to whole pixels. The selected
    reference determines the common centre; other frames are never recentred
    independently by brightness, which would invalidate their measured shifts.
    """
    from planetary_tools.core.document import ImageDocument
    from planetary_tools.io.loader import save_image

    result = DerotateSetResult()
    if not items:
        return result
    padded = pad_to_common([data for _path, data, _match in items])
    ref_path = items[_reference_index(items, ref_index)][0]
    valid: list[tuple[Path, np.ndarray, RigidMatch]] = []
    bounds: list[np.ndarray] = []
    for (path, _old, match), data in zip(items, padded):
        try:
            if not subpixel:
                match = replace(match, dy=round(match.dy), dx=round(match.dx))
            corners = _transformed_bounds(data, match)
            if not np.isfinite(corners).all():
                raise ValueError("Alignment transform must be finite.")
            valid.append((path, data, match))
            bounds.append(corners)
        except Exception as exc:
            result.failed.append((str(path), str(exc)))
    if not valid:
        return result
    ref_i = next((i for i, (path, _data, _match) in enumerate(valid) if path == ref_path), 0)
    ref_data, ref_match = valid[ref_i][1:]
    pivot = np.array(reference_pivot(ref_data)[::-1])
    centre = (np.array(ref_data.shape[:2]) - 1) / 2.0
    pivot = (
        _rotation_matrix(ref_match.angle_deg) @ (pivot - centre)
        + centre + [ref_match.dy, ref_match.dx]
    )
    corners = np.concatenate(bounds)
    # A single integer canvas origin preserves the reference's original pixels,
    # and prevents half-pixel offsets from differing expanded-frame parities.
    anchor = np.floor(pivot)
    radius = np.ceil(np.maximum(anchor - corners.min(axis=0), corners.max(axis=0) - anchor))
    origin = anchor - radius
    canvas_h, canvas_w = (2 * radius + 1).astype(int)
    result.canvas_size = (int(canvas_w), int(canvas_h))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (path, data, match) in enumerate(valid):
        if on_progress:
            on_progress(i, len(valid), f"Aligning and saving {path.name}")
        try:
            canvas = _render_rigid(data, match, (int(canvas_h), int(canvas_w)), origin)
            out_path = output_dir / f"{path.stem}{suffix}{path.suffix}"
            doc = ImageDocument(
                data=canvas, path=out_path, is_grayscale=canvas.ndim == 2,
                modified=True, storage_bits=bit_depth,
            )
            save_image(doc, out_path, bit_depth=bit_depth)
            result.processed += 1
            result.frames.append(DerotateFrameResult(path=path, output_path=out_path, match=match))
        except Exception as exc:
            result.failed.append((str(path), str(exc)))
    return result


IDENTITY_MATCH = RigidMatch(angle_deg=0.0, dy=0.0, dx=0.0, score=1.0, status="Reference")
