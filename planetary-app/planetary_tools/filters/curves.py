"""GIMP-style Value/RGB/alpha curves on linear-light image data.

Smooth sampling follows app/core/gimpcurve.c (gimp_curve_calculate/plot) in
GIMP, GPL-3.0-or-later. Mapping order follows gimpcurve-map.c: RGB first,
Value second; alpha is independent. GIMP's default curve has 256 samples.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from planetary_tools.core.colour import linear_to_srgb, srgb_to_linear

CHANNELS = ("Value", "Red", "Green", "Blue", "Alpha")
N_SAMPLES = 256


def identity_curve() -> dict[str, Any]:
    return {"mode": "smooth", "points": [[0.0, 0.0, "smooth"], [1.0, 1.0, "smooth"]]}


def default_curves_params() -> dict[str, Any]:
    return {"trc": "perceptual", "channels": {ch: identity_curve() for ch in CHANNELS}}


def normalize_curves_params(params: dict[str, Any]) -> dict[str, Any]:
    result = default_curves_params()
    trc = params.get("trc", "perceptual")
    if trc not in ("perceptual", "linear"):
        raise ValueError("Curves mode must be perceptual or linear.")
    result["trc"] = trc
    channels = params.get("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("Curves channels must be a dictionary.")
    for ch in CHANNELS:
        curve = channels.get(ch, identity_curve())
        if not isinstance(curve, dict):
            raise ValueError("Each curve must be a dictionary of points or samples.")
        mode = curve.get("mode", "smooth")
        if mode == "freehand":
            samples = np.asarray(curve.get("samples", []), dtype=np.float64)
            if samples.shape != (N_SAMPLES,) or not np.isfinite(samples).all():
                raise ValueError("Freehand curves require 256 finite samples.")
            if np.any((samples < 0) | (samples > 1)):
                raise ValueError("Curve samples must be between 0 and 1.")
            result["channels"][ch] = {"mode": mode, "samples": samples.tolist()}
        elif mode == "smooth":
            points = []
            for point in curve.get("points", identity_curve()["points"]):
                x, y = float(point[0]), float(point[1])
                kind = point[2] if len(point) > 2 else "smooth"
                if not (0 <= x <= 1 and 0 <= y <= 1) or kind not in ("smooth", "corner"):
                    raise ValueError("Curve points require input/output in [0, 1] and a smooth/corner type.")
                points.append([x, y, kind])
            points.sort(key=lambda point: point[0])
            if len(points) < 2 or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
                raise ValueError("Smooth curves need at least two points with distinct inputs.")
            result["channels"][ch] = {"mode": mode, "points": points}
        else:
            raise ValueError("Curve type must be smooth or freehand.")
    return result


def curve_samples(curve: dict[str, Any]) -> np.ndarray:
    """Build GIMP's sampled cubic Bézier curve, including corner tangents."""
    if curve["mode"] == "freehand":
        return np.array(curve["samples"], dtype=np.float64)
    points = curve["points"]
    samples = np.linspace(0.0, 1.0, N_SAMPLES)
    last = N_SAMPLES - 1

    def rounded(x: float) -> int:
        return int(np.floor(x * last + 0.5))

    samples[:rounded(points[0][0])] = points[0][1]
    samples[rounded(points[-1][0]):] = points[-1][1]
    for i in range(len(points) - 1):
        p1, p2, p3, p4 = max(i-1, 0), i, i+1, min(i+2, len(points)-1)
        if points[p2][2] == "corner":
            p1 = p2
        if points[p3][2] == "corner":
            p4 = p3
        x0, y0, _ = points[p2]
        x3, y3, _ = points[p3]
        dx, dy = x3 - x0, y3 - y0
        if dx <= 1e-6:
            samples[rounded(x0)] = y3
            continue
        if p1 == p2 and p3 == p4:
            y1, y2 = y0 + dy/3, y0 + 2*dy/3
        elif p1 == p2:
            slope = (points[p4][1] - y0) / (points[p4][0] - x0)
            y2 = y3 - slope*dx/3
            y1 = y0 + (y2-y0)/2
        elif p3 == p4:
            slope = (y3-points[p1][1]) / (x3-points[p1][0])
            y1 = y0 + slope*dx/3
            y2 = y3 + (y1-y3)/2
        else:
            y1 = y0 + (y3-points[p1][1]) / (x3-points[p1][0]) * dx/3
            y2 = y3 - (points[p4][1]-y0) / (points[p4][0]-x0) * dx/3
        offsets = np.arange(rounded(dx) + 1)
        indices = offsets + rounded(x0)
        keep = indices < N_SAMPLES
        t = offsets[keep] / dx / last
        y = y0*(1-t)**3 + 3*y1*(1-t)**2*t + 3*y2*(1-t)*t*t + y3*t**3
        samples[indices[keep]] = np.clip(y, 0, 1)
    for x, y, _kind in points:
        samples[rounded(x)] = y
    return samples


def is_identity(curve: dict[str, Any]) -> bool:
    if curve["mode"] == "smooth":
        return curve["points"] == identity_curve()["points"]
    return np.array_equal(curve["samples"], np.linspace(0, 1, N_SAMPLES))


def map_samples(values: np.ndarray, samples: np.ndarray) -> np.ndarray:
    # GIMP maps NaN and -inf to the first sample, +inf to the last.
    values = np.nan_to_num(values, nan=0.0, neginf=0.0, posinf=1.0)
    return np.interp(values, np.linspace(0, 1, len(samples)), samples)


def apply_curves(data: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply curves in the selected TRC; return linear float32, preserving shape."""
    params = normalize_curves_params(params)
    curves = params["channels"]
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] in (1, 3, 4)):
        raise ValueError("Curves requires a greyscale, RGB or RGBA image.")
    result = arr.copy()
    active = {ch: curve_samples(curve) for ch, curve in curves.items() if not is_identity(curve)}
    perceptual = params["trc"] == "perceptual"
    gray = arr.ndim == 2 or arr.shape[-1] == 1
    for index, ch in enumerate(("Value",) if gray else ("Red", "Green", "Blue")):
        if ch not in active and "Value" not in active:
            continue
        source = arr if arr.ndim == 2 else arr[..., index]
        mapped = linear_to_srgb(source, clamp=False) if perceptual else source
        if not gray and ch in active:
            mapped = map_samples(mapped, active[ch])
        if "Value" in active:
            mapped = map_samples(mapped, active["Value"])
        mapped = srgb_to_linear(mapped) if perceptual else mapped
        if arr.ndim == 2:
            result[:] = mapped
        else:
            result[..., index] = mapped
    if arr.ndim == 3 and arr.shape[-1] == 4 and "Alpha" in active:
        result[..., 3] = map_samples(arr[..., 3], active["Alpha"])
    return result


def change_curve_mode(curve: dict[str, Any], mode: str) -> dict[str, Any]:
    """GIMP converts freehand to smooth using nine evenly spaced samples."""
    if mode == curve["mode"]:
        return deepcopy(curve)
    samples = curve_samples(curve)
    if mode == "freehand":
        return {"mode": mode, "samples": samples.tolist()}
    indices = np.arange(9) * (N_SAMPLES - 1) // 8
    return {"mode": "smooth", "points": [
        [float(i / (N_SAMPLES - 1)), float(samples[i]), "smooth"] for i in indices
    ]}
