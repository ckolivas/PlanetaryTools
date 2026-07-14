"""Auto parameter search for adaptive deconvolution (noise + contrast targets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from planetary_tools.core.brightness import brightness_increase_pct
from planetary_tools.core.noise import (
    absolute_noise,
    estimate_texture_scale,
    is_chromatic,
)
from planetary_tools.filters.adaptive_deconv import adaptive_deconvolution

_MAX_AMOUNT = 100.0
_STEP = 0.1
_EPS = 1e-6


@dataclass(frozen=True)
class AutoDeconvResult:
    amount: float
    noise: float
    contrast_pct: float


def _round_step(value: float) -> float:
    """Round to 0.1 so UI spinboxes stay consistent."""
    return round(float(value) + 1e-9, 1)


def auto_adaptive_deconv_params(
    data: np.ndarray,
    is_grayscale: bool,
    target_noise: float = 3.5,
    target_contrast: float = 15.0,
    *,
    adaptive: bool = True,
    oklab: bool = True,
    max_amount: float = _MAX_AMOUNT,
    progress: Callable[[float, float, float], None] | None = None,
    texture_scale: float | None = None,
    chromatic: bool | None = None,
) -> AutoDeconvResult:
    """Binary-search amount to approach noise and contrast targets.

    Never exceeds ``target_noise`` or ``target_contrast``. Returns the largest
    amount in ``[0, max_amount]`` (0.1 steps) that stays within both limits —
    the single-slider analogue of wavelet-sharpen auto.

    Pass session-pinned ``texture_scale`` / ``chromatic`` (from the document at
    load) so residual probes match the UI noise readout.
    """
    target_noise = max(0.0, float(target_noise))
    target_contrast = max(0.0, float(target_contrast))
    max_amount = max(0.0, min(float(max_amount), _MAX_AMOUNT))
    max_amount = _round_step(max_amount)

    src = np.asarray(data, dtype=np.float32)
    if texture_scale is None:
        texture_scale = estimate_texture_scale(src, is_grayscale)
    if chromatic is None:
        chromatic = is_chromatic(src, is_grayscale)

    def metrics(amount: float) -> tuple[float, float]:
        out = adaptive_deconvolution(
            src,
            is_grayscale,
            float(amount),
            bool(adaptive),
            bool(oklab) and not is_grayscale,
        )
        noise = absolute_noise(
            out,
            is_grayscale,
            texture_scale=texture_scale,
            chromatic=chromatic,
        )
        contrast = brightness_increase_pct(src, out, is_grayscale)
        n = 0.0 if noise is None else float(noise)
        c = 0.0 if contrast is None else float(contrast)
        return n, c

    def ok(n: float, c: float) -> bool:
        return n <= target_noise + _EPS and c <= target_contrast + _EPS

    n0, c0 = metrics(0.0)
    if progress is not None:
        progress(0.0, n0, c0)
    if not ok(n0, c0):
        # Already over at zero strength — stay at identity.
        return AutoDeconvResult(0.0, n0, c0)

    # Discrete binary search over amount = i * 0.1 for i in [0, max_i].
    max_i = int(round(max_amount / _STEP))
    lo_i = 0
    hi_i = max_i
    best_i = 0
    best_n, best_c = n0, c0

    while lo_i <= hi_i:
        mid_i = (lo_i + hi_i) // 2
        amount = _round_step(mid_i * _STEP)
        n, c = metrics(amount)
        if progress is not None:
            progress(amount, n, c)
        if ok(n, c):
            best_i = mid_i
            best_n, best_c = n, c
            lo_i = mid_i + 1  # try stronger
        else:
            hi_i = mid_i - 1

    amount = _round_step(best_i * _STEP)
    if progress is not None:
        progress(amount, best_n, best_c)
    return AutoDeconvResult(amount, best_n, best_c)
