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
from planetary_tools.filters.adaptive_deconv import _PreparedDeconvolution

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
    luminance: bool = True,
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

    engine = _PreparedDeconvolution(
        src, is_grayscale, bool(adaptive), bool(luminance) and not is_grayscale,
    )

    def metrics(amount: float) -> tuple[float, float]:
        out = engine.apply(float(amount))
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

    return _search_amount(metrics, target_noise, target_contrast, max_amount, progress)


def _search_amount(metrics, target_noise, target_contrast, max_amount, progress):
    """Largest 0.1-step amount whose noise and contrast stay within the targets."""
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


def auto_adaptive_deconv_channel_params(
    data: np.ndarray,
    target_noise: float = 3.5,
    target_contrast: float = 15.0,
    *,
    adaptive: bool = True,
    max_amount: float = _MAX_AMOUNT,
    progress: Callable[[float, float, float], None] | None = None,
    channel_started: Callable[[int], None] | None = None,
    texture_scale: float | None = None,
    chromatic: bool | None = None,
) -> list[AutoDeconvResult]:
    """Search one amount per RGB channel, each against the same targets.

    Each channel is scored on its own, as a greyscale plane, using the
    image's texture scale and colour flag so its noise score is on the same
    scale as the RGB readout. ``channel_started`` is called with 0, 1, 2.
    """
    target_noise = max(0.0, float(target_noise))
    target_contrast = max(0.0, float(target_contrast))
    max_amount = _round_step(max(0.0, min(float(max_amount), _MAX_AMOUNT)))
    src = np.asarray(data, dtype=np.float32)
    if texture_scale is None:
        texture_scale = estimate_texture_scale(src, False)
    if chromatic is None:
        chromatic = is_chromatic(src, False)
    engine = _PreparedDeconvolution(src, False, bool(adaptive), False)
    results = []
    for index in range(3):
        if channel_started is not None:
            channel_started(index)
        plane = src[..., index]

        def metrics(amount: float, index=index, plane=plane) -> tuple[float, float]:
            out = engine.apply_channel(index, float(amount))
            noise = absolute_noise(out, True, texture_scale=texture_scale, chromatic=chromatic)
            contrast = brightness_increase_pct(plane, out, True)
            return (0.0 if noise is None else float(noise),
                    0.0 if contrast is None else float(contrast))

        results.append(_search_amount(
            metrics, target_noise, target_contrast, max_amount, progress,
        ))
    return results
