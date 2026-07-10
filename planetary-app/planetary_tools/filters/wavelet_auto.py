"""Auto parameter search for wavelet sharpen (noise + contrast targets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from planetary_tools.core.brightness import brightness_increase_pct
from planetary_tools.core.noise import absolute_noise, estimate_texture_scale
from planetary_tools.filters.wavelet import (
    NUM_SCALES,
    _UNSHARP_STD,
    _from_perceptual,
    _merge_wavelet,
    _to_perceptual,
    _unsharp_mask,
    _wavelet_decompose,
    wavelet_sharpen,
)

_STEP = 0.1
_FINE_STEPS = (4.0, 1.0, 0.1)  # coarse → refine
_MEDIUM_STEPS = (1.0, 0.1)
_COARSE_STEP = 0.1
_MAX_AMOUNT = 300.0
_EPS = 1e-6


@dataclass(frozen=True)
class AutoSharpenResult:
    fine: float
    medium: float
    coarse: float
    noise: float
    contrast_pct: float


def _round_step(value: float) -> float:
    """Round to 0.1 so UI spinboxes and amount keys stay consistent."""
    return round(float(value) + 1e-9, 1)


class _SharpenTrialEngine:
    """Decompose once; cache unsharp results per (channel, scale, amount)."""

    def __init__(self, data: np.ndarray, is_grayscale: bool) -> None:
        self.is_grayscale = is_grayscale
        self.src = np.asarray(data, dtype=np.float32)
        # Fix noise residual probe scales to the unsharpened source PSF/texture.
        self.texture_scale = estimate_texture_scale(self.src, is_grayscale)
        self._prepared: list[tuple[list[np.ndarray], np.ndarray]] = []
        if is_grayscale:
            ch = self.src if self.src.ndim == 2 else self.src[..., 0]
            work = _to_perceptual(ch)
            scales, residual = _wavelet_decompose(work)
            self._prepared.append((scales, residual))
        else:
            for c in range(3):
                work = _to_perceptual(self.src[..., c])
                scales, residual = _wavelet_decompose(work)
                self._prepared.append((scales, residual))
        # (channel_index, scale_index, amount_key) -> sharpened scale layer
        self._usm_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _amount_key(self, amount: float) -> int:
        return int(round(float(amount) * 10.0 + 1e-9))

    def _sharpened_scale(
        self,
        channel_index: int,
        scale_index: int,
        amount: float,
    ) -> np.ndarray:
        key = (channel_index, scale_index, self._amount_key(amount))
        cached = self._usm_cache.get(key)
        if cached is not None:
            return cached
        scale = self._prepared[channel_index][0][scale_index]
        if amount <= 0.0:
            out = np.asarray(scale, dtype=np.float32)
        else:
            out = _unsharp_mask(scale, _UNSHARP_STD, amount)
        self._usm_cache[key] = out
        return out

    def apply(self, fine: float, medium: float, coarse: float) -> np.ndarray:
        amounts = (fine, medium, coarse)
        channels: list[np.ndarray] = []
        for ch_i, (_scales, residual) in enumerate(self._prepared):
            sharpened = [
                self._sharpened_scale(ch_i, s_i, amounts[s_i])
                for s_i in range(NUM_SCALES)
            ]
            channels.append(_from_perceptual(_merge_wavelet(sharpened, residual)))
        if self.is_grayscale:
            return channels[0]
        return np.stack(channels, axis=-1)

    def metrics(self, fine: float, medium: float, coarse: float) -> tuple[float, float]:
        out = self.apply(fine, medium, coarse)
        noise = absolute_noise(
            out, self.is_grayscale, texture_scale=self.texture_scale
        )
        contrast = brightness_increase_pct(self.src, out, self.is_grayscale)
        n = 0.0 if noise is None else float(noise)
        c = 0.0 if contrast is None else float(contrast)
        return n, c


def auto_wavelet_sharpen_params(
    data: np.ndarray,
    is_grayscale: bool,
    target_noise: float = 3.0,
    target_contrast: float = 15.0,
    *,
    max_amount: float = _MAX_AMOUNT,
    progress: Callable[[float, float, float, float, float], None] | None = None,
) -> AutoSharpenResult:
    """Search fine/medium/coarse to approach noise and contrast targets.

    Never exceeds ``target_noise`` or ``target_contrast``. Starts at 0/0/0 and:

    1. Raises fine in steps of 4.0, then 1.0, then 0.1 until noise hits the
       target (stops early if contrast hits).
    2. If contrast remains low, raises medium in steps of 1.0 then 0.1; lowers
       fine if noise overshoots (step 1.0 while contrast < half target, else 0.1).
    3. If contrast remains low, raises coarse by 0.1; lowers fine then medium
       if noise overshoots (same adaptive step-down).
    """
    target_noise = max(0.0, float(target_noise))
    target_contrast = max(0.0, float(target_contrast))
    max_amount = float(max_amount)
    fine_steps = _FINE_STEPS
    medium_steps = _MEDIUM_STEPS
    coarse_step = _COARSE_STEP
    compensate_fine_step = _STEP
    compensate_coarse_step = _STEP * 10.0  # 1.0 when contrast is still low

    engine = _SharpenTrialEngine(data, is_grayscale)

    fine = 0.0
    medium = 0.0
    coarse = 0.0

    def report() -> tuple[float, float]:
        n, c = engine.metrics(fine, medium, coarse)
        if progress is not None:
            progress(fine, medium, coarse, n, c)
        return n, c

    def would_exceed(n: float, c: float) -> bool:
        return n > target_noise + _EPS or c > target_contrast + _EPS

    def contrast_reached(c: float) -> bool:
        return c >= target_contrast - _EPS

    def noise_reached(n: float) -> bool:
        return n >= target_noise - _EPS

    def compensate_step_for(contrast: float) -> float:
        """Larger steps down while contrast is under half the target."""
        if contrast < 0.5 * target_contrast - _EPS:
            return compensate_coarse_step
        return compensate_fine_step

    # --- Phase 1: fine (4.0 → 1.0 → 0.1) ---
    for step in fine_steps:
        while fine + step <= max_amount + _EPS:
            trial = _round_step(fine + step)
            n, c = engine.metrics(trial, medium, coarse)
            if would_exceed(n, c):
                break
            fine = trial
            if progress is not None:
                progress(fine, medium, coarse, n, c)
            if contrast_reached(c):
                return AutoSharpenResult(fine, medium, coarse, n, c)
            if noise_reached(n):
                break
        n, c = report()
        if contrast_reached(c) or noise_reached(n):
            if contrast_reached(c):
                return AutoSharpenResult(fine, medium, coarse, n, c)
            break

    n, c = report()
    if contrast_reached(c):
        return AutoSharpenResult(fine, medium, coarse, n, c)

    # --- Phase 2: medium (1.0 → 0.1); compensate noise via fine ---
    for step in medium_steps:
        while medium + step <= max_amount + _EPS:
            trial_medium = _round_step(medium + step)
            trial_fine = fine
            n, c = engine.metrics(trial_fine, trial_medium, coarse)
            if c > target_contrast + _EPS:
                break
            while n > target_noise + _EPS:
                down = compensate_step_for(c)
                if trial_fine < down - _EPS:
                    # Prefer a full adaptive step; fall back to fine step if needed.
                    if trial_fine >= compensate_fine_step - _EPS:
                        down = compensate_fine_step
                    else:
                        break
                trial_fine = _round_step(trial_fine - down)
                n, c = engine.metrics(trial_fine, trial_medium, coarse)
                if c > target_contrast + _EPS:
                    break
            if would_exceed(n, c):
                break
            fine, medium = trial_fine, trial_medium
            if progress is not None:
                progress(fine, medium, coarse, n, c)
            if contrast_reached(c):
                return AutoSharpenResult(fine, medium, coarse, n, c)
        n, c = report()
        if contrast_reached(c):
            return AutoSharpenResult(fine, medium, coarse, n, c)

    n, c = report()
    if contrast_reached(c):
        return AutoSharpenResult(fine, medium, coarse, n, c)

    # --- Phase 3: coarse @ 0.1; compensate noise via fine then medium ---
    while coarse + coarse_step <= max_amount + _EPS:
        trial_coarse = _round_step(coarse + coarse_step)
        trial_fine = fine
        trial_medium = medium
        n, c = engine.metrics(trial_fine, trial_medium, trial_coarse)
        if c > target_contrast + _EPS:
            break
        while n > target_noise + _EPS:
            down = compensate_step_for(c)
            if trial_fine >= down - _EPS:
                trial_fine = _round_step(trial_fine - down)
            elif trial_fine >= compensate_fine_step - _EPS:
                trial_fine = _round_step(trial_fine - compensate_fine_step)
            elif trial_medium >= down - _EPS:
                trial_medium = _round_step(trial_medium - down)
            elif trial_medium >= compensate_fine_step - _EPS:
                trial_medium = _round_step(trial_medium - compensate_fine_step)
            else:
                break
            n, c = engine.metrics(trial_fine, trial_medium, trial_coarse)
            if c > target_contrast + _EPS:
                break
        if would_exceed(n, c):
            break
        fine, medium, coarse = trial_fine, trial_medium, trial_coarse
        if progress is not None:
            progress(fine, medium, coarse, n, c)
        if contrast_reached(c):
            return AutoSharpenResult(fine, medium, coarse, n, c)

    n, c = report()
    return AutoSharpenResult(fine, medium, coarse, n, c)


def verify_auto_params(
    data: np.ndarray,
    is_grayscale: bool,
    fine: float,
    medium: float,
    coarse: float,
) -> tuple[float, float]:
    """Return (noise, contrast%) for the given amounts on full data."""
    out = wavelet_sharpen(data, is_grayscale, fine, medium, coarse)
    texture_scale = estimate_texture_scale(data, is_grayscale)
    noise = absolute_noise(out, is_grayscale, texture_scale=texture_scale)
    contrast = brightness_increase_pct(data, out, is_grayscale)
    return (
        0.0 if noise is None else float(noise),
        0.0 if contrast is None else float(contrast),
    )
