"""Align a single-channel image to a reference by luma cross-correlation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndi_shift

from planetary_tools.core.scale import scale_image

_UPSCALE_FACTOR = 3
_MAX_SHIFT_PX = 15  # search radius at the upscaled resolution (~5 px at original scale)


def _best_shift(reference: np.ndarray, target: np.ndarray, max_shift: int) -> tuple[int, int]:
    """Return the (dy, dx) integer shift of ``target`` that best matches ``reference``."""
    ref = reference.astype(np.float64)
    tgt = target.astype(np.float64)
    ref = ref - ref.mean()
    tgt = tgt - tgt.mean()

    corr = np.fft.ifft2(np.fft.fft2(ref) * np.conj(np.fft.fft2(tgt))).real
    height, width = ref.shape

    best_score = -np.inf
    best = (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            score = corr[dy % height, dx % width]
            if score > best_score:
                best_score = score
                best = (dy, dx)
    return best


def align_channel(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align ``target`` to ``reference`` by best luma match.

    Both are enlarged 3× for sub-pixel precision, the best integer-pixel shift
    is found via cross-correlation, ``target`` is shifted to match, then the
    result is resized back down to the original dimensions.
    """
    height, width = reference.shape
    up_w, up_h = width * _UPSCALE_FACTOR, height * _UPSCALE_FACTOR

    ref_up = scale_image(reference, up_w, up_h)
    tgt_up = scale_image(target, up_w, up_h)

    dy, dx = _best_shift(ref_up, tgt_up, _MAX_SHIFT_PX)
    if dy == 0 and dx == 0:
        return target

    aligned_up = ndi_shift(tgt_up, shift=(dy, dx), mode="nearest", order=1)
    return scale_image(aligned_up, width, height)
