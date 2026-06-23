"""Wavelet sharpen and denoise matching GIMP plug-in-wavelet-decompose + GEGL ops."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from planetary_tools.core.color import linear_to_srgb, srgb_to_linear

NUM_SCALES = 3
# GIMP wavelet-decompose: wavelet-blur radius 2**scale_index → 1, 2, 4.
_WAVELET_RADII = (1.0, 2.0, 4.0)
# GIMP unsharp-mask on scale layers uses std-dev 16.
_UNSHARP_STD = 16.0
# Grain extract / merge midpoint in R'G'B' float (GEGL non-legacy).
_GRAIN_MIDPOINT = 0.5


def _wavelet_blur_1d_horizontal(channel: np.ndarray, radius: float) -> np.ndarray:
    """One horizontal pass of gegl:wavelet-blur-1d (HAT, weights 0.25/0.5/0.25)."""
    r = int(np.ceil(radius))
    if r <= 0:
        return np.asarray(channel, dtype=np.float64)

    arr = np.asarray(channel, dtype=np.float64)
    _, width = arr.shape
    padded = np.pad(arr, ((0, 0), (r, r)), mode="edge")
    return (
        0.25 * padded[:, :width]
        + 0.5 * padded[:, r:r + width]
        + 0.25 * padded[:, 2 * r:2 * r + width]
    )


def _wavelet_blur_1d_vertical(channel: np.ndarray, radius: float) -> np.ndarray:
    """One vertical pass of gegl:wavelet-blur-1d."""
    r = int(np.ceil(radius))
    if r <= 0:
        return np.asarray(channel, dtype=np.float64)

    arr = np.asarray(channel, dtype=np.float64)
    height, _ = arr.shape
    padded = np.pad(arr, ((r, r), (0, 0)), mode="edge")
    return (
        0.25 * padded[:height, :]
        + 0.5 * padded[r:r + height, :]
        + 0.25 * padded[2 * r:2 * r + height, :]
    )


def wavelet_blur(channel: np.ndarray, radius: float) -> np.ndarray:
    """Full gegl:wavelet-blur (horizontal then vertical)."""
    if radius <= 0.0:
        return np.asarray(channel, dtype=np.float32)
    tmp = _wavelet_blur_1d_horizontal(channel, radius)
    tmp = _wavelet_blur_1d_vertical(tmp, radius)
    return tmp.astype(np.float32)


def _to_perceptual(channel: np.ndarray) -> np.ndarray:
    """Document linear light → R'G'B' float (gegl:wavelet-blur working format)."""
    return linear_to_srgb(channel).astype(np.float64)


def _from_perceptual(channel: np.ndarray) -> np.ndarray:
    """R'G'B' float result → document linear light."""
    return srgb_to_linear(np.clip(channel, 0.0, 1.0)).astype(np.float32)


def _grain_extract(channel: np.ndarray, blurred: np.ndarray) -> np.ndarray:
    """gimp:grain-extract-legacy with CLAMP(comp, 0, 1)."""
    comp = (
        np.asarray(channel, dtype=np.float64)
        - np.asarray(blurred, dtype=np.float64)
        + _GRAIN_MIDPOINT
    )
    return np.clip(comp, 0.0, 1.0).astype(np.float32)


def _grain_merge(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """gimp:grain-merge-legacy with CLAMP(comp, 0, 1)."""
    comp = (
        np.asarray(base, dtype=np.float64)
        + np.asarray(layer, dtype=np.float64)
        - _GRAIN_MIDPOINT
    )
    return np.clip(comp, 0.0, 1.0).astype(np.float32)


def _wavelet_decompose(
    channel: np.ndarray,
    n_scales: int = NUM_SCALES,
) -> tuple[list[np.ndarray], np.ndarray]:
    """plug-in-wavelet-decompose in R'G'B' float (grain-extract scales)."""
    scales: list[np.ndarray] = []
    current = np.asarray(channel, dtype=np.float64)
    for i in range(n_scales):
        radius = _WAVELET_RADII[i] if i < len(_WAVELET_RADII) else 2.0 ** i
        blurred = wavelet_blur(current, radius).astype(np.float64)
        scales.append(_grain_extract(current, blurred))
        current = blurred
    return scales, current.astype(np.float32)


def _merge_wavelet(scales: list[np.ndarray], residual: np.ndarray) -> np.ndarray:
    """Recompose with grain merge coarse → fine (GIMP layer-stack order)."""
    out = np.asarray(residual, dtype=np.float32)
    for scale in reversed(scales):
        out = _grain_merge(out, scale)
    return out


def _unsharp_mask(layer: np.ndarray, std_dev: float, amount: float) -> np.ndarray:
    """gegl:unsharp-mask with threshold 0: input + scale × (input − blur)."""
    if amount == 0.0:
        return np.asarray(layer, dtype=np.float32)
    layer = np.asarray(layer, dtype=np.float64)
    blurred = gaussian_filter(layer, std_dev)
    return (layer + amount * (layer - blurred)).astype(np.float32)


def _process_channels(
    data: np.ndarray,
    is_grayscale: bool,
    per_channel,
) -> np.ndarray:
    if is_grayscale:
        ch = data if data.ndim == 2 else data[..., 0]
        return per_channel(ch)

    channels = []
    for c in range(3):
        channels.append(per_channel(data[..., c]))
    return np.stack(channels, axis=-1)


def wavelet_sharpen(
    data: np.ndarray,
    is_grayscale: bool,
    fine: float = 16.0,
    medium: float = 8.0,
    coarse: float = 1.0,
) -> np.ndarray:
    """Wavelet sharpen matching GIMP plug-in-wavelet-sharpen.

    GEGL gegl:wavelet-blur-1d prepares with
    ``babl_format_with_space ("R'G'B' float", space)`` — perceptual
    sRGB-encoded float, not linear light.  ImageDocument stores linear
    RGB; this filter converts to R'G'B' for decompose / unsharp / merge,
    then converts back to linear for the document.
    """
    amounts = (fine, medium, coarse)

    def sharpen_channel(ch: np.ndarray) -> np.ndarray:
        work = _to_perceptual(ch)
        scales, residual = _wavelet_decompose(work)
        sharpened = [
            _unsharp_mask(scale, _UNSHARP_STD, amounts[i])
            for i, scale in enumerate(scales)
        ]
        return _from_perceptual(_merge_wavelet(sharpened, residual))

    return _process_channels(data, is_grayscale, sharpen_channel)


def wavelet_denoise(
    data: np.ndarray,
    is_grayscale: bool,
    fine: float = 3.0,
    medium: float = 1.0,
    coarse: float = 0.0,
) -> np.ndarray:
    """Wavelet denoise in the same R'G'B' float space as GIMP decompose."""
    radii = (fine, medium, coarse)

    def denoise_channel(ch: np.ndarray) -> np.ndarray:
        work = _to_perceptual(ch)
        scales, residual = _wavelet_decompose(work)
        denoised = []
        for i, scale in enumerate(scales):
            r = radii[i]
            if r > 0.0:
                denoised.append(gaussian_filter(scale, r).astype(np.float32))
            else:
                denoised.append(scale)
        return _from_perceptual(_merge_wavelet(denoised, residual))

    return _process_channels(data, is_grayscale, denoise_channel)