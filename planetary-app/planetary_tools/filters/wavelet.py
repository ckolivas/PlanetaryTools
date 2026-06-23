"""Wavelet sharpen and denoise matching GIMP plug-in-wavelet-decompose + GEGL ops."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

NUM_SCALES = 3
# GIMP wavelet-decompose: wavelet-blur radius 2**scale_index → 1, 2, 4.
_WAVELET_RADII = (1.0, 2.0, 4.0)
# GIMP unsharp-mask on scale layers uses std-dev 16.
_UNSHARP_STD = 16.0
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


def _wavelet_decompose(
    channel: np.ndarray,
    n_scales: int = NUM_SCALES,
) -> tuple[list[np.ndarray], np.ndarray]:
    """plug-in-wavelet-decompose for float linear data (grain extract scales)."""
    scales: list[np.ndarray] = []
    current = np.asarray(channel, dtype=np.float64)
    for i in range(n_scales):
        radius = _WAVELET_RADII[i] if i < len(_WAVELET_RADII) else 2.0 ** i
        blurred = wavelet_blur(current, radius).astype(np.float64)
        scales.append((current - blurred + _GRAIN_MIDPOINT).astype(np.float32))
        current = blurred
    return scales, current.astype(np.float32)


def _merge_wavelet(scales: list[np.ndarray], residual: np.ndarray) -> np.ndarray:
    """Recompose with grain merge: lower + upper − midpoint per scale layer."""
    out = np.asarray(residual, dtype=np.float64)
    for scale in scales:
        out += np.asarray(scale, dtype=np.float64) - _GRAIN_MIDPOINT
    return out.astype(np.float32)


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
    amounts = (fine, medium, coarse)

    def sharpen_channel(ch: np.ndarray) -> np.ndarray:
        scales, residual = _wavelet_decompose(ch)
        sharpened = [
            _unsharp_mask(scale, _UNSHARP_STD, amounts[i])
            for i, scale in enumerate(scales)
        ]
        return _merge_wavelet(sharpened, residual)

    return _process_channels(data, is_grayscale, sharpen_channel)


def wavelet_denoise(
    data: np.ndarray,
    is_grayscale: bool,
    fine: float = 3.0,
    medium: float = 1.0,
    coarse: float = 0.0,
) -> np.ndarray:
    radii = (fine, medium, coarse)

    def denoise_channel(ch: np.ndarray) -> np.ndarray:
        scales, residual = _wavelet_decompose(ch)
        denoised = []
        for i, scale in enumerate(scales):
            r = radii[i]
            if r > 0.0:
                denoised.append(gaussian_filter(scale, r).astype(np.float32))
            else:
                denoised.append(scale)
        return _merge_wavelet(denoised, residual)

    return _process_channels(data, is_grayscale, denoise_channel)