"""À-trous wavelet sharpen and denoise — equivalents of the GIMP plug-ins."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter



NUM_SCALES = 3
# GIMP wavelet-decompose uses successive blurs with doubling radius.
_SIGMAS = (1.0, 2.0, 4.0)
# GIMP unsharp-mask on scale layers uses std-dev 16.
_UNSHARP_STD = 16.0


def _atrous_decompose(channel: np.ndarray, n_scales: int = NUM_SCALES) -> tuple[list[np.ndarray], np.ndarray]:
    """Decompose a single channel into detail scales and residual."""
    scales: list[np.ndarray] = []
    smooth = channel.astype(np.float64)
    for i in range(n_scales):
        sigma = _SIGMAS[i] if i < len(_SIGMAS) else 2.0 ** i
        blurred = gaussian_filter(smooth, sigma)
        scales.append((smooth - blurred).astype(np.float32))
        smooth = blurred
    return scales, smooth.astype(np.float32)


def _merge_wavelet(scales: list[np.ndarray], residual: np.ndarray) -> np.ndarray:
    out = residual.astype(np.float64)
    for s in scales:
        out += s
    return out.astype(np.float32)


def _unsharp_mask(layer: np.ndarray, std_dev: float, amount: float) -> np.ndarray:
    if amount == 0.0:
        return layer
    blurred = gaussian_filter(layer, std_dev)
    return layer + amount * (layer - blurred)


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
        scales, residual = _atrous_decompose(ch)
        sharpened = []
        for i, scale in enumerate(scales):
            sharpened.append(_unsharp_mask(scale, _UNSHARP_STD, amounts[i]))
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
        scales, residual = _atrous_decompose(ch)
        denoised = []
        for i, scale in enumerate(scales):
            r = radii[i]
            if r > 0.0:
                denoised.append(gaussian_filter(scale, r).astype(np.float32))
            else:
                denoised.append(scale)
        return _merge_wavelet(denoised, residual)

    return _process_channels(data, is_grayscale, denoise_channel)