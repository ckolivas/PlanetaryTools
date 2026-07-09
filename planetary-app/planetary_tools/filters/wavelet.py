"""Wavelet sharpen and denoise matching GIMP plug-in-wavelet-decompose + GEGL ops."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from planetary_tools.core.colour import linear_to_srgb, srgb_to_linear

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
    # Do not clamp: preserve values already above 100% linear.
    return linear_to_srgb(channel, clamp=False).astype(np.float64)


def _from_perceptual(channel: np.ndarray) -> np.ndarray:
    """R'G'B' float result → document linear light (unclamped).

    Highlight overshoot is left intact so optional post-process clamp and the
    brightness-increase readout can see values above 100%.
    """
    return srgb_to_linear(channel, clamp=False).astype(np.float32)


def _grain_extract(channel: np.ndarray, blurred: np.ndarray) -> np.ndarray:
    """Grain-extract (GIMP legacy formula) without clamping scale layers."""
    comp = (
        np.asarray(channel, dtype=np.float64)
        - np.asarray(blurred, dtype=np.float64)
        + _GRAIN_MIDPOINT
    )
    return comp.astype(np.float32)


def _grain_merge(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Grain-merge (GIMP legacy formula) without clamping the result.

    Left open so the recomposed image can exceed 100% (highlight overshoot)
    for the optional clamp post-process and brightness-increase readout.
    """
    comp = (
        np.asarray(base, dtype=np.float64)
        + np.asarray(layer, dtype=np.float64)
        - _GRAIN_MIDPOINT
    )
    return comp.astype(np.float32)


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
    """gegl:unsharp-mask with threshold 0: input + scale × (input − blur).

    GEGL's gegl:gaussian-blur prepares with "RGB float" (linear), so for
    R'G'B' scale layers the gaussian is computed in linear light.  The USM
    arithmetic also runs in linear; results are not clipped so scale-layer
    and recomposed overshoot are visible to the brightness readout.
    """
    if amount == 0.0:
        return np.asarray(layer, dtype=np.float32)
    layer_f32 = np.asarray(layer, dtype=np.float32)
    layer_lin = srgb_to_linear(layer_f32, clamp=False).astype(np.float64)
    blurred = gaussian_filter(layer_lin, std_dev)
    usm_lin = layer_lin + amount * (layer_lin - blurred)
    return linear_to_srgb(usm_lin, clamp=False)


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

    Decompose and merge operate in R'G'B' perceptual float (matching
    gegl:wavelet-blur-1d).  The unsharp-mask step converts each scale
    layer to linear light (matching gegl:gaussian-blur which uses "RGB
    float"), applies USM there, and converts back before merge.
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


def merge_wavelet_detail(
    main_data: np.ndarray,
    secondary_data: np.ndarray,
    main_is_grayscale: bool,
    n_secondary_scales: int = 3,
) -> np.ndarray:
    """Replace the finest N wavelet scales from a secondary (NIR) image.

    The main image's residual (low-frequency colour) is preserved; the
    finest ``n_secondary_scales`` detail layers come from the secondary
    image, with the remaining coarser layers kept from the main image.
    The secondary is reduced to luminance so it can drive all colour
    channels of the main image.
    """
    # Reduce secondary to a single luminance channel (NIR is typically grey)
    if secondary_data.ndim == 3 and secondary_data.shape[2] >= 3:
        sec_lin: np.ndarray = (
            0.2126 * secondary_data[..., 0]
            + 0.7152 * secondary_data[..., 1]
            + 0.0722 * secondary_data[..., 2]
        ).astype(np.float32)
    elif secondary_data.ndim == 3:
        sec_lin = secondary_data[..., 0].astype(np.float32)
    else:
        sec_lin = np.asarray(secondary_data, dtype=np.float32)

    # Resize secondary if dimensions differ from main
    main_h, main_w = main_data.shape[:2]
    if sec_lin.shape != (main_h, main_w):
        from planetary_tools.core.scale import scale_image  # avoid circular
        sec_3ch = np.stack([sec_lin, sec_lin, sec_lin], axis=-1)
        sec_lin = scale_image(sec_3ch, main_w, main_h)[..., 0]

    # Decompose secondary once; reuse its scales for every main channel
    sec_perc = linear_to_srgb(sec_lin, clamp=False).astype(np.float64)
    sec_scales, _ = _wavelet_decompose(sec_perc)

    n = min(max(n_secondary_scales, 0), NUM_SCALES)

    def merge_channel(main_ch: np.ndarray) -> np.ndarray:
        work = _to_perceptual(main_ch)
        main_scales, main_residual = _wavelet_decompose(work)
        merged = [
            sec_scales[i] if i < n else main_scales[i]
            for i in range(NUM_SCALES)
        ]
        return _from_perceptual(_merge_wavelet(merged, main_residual))

    return _process_channels(main_data, main_is_grayscale, merge_channel)


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