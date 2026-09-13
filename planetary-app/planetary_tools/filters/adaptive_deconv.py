"""Adaptive deconvolution — numpy/scipy port of the GIMP plug-in."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve, uniform_filter

from planetary_tools.core.colour import linear_luminance

# Moffat PSF (gamma=1, beta=2, 5x5) — matches GIMP plug-in
_GAMMA = 1.0
_BETA = 2.0
_PSF_SIZE = 5


def _generate_moffat_kernel(gamma: float = 1.0, beta: float = 2.0, size: int = 5) -> np.ndarray:
    half = size // 2
    psf = np.zeros((size, size), dtype=np.float64)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            r = math.sqrt(dx * dx + dy * dy)
            psf[dy + half, dx + half] = (1.0 + (r / gamma) ** 2) ** (-beta)
    psf /= psf.sum()
    return psf.astype(np.float32)


_PSF = _generate_moffat_kernel(_GAMMA, _BETA, _PSF_SIZE)
_PSF_MIRROR = _PSF[::-1, ::-1]


def _std_windowed(lum: np.ndarray, win_size: tuple[int, int] = (7, 7)) -> np.ndarray:
    """Local standard deviation via box-filtered mean and mean-of-squares."""
    size = win_size[0]
    lum64 = np.asarray(lum, dtype=np.float64)
    mean = uniform_filter(lum64, size=size, mode="reflect")
    mean_sq = uniform_filter(lum64 ** 2, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def _convolve2d(flat: np.ndarray, kernel: np.ndarray, width: int, height: int) -> np.ndarray:
    img = flat.reshape(height, width)
    out = convolve(img, kernel, mode="reflect")
    return out.ravel().astype(np.float32, copy=False)


def _luma_sharpen_rgb(
    rgb: np.ndarray,
    damped: np.ndarray,
    *, y: np.ndarray | None = None,
) -> np.ndarray:
    """Apply grayscale-style luma gain as an additive Rec.709 delta.

    ``damped`` is the same per-pixel factor used on grayscale, where the
    linear result is ``Y * damped**3``. Adding ``Y_new − Y`` leaves
    ``R−Y, G−Y, B−Y`` unchanged. RGB is not clamped so highlight overshoot
    remains for the optional clamp post-process and brightness readout.
    """
    if y is None:
        y = linear_luminance(rgb)
    gain = np.asarray(damped, dtype=np.float32).reshape(y.shape)
    y_new = y * gain * gain * gain
    return (rgb + (y_new - y)[..., None]).astype(np.float32, copy=False)


def adaptive_deconvolution(
    data: np.ndarray,
    is_grayscale: bool,
    amount: float = 10.0,
    adaptive: bool = True,
    luminance: bool = True,
) -> np.ndarray:
    """Adaptive Moffat deconvolution without hard-capping highlights at 100%.

    The GIMP plug-in floors results with ``min(result, min(2*peak, 1))``, which
    always clamps when the image is already at full scale.  We leave values
    open so overshoot is visible; optional dialog clamp handles 100% limiting.

    On RGB, ``luminance=True`` sharpens BT.709 luma and adds the delta back
    to linear RGB. ``luminance=False`` deconvolves each channel independently.
    """
    return _PreparedDeconvolution(data, is_grayscale, adaptive, luminance).apply(amount)


class _PreparedDeconvolution:
    """Amount-independent fields retained only for one filter/search session."""

    def __init__(
        self, data: np.ndarray, is_grayscale: bool,
        adaptive: bool = True, luminance: bool = True,
    ) -> None:
        self.src = np.asarray(data, dtype=np.float32)
        self.is_gray = is_grayscale
        self.per_channel = not luminance and not is_grayscale
        self.height, self.width = self.src.shape[:2]
        if is_grayscale:
            ch = self.src if self.src.ndim == 2 else self.src[..., 0]
            self.lum = ch.flatten()
        else:
            self.lum = linear_luminance(self.src).ravel()

        self.sqrt_contrast = None
        if adaptive:
            contrast = _std_windowed(self.lum.reshape(self.height, self.width)).ravel()
            c_min = float(contrast.min())
            c_max = float(contrast.max())
            contrast_norm = (contrast - c_min) / (c_max - c_min + 1e-10)
            self.sqrt_contrast = np.sqrt(contrast_norm)

        if self.per_channel:
            self.channels = [self.src[..., c].flatten() for c in range(3)]
        else:
            self.channels = [self.lum]
        self.corrections = []
        for channel in self.channels:
            conv = _convolve2d(channel, _PSF, self.width, self.height)
            relative = channel / (conv + 1e-12)
            correction = _convolve2d(relative, _PSF_MIRROR, self.width, self.height)
            self.corrections.append(correction - 1.0)

    def apply(self, amount: float) -> np.ndarray:
        strength = amount / math.pi
        weighted_strength = (
            strength * self.sqrt_contrast if self.sqrt_contrast is not None else strength
        )
        sharpened = []
        for channel, correction in zip(self.channels, self.corrections):
            # Keep the original multiplication order and float32 rounding.
            damped = 1.0 + weighted_strength * correction
            if not self.is_gray and not self.per_channel:
                return _luma_sharpen_rgb(
                    self.src, damped, y=self.lum.reshape(self.height, self.width),
                )
            sharpened.append(channel * damped * damped * damped)

        if self.is_gray:
            return sharpened[0].reshape(self.height, self.width)
        return np.stack(sharpened, axis=-1).reshape(self.height, self.width, 3)
