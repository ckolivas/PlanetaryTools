"""Estimate and subtract diffuse sky glow while protecting the bright subject.

Work in native linear light, estimating each colour channel separately. Bright
regions are excluded from the background estimate; a median removes compact
sources before smoothing. A feathered protection mask leaves the disk/rings
unchanged. The dark-sky pedestal is retained rather than forcing sky to black.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from planetary_tools.core.colour import linear_luminance

DEFAULT_DEGLOW_PARAMS = {
    'amount': 100.0,
    'radius': 7.0,
    'threshold': 7.0,
    'margin': 7.0,
    'feather': 15.0,
}

# Shared by the interactive panel and batch parameter editor.
DEGLOW_CONTROLS = (
    ('amount', 'Amount', 0.0, 100.0, 1, ' %',
     'Fraction of the estimated glow to remove. Zero leaves the image unchanged.'),
    ('radius', 'Glow scale', 2.0, 300.0, 1, ' px',
     'Scale of the diffuse glow to estimate. Single-pixel sources are excluded before background sampling.'),
    ('threshold', 'Protect above', 0.1, 80.0, 1, ' %',
     'Protect pixels above this percentage of the subject peak above sky. Lower protects more of the disk and rings.'),
    ('margin', 'Protection margin', 0.0, 200.0, 1, ' px',
     'Extra unchanged pixels around the protected subject.'),
    ('feather', 'Feather', 1.0, 200.0, 1, ' px',
     'Transition from the protected subject to full glow removal.'),
)


def _resize(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if plane.shape == shape:
        return np.asarray(plane, dtype=np.float32)
    return np.asarray(Image.fromarray(np.asarray(plane, dtype=np.float32)).resize(
        (shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def deglow(
    data: np.ndarray, is_grayscale: bool = False, *,
    amount: float = 100.0, radius: float = 7.0, threshold: float = 7.0,
    margin: float = 7.0, feather: float = 15.0,
) -> np.ndarray:
    """Return a float32 image, preserving shape, alpha and protected pixels."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
        raise ValueError('Deglow requires a grayscale, RGB or RGBA image.')
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError('Deglow requires nonempty, finite image data.')
    values = (amount, radius, threshold, margin, feather)
    if not all(np.isfinite(v) for v in values):
        raise ValueError('Deglow parameters must be finite.')
    if not (0 <= amount <= 100 and radius >= 2 and 0 < threshold <= 100
            and margin >= 0 and feather > 0):
        raise ValueError('Invalid deglow amount, scale or protection settings.')
    if amount == 0:
        return arr.copy()
    planes = arr[..., None] if arr.ndim == 2 else arr
    colour = planes[..., :3] if planes.shape[2] >= 3 else planes
    lum = (colour[..., 0] if is_grayscale or colour.shape[2] == 1
           else linear_luminance(colour))
    sky = float(np.percentile(lum, 10))
    # Suppress isolated hot pixels when measuring the subject peak.
    peak_model = ndimage.median_filter(lum, size=3, mode='nearest')
    peak = float(peak_model.max())
    # In mono, luminance is exactly the first channel. Its median is also
    # the first channel's glow-model source; avoid filtering it a second time.
    if not (is_grayscale or colour.shape[2] == 1):
        peak_model = None
    if peak <= sky:
        return arr.copy()
    bright = lum >= sky + (peak - sky)*threshold/100
    labels, _ = ndimage.label(bright)
    sizes = np.bincount(labels.ravel())
    extended = sizes >= 9
    extended[0] = False
    # An isolated star must not create a protected island of uncorrected sky.
    # Connected thin ring structures are retained by area, not by thickness.
    protected = ndimage.binary_fill_holes(extended[labels])
    if not protected.any():
        return arr.copy()
    distance = ndimage.distance_transform_edt(~protected)
    blend = np.clip((distance - margin) / feather, 0, 1).astype(np.float32)
    blend = blend*blend*(3 - 2*blend)
    if not np.any(blend):
        return arr.copy()

    # Bound preview cost and median footprint independently of image size.
    h, w = lum.shape
    factor = max(1.0, radius/5.0, max(h, w)/768.0)
    shape = (max(1, round(h/factor)), max(1, round(w/factor)))
    excluded = _resize((distance <= margin).astype(np.float32), shape) > 0.01
    excluded = ndimage.binary_dilation(excluded, iterations=1)
    if excluded.all():
        return arr.copy()
    nearest = ndimage.distance_transform_edt(excluded, return_distances=False,
                                            return_indices=True)
    kernel = max(3, 2*int(round(radius/factor)) + 1)
    sigma = max(0.5, radius/(3*factor))
    out = planes.copy()
    previous_source = None
    for channel in range(colour.shape[2]):
        source = colour[..., channel]
        # Loaded grayscale files use R=G=B. Exactly equal source channels
        # receive identical models and corrections under the shared mask.
        if previous_source is not None and np.array_equal(
            source.view(np.uint32), previous_source.view(np.uint32),
        ):
            out[..., channel] = out[..., channel - 1]
            continue
        # Remove single-pixel sources from the MODEL before downsampling can
        # spread their flux into surrounding samples. The output still uses
        # the untouched source pixels, minus only the smooth glow estimate.
        model_source = (peak_model if peak_model is not None
                        else ndimage.median_filter(source, size=3, mode='nearest'))
        peak_model = None
        previous_source = source
        small = _resize(model_source, shape).copy()
        # Inpainting before the median keeps bright disk/ring light out of
        # the glow model; otherwise an ordinary blur creates dark edge halos.
        small[excluded] = small[tuple(nearest[:, excluded])]
        background = ndimage.median_filter(small, size=kernel, mode='nearest')
        background = ndimage.gaussian_filter(background, sigma=sigma, mode='nearest')
        pedestal = float(np.percentile(background[~excluded], 10))
        glow = np.maximum(_resize(background, (h, w)) - pedestal, 0)
        corrected = np.maximum(source - glow*blend*(amount/100), np.minimum(source, 0))
        out[..., channel] = np.where(blend == 0, source, corrected)
    return out[..., 0] if arr.ndim == 2 else out
