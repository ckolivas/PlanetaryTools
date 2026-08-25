"""Crop or expand an image by rectangle, or autocrop to the central bright object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from planetary_tools.core.colour import linear_luminance

DEFAULT_BORDER_PX = 100
DEFAULT_MIN_BRIGHTNESS_PCT = 10.0
_MIN_OBJECT_AREA = 16


@dataclass(frozen=True)
class CropRect:
    """Axis-aligned region in image pixels. ``(x, y)`` is the top-left.

    Coordinates may lie outside the source image; those pixels are filled
    with black when the region is applied.
    """

    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def extends_outside(self, img_w: int, img_h: int) -> bool:
        return (
            self.x < 0
            or self.y < 0
            or self.x + self.width > int(img_w)
            or self.y + self.height > int(img_h)
        )


def normalize_crop_rect(x: int, y: int, width: int, height: int) -> CropRect:
    """Ensure width and height are at least 1 pixel; position is unchanged."""
    return CropRect(int(x), int(y), max(1, int(width)), max(1, int(height)))


def rect_from_size_offset(
    img_w: int,
    img_h: int,
    crop_w: int,
    crop_h: int,
    offset_x: int,
    offset_y: int,
) -> CropRect:
    """Region of ``crop_w``×``crop_h`` whose centre is offset from the image middle.

    ``offset_x`` / ``offset_y`` are in pixels; positive is right / down.
    Size may exceed the image (expand); ``x``/``y`` may be negative.
    """
    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))
    crop_w = max(1, int(crop_w))
    crop_h = max(1, int(crop_h))
    x = (img_w - crop_w) // 2 + int(offset_x)
    y = (img_h - crop_h) // 2 + int(offset_y)
    return CropRect(x, y, crop_w, crop_h)


def offset_from_rect(img_w: int, img_h: int, rect: CropRect) -> tuple[int, int]:
    """Integer centre-offsets that reproduce ``rect`` via ``rect_from_size_offset``."""
    ox = int(rect.x) - (int(img_w) - int(rect.width)) // 2
    oy = int(rect.y) - (int(img_h) - int(rect.height)) // 2
    return ox, oy


def _luminance(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if is_grayscale:
        return arr[..., 0]
    return linear_luminance(arr)


def autocrop_rect(
    data: np.ndarray,
    is_grayscale: bool = False,
    *,
    border_px: int = DEFAULT_BORDER_PX,
    min_brightness_pct: float = DEFAULT_MIN_BRIGHTNESS_PCT,
) -> tuple[CropRect, bool]:
    """Bounding box of the central bright object, expanded by ``border_px``.

    Pixels at least ``min_brightness_pct`` of the image's peak luminance are
    treated as object. The largest connected component is the object (planet
    disk, not scattered moons or noise). ``border_px`` is added on every side;
    if that reaches past the frame the canvas expands (new pixels are black).

    Returns ``(rect, found)``. ``found`` is False when no object meets the
    threshold; ``rect`` is then the full image.
    """
    arr = np.asarray(data, dtype=np.float32)
    img_h, img_w = int(arr.shape[0]), int(arr.shape[1])
    full = CropRect(0, 0, img_w, img_h)
    lum = _luminance(arr, is_grayscale)
    peak = float(np.max(lum)) if lum.size else 0.0
    if peak <= 0.0:
        return full, False

    thresh = (float(min_brightness_pct) / 100.0) * peak
    mask = lum >= thresh
    labels, count = ndimage.label(mask)
    if count == 0:
        return full, False

    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    sizes = np.atleast_1d(np.asarray(sizes, dtype=np.float64))
    best = int(np.argmax(sizes))
    if float(sizes[best]) < _MIN_OBJECT_AREA:
        return full, False

    obj = labels == (best + 1)
    obj = ndimage.binary_fill_holes(obj)
    ys, xs = np.where(obj)
    if ys.size == 0:
        return full, False

    border = max(0, int(border_px))
    x0 = int(xs.min()) - border
    y0 = int(ys.min()) - border
    x1 = int(xs.max()) + 1 + border
    y1 = int(ys.max()) + 1 + border
    return CropRect(x0, y0, x1 - x0, y1 - y0), True


def crop_image(
    data: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Copy ``data`` into a ``width``×``height`` canvas whose top-left is ``(x, y)``.

    Source pixels outside the image are filled with 0 (black). ``x``/``y`` may
    be negative and ``width``/``height`` may exceed the source size.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape for crop: {arr.shape}")
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    rect = normalize_crop_rect(x, y, width, height)
    if arr.ndim == 2:
        out = np.zeros((rect.height, rect.width), dtype=np.float32)
    else:
        out = np.zeros((rect.height, rect.width, arr.shape[2]), dtype=np.float32)

    src_x0 = max(0, rect.x)
    src_y0 = max(0, rect.y)
    src_x1 = min(src_w, rect.x + rect.width)
    src_y1 = min(src_h, rect.y + rect.height)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return out

    dst_x0 = src_x0 - rect.x
    dst_y0 = src_y0 - rect.y
    dh = src_y1 - src_y0
    dw = src_x1 - src_x0
    out[dst_y0 : dst_y0 + dh, dst_x0 : dst_x0 + dw, ...] = arr[
        src_y0:src_y1, src_x0:src_x1, ...
    ]
    return out
