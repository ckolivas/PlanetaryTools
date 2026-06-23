"""Load and save images as 32-bit float linear colour."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

from planetary_tools.core.color import srgb_to_linear
from planetary_tools.core.document import ImageDocument

_IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    ".fits", ".fit", ".fts",
}

_FLOAT_EXTENSIONS = {".tif", ".tiff", ".fits", ".fit", ".fts"}


def supported_extensions() -> list[str]:
    return sorted(_IMAGE_EXTENSIONS)


def _is_probably_linear(path: Path, arr: np.ndarray) -> bool:
    """Heuristic: float TIFF/FITS from stacking pipelines is usually linear."""
    if path.suffix.lower() in _FLOAT_EXTENSIONS and arr.dtype in (np.float32, np.float64):
        return True
    if arr.dtype in (np.uint16, np.int16) and path.suffix.lower() in _FLOAT_EXTENSIONS:
        return True
    return False


def _normalize_array(arr: np.ndarray, path: Path) -> tuple[np.ndarray, bool]:
    """Return (float32 linear HxW or HxWx3, is_grayscale)."""
    arr = np.asarray(arr)

    if arr.ndim == 2:
        grayscale = True
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
            grayscale = True
        elif arr.shape[2] >= 3:
            arr = arr[..., :3]
            grayscale = False
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
    else:
        raise ValueError(f"Unsupported image rank: {arr.ndim}")

    linear_input = _is_probably_linear(path, arr)

    if arr.dtype == np.uint8:
        f = arr.astype(np.float32) / 255.0
        if not linear_input:
            f = srgb_to_linear(f)
    elif arr.dtype == np.uint16:
        f = arr.astype(np.float32) / 65535.0
        if not linear_input:
            f = srgb_to_linear(f)
    elif arr.dtype in (np.float32, np.float64):
        f = arr.astype(np.float32)
        if f.max() > 1.5:
            # 16-bit float stored in 32-bit container
            f = f / 65535.0
    else:
        f = arr.astype(np.float32)
        if f.max() > 1.0:
            f = f / f.max()

    f = np.clip(f, 0.0, None).astype(np.float32)
    return f, grayscale


def load_image(path: str | Path) -> ImageDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        try:
            arr = tifffile.imread(path)
        except Exception:
            arr = iio.imread(path)
    else:
        arr = iio.imread(path)

    data, grayscale = _normalize_array(arr, path)
    return ImageDocument(data=data, path=path, is_grayscale=grayscale, modified=False)


def save_image(doc: ImageDocument, path: str | Path, *, bit_depth: int = 16) -> None:
    """Save document. Float TIFF preserves linear data; other formats use sRGB."""
    from planetary_tools.core.color import linear_to_srgb

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"} and bit_depth == 32:
        if doc.is_grayscale:
            tifffile.imwrite(path, doc.data.astype(np.float32), photometric="minisblack")
        else:
            tifffile.imwrite(path, doc.data.astype(np.float32))
        doc.path = path
        doc.modified = False
        return

    if doc.is_grayscale:
        src = doc.data
        if suffix in _FLOAT_EXTENSIONS and bit_depth == 16:
            out = np.clip(src, 0.0, 1.0)
            out = (out * 65535.0 + 0.5).astype(np.uint16)
            tifffile.imwrite(path, out, photometric="minisblack")
            doc.path = path
            doc.modified = False
            return
        srgb = linear_to_srgb(src)
        out = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        iio.imwrite(path, out)
    else:
        srgb = linear_to_srgb(doc.data)
        srgb = np.clip(srgb, 0.0, 1.0)
        if suffix in _FLOAT_EXTENSIONS and bit_depth == 16:
            out = (srgb * 65535.0 + 0.5).astype(np.uint16)
            tifffile.imwrite(path, out)
        else:
            out = (srgb * 255.0 + 0.5).astype(np.uint8)
            iio.imwrite(path, out)

    doc.path = path
    doc.modified = False