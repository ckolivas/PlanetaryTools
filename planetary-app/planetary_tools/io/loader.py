"""Load and save images as 32-bit float linear colour."""

from __future__ import annotations

import zlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile

from planetary_tools.core.colour import srgb_to_linear
from planetary_tools.core.document import ImageDocument
from planetary_tools.io.png_read import read_png_ihdr, read_png_rgb16
from planetary_tools.io.png_write import write_png_gray16, write_png_rgb16

_IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    ".fits", ".fit", ".fts",
}

_FLOAT_EXTENSIONS = {".tif", ".tiff", ".fits", ".fit", ".fts"}
_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
_JPEG_QUALITY = 100


def supported_extensions() -> list[str]:
    return sorted(_IMAGE_EXTENSIONS)


def _is_probably_linear(path: Path, arr: np.ndarray) -> bool:
    """Heuristic for whether integer samples are radiometric vs display-encoded."""
    suffix = path.suffix.lower()
    if arr.dtype in (np.float32, np.float64):
        return suffix in _FLOAT_EXTENSIONS
    # 16-bit TIFF written by this app (and most display TIFFs) is sRGB-encoded.
    if arr.dtype in (np.uint16, np.int16):
        return suffix in {".fits", ".fit", ".fts"}
    return False


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".png":
        try:
            _, _, bit_depth, colour_type = read_png_ihdr(path)
            if bit_depth == 16 and colour_type == 2:
                return read_png_rgb16(path)
        except (ValueError, OSError, zlib.error):
            pass
        except Exception:
            pass
    if suffix in {".fits", ".fit", ".fts"}:
        try:
            return tifffile.imread(path)
        except Exception:
            return iio.imread(path)
    return iio.imread(path)


def _storage_bits(arr: np.ndarray, path: Path) -> int:
    if arr.dtype == np.uint8:
        return 8
    if arr.dtype == np.uint16:
        return 16
    if arr.dtype in (np.float32, np.float64):
        return 32
    return 8


def _normalize_array(arr: np.ndarray, path: Path) -> tuple[np.ndarray, bool, int]:
    """Return (float32 linear HxW or HxWx3, is_grayscale, storage_bits)."""
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

    storage_bits = _storage_bits(arr, path)
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
            f = f / 65535.0
    else:
        f = arr.astype(np.float32)
        if f.max() > 1.0:
            f = f / f.max()

    f = np.clip(f, 0.0, None).astype(np.float32)
    if grayscale:
        f = np.stack([f, f, f], axis=-1)
        grayscale = False
    return f, grayscale, storage_bits


def load_image(path: str | Path) -> ImageDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    arr = _load_array(path)
    data, grayscale, storage_bits = _normalize_array(arr, path)
    return ImageDocument(
        data=data,
        path=path,
        is_grayscale=grayscale,
        modified=False,
        storage_bits=storage_bits,
    )


def _effective_bit_depth(doc: ImageDocument, path: Path, bit_depth: int | None) -> int:
    if bit_depth is not None:
        return bit_depth
    if doc.storage_bits in (8, 16, 32):
        return doc.storage_bits
    return 16


def _write_imageio(path: Path, arr: np.ndarray) -> None:
    """Write 8-bit image via imageio; JPEG uses maximum quality."""
    if path.suffix.lower() in _JPEG_EXTENSIONS:
        iio.imwrite(path, arr, quality=_JPEG_QUALITY)
    else:
        iio.imwrite(path, arr)


def _finalize_save(doc: ImageDocument, path: Path, depth: int) -> None:
    doc.path = path
    doc.modified = False
    if depth in (8, 16, 32):
        doc.storage_bits = depth


def save_image(doc: ImageDocument, path: str | Path, *, bit_depth: int | None = None) -> None:
    """Save document. Float TIFF preserves linear data; PNG/TIFF honour bit depth."""
    from planetary_tools.core.colour import linear_to_srgb

    path = Path(path)
    suffix = path.suffix.lower()
    depth = _effective_bit_depth(doc, path, bit_depth)

    if suffix in {".tif", ".tiff"} and depth == 32:
        if doc.is_grayscale:
            tifffile.imwrite(path, doc.data.astype(np.float32), photometric="minisblack")
        else:
            tifffile.imwrite(path, doc.data.astype(np.float32))
        _finalize_save(doc, path, depth)
        return

    if doc.is_grayscale:
        src = doc.data
        if suffix in _FLOAT_EXTENSIONS and depth == 16:
            out = np.clip(src, 0.0, 1.0)
            out = (out * 65535.0 + 0.5).astype(np.uint16)
            tifffile.imwrite(path, out, photometric="minisblack")
            _finalize_save(doc, path, depth)
            return
        srgb = linear_to_srgb(src)
        srgb = np.clip(srgb, 0.0, 1.0)
        if suffix == ".png" and depth >= 16:
            write_png_gray16(path, (srgb * 65535.0 + 0.5).astype(np.uint16))
        else:
            out = (srgb * 255.0 + 0.5).astype(np.uint8)
            _write_imageio(path, out)
    else:
        srgb = linear_to_srgb(doc.data)
        srgb = np.clip(srgb, 0.0, 1.0)
        if suffix in _FLOAT_EXTENSIONS and depth == 16:
            out = (srgb * 65535.0 + 0.5).astype(np.uint16)
            tifffile.imwrite(path, out)
        elif suffix == ".png" and depth >= 16:
            write_png_rgb16(path, (srgb * 65535.0 + 0.5).astype(np.uint16))
        else:
            out = (srgb * 255.0 + 0.5).astype(np.uint8)
            _write_imageio(path, out)

    _finalize_save(doc, path, depth)