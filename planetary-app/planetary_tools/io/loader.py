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


def _tiff_colour_space(path: Path, grayscale: bool):
    """Read TIFF colour interpretation separately from its raw sample array."""
    if path.suffix.lower() not in {".tif", ".tiff"} or not path.is_file():
        return None
    with tifffile.TiffFile(path) as image:
        tag = image.pages[0].tags.get("InterColorProfile")
        if tag is None:
            return None
        profile = bytes(tag.value)
    from PyQt6.QtGui import QColorSpace

    model = profile[16:20]
    if model not in (b"RGB ", b"GRAY") or (model == b"GRAY" and not grayscale):
        raise ValueError("TIFF ICC profile does not describe the image's RGB/gray channels")
    space = QColorSpace.fromIccProfile(profile)
    if not space.isValid():
        raise ValueError("TIFF contains an invalid or unsupported ICC colour profile")
    if model == b"GRAY":
        # A gray profile defines a neutral tone curve. Expand that curve onto
        # neutral RGB for our RGB working document and Qt's float image format.
        space.setPrimaries(QColorSpace.Primaries.SRgb)
    return space


def _profile_to_linear(samples: np.ndarray, space) -> np.ndarray:
    """Convert tagged samples to linear sRGB without an 8-bit intermediate."""
    from PyQt6.QtGui import QColorSpace, QImage

    space = QColorSpace(space)
    if space.transferFunction() == QColorSpace.TransferFunction.SRgb:
        # Exact floating transfer for the common case (including HDR), avoiding
        # a sampled ICC LUT for PlanetRecon's sRGB RGB and D65 gray exports.
        samples = srgb_to_linear(samples, clamp=False)
        space.setTransferFunction(QColorSpace.TransferFunction.Linear)
    if (space.transferFunction() == QColorSpace.TransferFunction.Linear
            and space.primaries() == QColorSpace.Primaries.SRgb):
        return samples.astype(np.float32)
    if samples.ndim == 2:
        samples = np.repeat(samples[..., None], 3, axis=-1)
    height, width = samples.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3], rgba[..., 3] = samples, 1.
    image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA32FPx4)
    image.setColorSpace(space)
    converted = image.convertedToColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgbLinear))
    if converted.isNull():
        raise ValueError("Cannot convert TIFF ICC profile to the linear RGB working space")
    pixels = converted.constBits()
    pixels.setsize(converted.sizeInBytes())
    rows = np.frombuffer(pixels, dtype=np.float32).reshape(height, converted.bytesPerLine() // 4)
    return rows[:, :width*4].reshape(height, width, 4)[..., :3].copy()


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
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(path)  # Raw samples; apply the ICC profile exactly once below.
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
    space = _tiff_colour_space(path, grayscale)
    if space is not None:
        if arr.dtype.kind == "u":
            samples = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        elif arr.dtype.kind == "f":
            samples = arr.astype(np.float32)
        else:
            raise ValueError("Profiled TIFF requires unsigned integer or floating samples")
        # Tagged floats already have a defined scale: never guess from maxima,
        # rescale HDR highlights to 16-bit ADU, or clip negative filter residuals.
        f = _profile_to_linear(samples, space)
        if f.ndim == 2:
            f = np.repeat(f[..., None], 3, axis=-1)
        return f, False, storage_bits
    linear_input = _is_probably_linear(path, arr)

    if arr.dtype == np.uint8:
        f = arr.astype(np.float32)
        f /= 255.0
        if not linear_input:
            f = srgb_to_linear(f)
    elif arr.dtype == np.uint16:
        f = arr.astype(np.float32)
        f /= 65535.0
        if not linear_input:
            f = srgb_to_linear(f)
    elif arr.dtype in (np.float32, np.float64):
        f = arr.astype(np.float32)
        if f.max() > 1.5:
            f /= 65535.0
    else:
        f = arr.astype(np.float32)
        peak = f.max()
        if peak > 1.0:
            f /= peak

    # Every branch above owns its float32 result, so clipping needs no copy.
    np.clip(f, 0.0, None, out=f)
    if grayscale:
        f = np.stack([f, f, f], axis=-1)
        grayscale = False
    return f, grayscale, storage_bits


def load_image(path: str | Path, *, pin_noise: bool = True) -> ImageDocument:
    """Load pixels and metadata, pinning editing noise context by default.

    Workers that only consume pixels can skip that analysis explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    arr = _load_array(path)
    data, grayscale, storage_bits = _normalize_array(arr, path)
    doc = ImageDocument(
        data=data,
        path=path,
        is_grayscale=grayscale,
        modified=False,
        storage_bits=storage_bits,
    )
    # Pin noise residual probes to the loaded stack so later enhance applies
    # do not re-estimate texture scale and change the absolute noise score.
    if pin_noise:
        doc.pin_noise_context()
    return doc


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
            tifffile.imwrite(path, np.asarray(doc.data, dtype=np.float32), photometric="minisblack")
        else:
            tifffile.imwrite(path, np.asarray(doc.data, dtype=np.float32))
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


def save_channel(data: np.ndarray, path: str | Path, *, bit_depth: int) -> None:
    """Save a single (H, W) linear channel as a greyscale image file."""
    from planetary_tools.core.colour import linear_to_srgb

    path = Path(path)
    suffix = path.suffix.lower()
    data = np.asarray(data, dtype=np.float32)

    if suffix in _FLOAT_EXTENSIONS and bit_depth == 32:
        tifffile.imwrite(path, data, photometric="minisblack")
        return
    if suffix in _FLOAT_EXTENSIONS and bit_depth == 16:
        out = np.clip(data, 0.0, 1.0)
        out = (out * 65535.0 + 0.5).astype(np.uint16)
        tifffile.imwrite(path, out, photometric="minisblack")
        return

    srgb = np.clip(linear_to_srgb(data), 0.0, 1.0)
    if suffix == ".png" and bit_depth >= 16:
        write_png_gray16(path, (srgb * 65535.0 + 0.5).astype(np.uint16))
    else:
        out = (srgb * 255.0 + 0.5).astype(np.uint8)
        _write_imageio(path, out)
