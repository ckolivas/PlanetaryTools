"""Write 16-bit RGB PNG (Pillow/imageio only support 8-bit RGB PNG)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _write_png16(path: str | Path, arr: np.ndarray, *, colour_type: int, channels: int) -> None:
    height, width = arr.shape[:2]
    bpp = channels * 2
    rows: list[bytes] = []
    for y in range(height):
        row = arr[y].astype(">u2", copy=False).tobytes()
        if len(row) != width * bpp:
            raise ValueError("Unexpected row size for PNG encode")
        rows.append(b"\x00" + row)

    raw = b"".join(rows)
    compressed = zlib.compress(raw, level=6)
    ihdr = struct.pack(">IIBBBBB", width, height, 16, colour_type, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)


def write_png_rgb16(path: str | Path, rgb: np.ndarray) -> None:
    """Write uint16 RGB array (H, W, 3) as a 16-bit PNG."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"write_png_rgb16 expects HxWx3, got {arr.shape}")
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    _write_png16(path, arr, colour_type=2, channels=3)


def write_png_gray16(path: str | Path, gray: np.ndarray) -> None:
    """Write uint16 grayscale array (H, W) as a 16-bit PNG."""
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError(f"write_png_gray16 expects HxW, got {arr.shape}")
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    _write_png16(path, arr, colour_type=0, channels=1)