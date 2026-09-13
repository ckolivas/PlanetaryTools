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
    # One contiguous scanline buffer instead of retaining row bytes and then
    # joining them into a second full-image allocation.
    raw = np.empty((height, width * bpp + 1), dtype=np.uint8)
    raw[:, 0] = 0  # PNG filter type: None (same encoding as before).
    for y in range(height):
        row = arr[y].astype(">u2", copy=False).tobytes()
        if len(row) != width * bpp:
            raise ValueError("Unexpected row size for PNG encode")
        raw[y, 1:] = np.frombuffer(row, dtype=np.uint8)

    compressed = zlib.compress(raw, level=6)
    del raw
    ihdr = struct.pack(">IIBBBBB", width, height, 16, colour_type, 0, 0, 0)
    # Validate/encode before opening the destination, just as before. Write
    # the already-compressed payload directly without assembling another PNG
    # or IDAT-sized bytes object. Chunk contents and CRCs are unchanged.
    crc = zlib.crc32(compressed, zlib.crc32(b"IDAT")) & 0xFFFFFFFF
    with Path(path).open('wb') as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(_chunk(b"IHDR", ihdr))
        stream.write(struct.pack(">I", len(compressed)) + b"IDAT")
        stream.write(compressed)
        stream.write(struct.pack(">I", crc))
        stream.write(_chunk(b"IEND", b""))


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
