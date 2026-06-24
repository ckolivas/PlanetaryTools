"""Read 16-bit PNG without silent downconversion (Pillow/imageio drop to 8-bit)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_row(
    filter_type: int,
    row: bytes,
    prev: bytes | None,
    bpp: int,
) -> bytes:
    length = len(row)
    out = bytearray(length)
    raw = row
    if filter_type == 0:
        return raw
    for i in range(length):
        x = raw[i]
        a = out[i - bpp] if i >= bpp else 0
        b = prev[i] if prev is not None else 0
        c = prev[i - bpp] if prev is not None and i >= bpp else 0
        if filter_type == 1:
            out[i] = (x + a) & 0xFF
        elif filter_type == 2:
            out[i] = (x + b) & 0xFF
        elif filter_type == 3:
            out[i] = (x + ((a + b) // 2)) & 0xFF
        elif filter_type == 4:
            out[i] = (x + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")
    return bytes(out)


def read_png_ihdr(path: str | Path) -> tuple[int, int, int, int]:
    """Return (width, height, bit_depth, colour_type)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")
    pos = 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IHDR":
            return struct.unpack(">IIBBBBB", chunk)[:4]
        pos += 12 + length
    raise ValueError("PNG missing IHDR")


def read_png_rgb16(path: str | Path) -> np.ndarray:
    """Decode 16-bit-per-channel RGB PNG to uint16 array (H, W, 3)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")

    width = height = bit_depth = colour_type = None
    idat = bytearray()
    pos = 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IHDR":
            width, height, bit_depth, colour_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        pos += 12 + length

    if bit_depth != 16 or colour_type != 2:
        raise ValueError(f"read_png_rgb16 requires 16-bit RGB, got depth={bit_depth} type={colour_type}")

    raw = zlib.decompress(bytes(idat))
    bpp = 6  # 3 channels × 2 bytes
    stride = width * bpp
    prev: bytes | None = None
    rows: list[bytes] = []
    i = 0
    for _ in range(height):
        filter_type = raw[i]
        i += 1
        row = raw[i:i + stride]
        i += stride
        recon = _unfilter_row(filter_type, row, prev, bpp)
        rows.append(recon)
        prev = recon

    flat = b"".join(rows)
    arr = np.frombuffer(flat, dtype=">u2").reshape(height, width, 3).astype(np.uint16)
    return arr