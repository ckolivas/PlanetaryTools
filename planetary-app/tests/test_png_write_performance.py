"""Lower-allocation PNG writing retains the exact existing file encoding."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from planetary_tools.io import png_write


def reference_png(data, gray):
    data = np.asarray(data)
    if data.dtype != np.uint16:
        data = np.clip(data, 0, 65535).astype(np.uint16)
    rows = [b'\x00' + row.astype('>u2', copy=False).tobytes() for row in data]
    header = struct.pack('>IIBBBBB', data.shape[1], data.shape[0], 16, 0 if gray else 2, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + png_write._chunk(b'IHDR', header)
            + png_write._chunk(b'IDAT', zlib.compress(b''.join(rows), level=6))
            + png_write._chunk(b'IEND', b''))


class PngWritePerformanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'output.png'

    def test_rgb_and_grayscale_files_match_byte_for_byte(self):
        rng = np.random.default_rng(57)
        rgb = rng.integers(0, 65536, (41, 53, 3), dtype=np.uint16)
        for gray in (False, True):
            source = rgb[..., 0] if gray else rgb
            writer = png_write.write_png_gray16 if gray else png_write.write_png_rgb16
            cases = (source, source[::-1, ::2], source.astype('>u2'), source.astype(np.float64)-100,
                     source[:1, :1], source[:0], source[:, :0])
            for data in cases:
                with self.subTest(gray=gray, dtype=data.dtype, shape=data.shape):
                    expected = reference_png(data, gray)
                    before = data.copy()
                    data.setflags(write=False)
                    writer(self.path, data)
                    self.assertEqual(self.path.read_bytes(), expected)
                    np.testing.assert_array_equal(data, before)

    def test_crc_and_uncompressed_scanlines_preserve_extreme_levels(self):
        source = np.array([[[0, 1, 65535], [256, 257, 65534]], [[32767, 32768, 32769], [12345, 54321, 42]]], dtype=np.uint16)
        png_write.write_png_rgb16(self.path, source)
        data = self.path.read_bytes()
        position = 8
        types = []
        while position < len(data):
            length = struct.unpack('>I', data[position:position+4])[0]
            kind = data[position+4:position+8]
            payload = data[position+8:position+8+length]
            crc = struct.unpack('>I', data[position+8+length:position+12+length])[0]
            self.assertEqual(crc, zlib.crc32(kind+payload) & 0xffffffff)
            types.append(kind)
            if kind == b'IDAT':
                raw = zlib.decompress(payload)
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(2, 13)
                np.testing.assert_array_equal(rows[:, 0], 0)
                samples = np.frombuffer(rows[:, 1:].tobytes(), dtype='>u2').reshape(source.shape)
                np.testing.assert_array_equal(samples, source)
            position += 12+length
        self.assertEqual(types, [b'IHDR', b'IDAT', b'IEND'])

    def test_validation_or_compression_failure_does_not_touch_destination(self):
        self.path.write_bytes(b'existing file')
        with self.assertRaises(ValueError):
            png_write.write_png_rgb16(self.path, np.ones((3, 4)))
        self.assertEqual(self.path.read_bytes(), b'existing file')
        with patch.object(png_write.zlib, 'compress', side_effect=zlib.error('failure')):
            with self.assertRaises(zlib.error):
                png_write.write_png_rgb16(self.path, np.ones((3, 4, 3), dtype=np.uint16))
        self.assertEqual(self.path.read_bytes(), b'existing file')


if __name__ == '__main__':
    unittest.main()
