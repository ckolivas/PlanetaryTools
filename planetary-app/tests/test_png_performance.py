"""Native decoding must preserve PNG's encoded 16-bit channel samples."""
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PyQt6.QtGui import QColorSpace, QImage

from planetary_tools.io import png_read
from planetary_tools.io import loader
from planetary_tools.io.png_write import _chunk


def png_fixture(samples, filters, metadata=()):
    """Encode known samples with each PNG predictor independently of the reader."""
    height, width = samples.shape[:2]
    previous = bytes(width * 6)
    scanlines = []
    for y in range(height):
        row = samples[y].astype('>u2').tobytes()
        kind = filters[y % len(filters)]
        encoded = bytearray(len(row))
        for i, value in enumerate(row):
            a = row[i-6] if i >= 6 else 0
            b = previous[i]
            c = previous[i-6] if i >= 6 else 0
            if kind == 0:
                prediction = 0
            elif kind == 1:
                prediction = a
            elif kind == 2:
                prediction = b
            elif kind == 3:
                prediction = (a+b)//2
            else:
                p = a+b-c
                distances = (abs(p-a), abs(p-b), abs(p-c))
                prediction = (a,b,c)[distances.index(min(distances))]
            encoded[i] = (value-prediction) % 256
        scanlines.append(bytes([kind])+encoded)
        previous = row
    header = struct.pack('>IIBBBBB', width, height, 16, 2, 0, 0, 0)
    compressed = zlib.compress(b''.join(scanlines))
    middle = len(compressed)//2
    return (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', header)
            + b''.join(_chunk(name, payload) for name, payload in metadata)
            + _chunk(b'IDAT', compressed[:middle])
            + _chunk(b'IDAT', compressed[middle:]) + _chunk(b'IEND', b''))


class PngPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'sample.png'
        self.samples = np.random.default_rng(81).integers(0, 65536, (9, 13, 3), dtype=np.uint16)
        self.samples[0, :3] = [[0, 65535, 1], [256, 257, 32768], [65534, 12345, 54321]]

    def test_all_predictors_and_mixed_rows_preserve_every_bit(self):
        for filters in ((0,), (1,), (2,), (3,), (4,), (0,1,2,3,4)):
            with self.subTest(filters=filters):
                self.path.write_bytes(png_fixture(self.samples, filters))
                np.testing.assert_array_equal(png_read._read_png_rgb16_python(self.path), self.samples)
                with patch.object(png_read, '_read_png_rgb16_python', side_effect=AssertionError('unexpected fallback')):
                    result = png_read.read_png_rgb16(self.path)
                np.testing.assert_array_equal(result, self.samples)
                self.assertEqual(result.dtype, np.uint16)
                self.assertTrue(result.flags.owndata)
                self.assertTrue(result.flags.c_contiguous)

    def test_gamma_and_icc_do_not_transform_samples(self):
        profile = bytes(QColorSpace(QColorSpace.NamedColorSpace.SRgbLinear).iccProfile())
        for metadata in (
            [(b'gAMA', struct.pack('>I', 100000))],
            [(b'sRGB', b'\x00')],
            [(b'sBIT', bytes([10, 11, 12]))],
            [(b'iCCP', b'Linear RGB\x00\x00' + zlib.compress(profile))],
        ):
            with self.subTest(chunk=metadata[0][0]):
                self.path.write_bytes(png_fixture(self.samples, (4,), metadata))
                with patch.object(png_read, '_read_png_rgb16_python', side_effect=AssertionError('unexpected fallback')):
                    result = png_read.read_png_rgb16(self.path)
                np.testing.assert_array_equal(result, self.samples)

    def test_transparent_key_uses_original_decoder_to_preserve_rgb(self):
        metadata = [(b'tRNS', struct.pack('>HHH', *self.samples[0,0]))]
        self.path.write_bytes(png_fixture(self.samples, (4,), metadata))
        with patch.object(png_read, '_read_png_rgb16_python', wraps=png_read._read_png_rgb16_python) as fallback:
            result = png_read.read_png_rgb16(self.path)
            fallback.assert_called_once()
        np.testing.assert_array_equal(result, self.samples)

    def test_unavailable_or_lower_precision_native_decode_uses_exact_fallback(self):
        self.path.write_bytes(png_fixture(self.samples, (4,)))
        for image in (QImage(), QImage(13, 9, QImage.Format.Format_RGB32)):
            with patch('PyQt6.QtGui.QImageReader') as reader:
                reader.return_value.read.return_value = image
                result = png_read.read_png_rgb16(self.path)
            np.testing.assert_array_equal(result, self.samples)

    def test_header_reads_only_33_bytes_and_rejects_invalid_input(self):
        payload = png_fixture(self.samples, (0,))
        class TrackedStream(io.BytesIO):
            requested = []
            def read(self, size=-1):
                self.requested.append(size)
                return super().read(size)
        stream = TrackedStream(payload)
        with patch.object(Path, 'open', return_value=stream):
            self.assertEqual(png_read.read_png_ihdr('ignored.png'), (13, 9, 16, 2))
        self.assertEqual(stream.requested, [33])
        for data in (b'', b'not a PNG', payload[:28], payload[:12]+b'IDAT'+payload[16:]):
            self.path.write_bytes(data)
            with self.assertRaises(ValueError):
                png_read.read_png_ihdr(self.path)

    def test_rgb16_entry_point_rejects_8bit_png(self):
        header = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        self.path.write_bytes(b'\x89PNG\r\n\x1a\n'+_chunk(b'IHDR', header))
        with self.assertRaisesRegex(ValueError, 'depth=8'):
            png_read.read_png_rgb16(self.path)

    def test_full_load_preserves_linear_pixels_precision_and_noise_context(self):
        self.path.write_bytes(png_fixture(self.samples, (0,1,2,3,4)))
        with patch.object(loader, 'read_png_rgb16', png_read._read_png_rgb16_python):
            expected = loader.load_image(self.path)
        actual = loader.load_image(self.path)
        np.testing.assert_array_equal(actual.data, expected.data)
        self.assertEqual(actual.storage_bits, 16)
        self.assertEqual(actual.is_grayscale, expected.is_grayscale)
        self.assertEqual(actual.noise_context(), expected.noise_context())


if __name__ == '__main__':
    unittest.main()
