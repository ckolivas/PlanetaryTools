"""Bounded transfer functions retain the original float64 equations/rounding."""
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.core import colour
from planetary_tools.filters import wavelet


def reference(value, encode, clamp):
    v = np.asarray(value, dtype=np.float64)
    if clamp:
        v = np.clip(v, 0, 1)
    else:
        sign = np.sign(v)
        v = np.abs(v)
    if encode:
        out = np.where(v <= .0031308, v * 12.92, 1.055 * np.power(v, 1/2.4) - .055)
    else:
        out = np.where(v <= .04045, v / 12.92, np.power((v + .055)/1.055, 2.4))
    return (out if clamp else sign * out).astype(np.float32)


def assert_bits(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


class TransferPerformanceTests(unittest.TestCase):
    def test_every_16bit_level_and_hdr_values_match_exact_bits(self):
        rng = np.random.default_rng(99)
        levels = np.arange(65536, dtype=np.float64)/65535
        hdr = rng.uniform(-1.5, 4, (192, 384, 3))
        for data in (levels, levels.astype(np.float32), hdr, hdr.astype(np.float32), hdr[::-1, ::2], np.asfortranarray(hdr)):
            data.setflags(write=False)
            before = data.copy()
            for encode, fn in ((True, colour.linear_to_srgb), (False, colour.srgb_to_linear)):
                for clamp in (False, True):
                    with self.subTest(shape=data.shape, dtype=data.dtype, encode=encode, clamp=clamp):
                        # Also force awkward block boundaries for the uint16 grid.
                        with patch.object(colour, '_TRANSFER_BLOCK_SIZE', 1003):
                            actual = fn(data, clamp=clamp)
                        assert_bits(self, actual, reference(data, encode, clamp))
                        self.assertFalse(np.shares_memory(data, actual))
            np.testing.assert_array_equal(data, before)

    def test_breakpoints_halfway_rounding_and_nonfinite_values(self):
        rng = np.random.default_rng(15)
        rounded = rng.uniform(.01, 1, 70001).astype(np.float32)
        midpoints = (rounded.astype(np.float64) + np.nextafter(rounded, np.float32(np.inf)).astype(np.float64))/2
        encoded_ties = np.power((midpoints+.055)/1.055, 2.4)
        linear_ties = 1.055 * np.power(midpoints, 1/2.4) - .055
        edges = np.array([-np.inf, np.nan, -.04045, -.0031308, -0., 0., .0031308, .04045, 1., np.inf])
        values = np.concatenate([edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf), encoded_ties, linear_ties])
        for encode, fn in ((True, colour.linear_to_srgb), (False, colour.srgb_to_linear)):
            for clamp in (False, True):
                with np.errstate(all='ignore'):
                    assert_bits(self, fn(values, clamp=clamp), reference(values, encode, clamp))

    def test_scalar_empty_integer_and_small_arrays(self):
        for data in (np.array(.5), np.empty((0,3)), np.arange(40, dtype=np.uint16).reshape(5,8), np.array([0., 1.], dtype=np.float16), np.full(65537, .5, dtype=object)):
            for encode, fn in ((True, colour.linear_to_srgb), (False, colour.srgb_to_linear)):
                for clamp in (False, True):
                    assert_bits(self, fn(data, clamp=clamp), reference(data, encode, clamp))

    def test_large_transfer_bounds_temporary_block_size(self):
        data = np.ones((321, 412, 3), dtype=np.float32)
        with patch.object(colour, '_transfer_block', wraps=colour._transfer_block) as transfer:
            actual = colour.linear_to_srgb(data)
            self.assertGreater(transfer.call_count, 1)
            self.assertTrue(all(call.args[0].size <= 65536 for call in transfer.call_args_list))
        assert_bits(self, actual, reference(data, True, True))

    def test_wavelet_pixels_match_with_large_transfer_blocks(self):
        source = np.random.default_rng(76).uniform(.001, .9, (265, 257, 3)).astype(np.float32)
        for gray in (False, True):
            data = source[..., 0] if gray else source
            with patch.object(wavelet, 'linear_to_srgb', side_effect=lambda x, clamp=True: reference(x, True, clamp)), patch.object(wavelet, 'srgb_to_linear', side_effect=lambda x, clamp=False: reference(x, False, clamp)):
                expected = wavelet.wavelet_sharpen(data, gray, fine=4, medium=2, coarse=1)
            actual = wavelet.wavelet_sharpen(data, gray, fine=4, medium=2, coarse=1)
            assert_bits(self, actual, expected)


if __name__ == '__main__':
    unittest.main()
