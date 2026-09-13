"""Levels histogram optimizations must retain samples and chosen parameters."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.core.colour import linear_to_srgb, rgb_to_oklab
from planetary_tools.filters import levels


def original_channel_values(data, channel):
    rgb = np.asarray(data, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if channel == 'L':
        return rgb_to_oklab(rgb)[..., 0].ravel()
    return linear_to_srgb(rgb)[..., {'R': 0, 'G': 1, 'B': 2}[channel]].ravel()


def original_auto_input(values):
    flat = np.asarray(values, dtype=np.float64).ravel()
    if not flat.size:
        return levels.identity_levels()
    lo = float(np.percentile(flat, 2.))
    hi = float(np.percentile(flat, 98.))
    if hi - lo < 1e-10:
        return levels.identity_levels()
    return dict(in_min=lo, in_max=hi, gamma=1., out_min=0., out_max=1.)


class LevelsPerformanceTests(unittest.TestCase):
    def test_samples_match_original_conversion_bits_and_select_one_channel(self):
        rng = np.random.default_rng(93)
        rgb = rng.uniform(-.2, 1.2, (271, 257, 3)).astype(np.float32)
        for source in (rgb, rgb[::-1, ::2], np.asfortranarray(rgb), rgb.astype(np.float64), rgb[...,0], np.empty((0, 3, 3), dtype=np.float32)):
            source.setflags(write=False)
            for channel in levels.LEVEL_CHANNELS:
                with self.subTest(shape=source.shape, channel=channel):
                    expected = original_channel_values(source, channel)
                    with patch.object(levels, 'linear_to_srgb', wraps=linear_to_srgb) as convert:
                        actual = levels._channel_values(source, channel)
                        if channel != 'L':
                            self.assertEqual(convert.call_count, 1)
                            self.assertEqual(convert.call_args.args[0].shape, source.shape[:2])
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_joint_percentiles_match_separate_original_percentiles(self):
        rng = np.random.default_rng(37)
        cases = [np.empty(0), np.ones(100), np.linspace(0, 1, 65536),
                 np.array([0., 1e-10]), np.array([0., np.nextafter(1e-10, 1.)])]
        cases.extend(rng.random(n).astype(dtype) for n in (1, 2, 3, 49, 50, 51, 113, 10001)
                     for dtype in (np.float32, np.float64))
        cases.extend((np.array([0., np.nan, 1.]), np.array([-np.inf, 0., np.inf])))
        for values in cases:
            original = values.copy()
            values.setflags(write=False)
            with np.errstate(invalid='ignore'):
                expected = original_auto_input(values)
                actual = levels.auto_input_levels_for_channel(values)
            np.testing.assert_array_equal(list(actual.values()), list(expected.values()))
            np.testing.assert_array_equal(values, original)

    def test_auto_balance_and_applied_pixels_match_original_histograms(self):
        rng = np.random.default_rng(94)
        rgb = rng.uniform(0, 1, (97, 113, 3)).astype(np.float32)
        for source in (rgb, rgb[::-1, ::2], np.asfortranarray(rgb), rgb[...,0], np.ones_like(rgb)):
            for gray in (False, True):
                with ExitStack() as stack:
                    stack.enter_context(patch.object(levels, '_channel_values', original_channel_values))
                    stack.enter_context(patch.object(levels, 'auto_input_levels_for_channel', original_auto_input))
                    expected = levels.auto_balance_levels(source, is_grayscale=gray)
                actual = levels.auto_balance_levels(source, is_grayscale=gray)
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(levels.apply_levels(source, actual), levels.apply_levels(source, expected))

    def test_channel_peaks_keep_clamping_and_empty_defaults(self):
        for source in (np.array([[[-.1, .25, 1.2]]]), np.empty((0, 2, 3)), np.ones((5, 7))):
            for channel in levels.LEVEL_CHANNELS:
                values = original_channel_values(source, channel)
                expected = float(np.clip(values.max(), 0, 1)) if values.size else 1.
                self.assertEqual(levels.channel_input_peak(source, channel), expected)


if __name__ == '__main__':
    unittest.main()
