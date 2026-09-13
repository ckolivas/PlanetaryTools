"""Pixel/measurement parity for reduced image-buffer work."""
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.core import brightness, scale
from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.filters.wavelet import merge_wavelet_detail
from planetary_tools.filters import registry
from planetary_tools.ui import histogram


def reference_histograms(data, perceptual):
    rgb = np.asarray(data, dtype=np.float32)
    rgb = np.stack([rgb] * 3, axis=-1) if rgb.ndim == 2 else rgb[..., :3]
    if perceptual:
        rgb = linear_to_srgb(rgb)
    samples = np.clip(rgb, 0, 1).reshape(-1, 3)
    pixels = len(samples)
    if pixels > histogram._MAX_SAMPLES:
        samples = samples[np.linspace(0, pixels - 1, histogram._MAX_SAMPLES, dtype=np.int64)]
    counts = np.array([
        np.bincount((np.clip(samples[:, c], 0, 1) * 255).astype(np.intp), minlength=256)
        for c in range(3)
    ], dtype=np.float32)
    if pixels > len(samples):
        counts *= pixels / max(len(samples), 1)
    heights = np.log1p(counts)
    if heights.max() > 0:
        heights /= float(heights.max())
    return heights, counts


class ImagePerformanceTests(unittest.TestCase):
    def test_identity_check_matches_full_precision_and_strict_threshold(self):
        for dtype in (np.float32, np.float64):
            source = np.zeros((300, 400, 3), dtype=dtype)
            for delta in (0, .999e-5, 1e-5, 1.001e-5, np.nan, np.inf):
                output = source.copy()
                output[-1, -1, -1] = delta
                for left, right in ((output, source), (output[::-1, ::2], source[::-1, ::2]), (output.transpose(1, 0, 2), source.transpose(1, 0, 2))):
                    expected = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) < 1e-5
                    self.assertEqual(registry._near_identity(left, right), expected)
        self.assertFalse(registry._near_identity(np.zeros((2, 3)), np.zeros((3, 2))))

    def test_identity_check_bounds_buffers_and_stops_on_changed_block(self):
        source = np.zeros((300, 400, 3), dtype=np.float32)
        for output, expected in ((source, True), (source + 1, False)):
            with patch.object(registry.np, 'max', wraps=np.max) as maximum:
                self.assertEqual(registry._near_identity(output, source), expected)
                self.assertTrue(maximum.called)
                self.assertTrue(all(call.args[0].size <= 65536 for call in maximum.call_args_list))
                if not expected:
                    self.assertEqual(maximum.call_count, 1)

    def test_identity_stats_reuse_source_only_within_tolerance(self):
        source = np.zeros((300, 400), dtype=np.float32)
        for delta, expected_noise, calls in ((0, 2., 1), (1e-6, 2., 1), (1e-4, 3., 2)):
            raw = source.copy()
            raw[-1, -1] = delta
            with patch.object(registry, 'absolute_noise', side_effect=[2., 3.]) as noise:
                result = registry._stats_from_raw('wavelet_sharpen', source, raw, True, texture_scale=2, chromatic=False)
                self.assertEqual(result.noise_level, expected_noise)
                self.assertEqual(noise.call_count, calls)
        with patch.object(registry, 'absolute_noise', return_value=None), patch.object(registry, '_near_identity') as identity:
            registry._stats_from_raw('wavelet_sharpen', source, source, True, texture_scale=2, chromatic=False)
            identity.assert_not_called()

    def test_brightness_matches_float64_measurements(self):
        rng = np.random.default_rng(29)
        base = rng.uniform(-.2, 1.4, (21, 35, 3))
        for dtype in (np.float16, np.float32, np.float64, np.uint16):
            for gray in (True, False):
                for data in (base.astype(dtype), base.astype(dtype)[::2, ::-1]):
                    with self.subTest(dtype=dtype, gray=gray, shape=data.shape):
                        before = data.copy()
                        ref = np.asarray(data, dtype=np.float64)
                        ref = ref[..., 0] if gray else ref
                        lo, hi = float(ref.min()), float(ref.max())
                        info = brightness.measure_brightness(data, gray)
                        self.assertEqual((info.min_pct, info.max_pct), (lo * 100, hi * 100))
                        self.assertEqual(info.would_clip, lo < -1e-6 or hi > 1 + 1e-6)
                        self.assertEqual(brightness.channel_range(data, gray), (lo * 100, hi * 100))
                        self.assertEqual(brightness.would_clip_low(data, gray), lo < -1e-6)
                        self.assertEqual(brightness.would_clip_high(data, gray), hi > 1 + 1e-6)
                        output = data * .8
                        ref_out = np.asarray(output, dtype=np.float64)
                        ref_out = ref_out[..., 0] if gray else ref_out
                        expected = None if hi * 100 < 1e-6 else (float(ref_out.max()) * 100 / (hi * 100) - 1) * 100
                        self.assertEqual(brightness.brightness_increase_pct(data, output, gray), expected)
                        np.testing.assert_array_equal(data, before)

    def test_brightness_thresholds_grayscale_and_nonfinite(self):
        for lo, hi in ((0., 0.), (-1e-6, 1 + 1e-6), (-1.001e-6, 1.000001001), (-np.inf, np.inf)):
            for dtype in (np.float32, np.float64):
                data = np.array([[lo, hi]], dtype=dtype)
                ref = data.astype(np.float64)
                info = brightness.measure_brightness(data, True)
                self.assertEqual(info.min_pct, float(ref.min()) * 100)
                self.assertEqual(info.max_pct, float(ref.max()) * 100)
                self.assertEqual(info.would_clip, float(ref.min()) < -1e-6 or float(ref.max()) > 1 + 1e-6)
        info = brightness.measure_brightness(np.array([[0, np.nan]], dtype=np.float32), True)
        self.assertTrue(np.isnan(info.min_pct) and np.isnan(info.max_pct))
        self.assertFalse(info.would_clip)

    def test_histogram_preserves_bins_counts_and_sampling(self):
        rng = np.random.default_rng(8)
        # A small limit exercises sampling boundaries without large fixtures.
        with patch.object(histogram, '_MAX_SAMPLES', 250):
            for shape in ((10, 20), (10, 25), (17, 29), (17, 29, 3), (17, 29, 4), (0, 5, 3)):
                data = rng.uniform(-.1, 1.4, shape).astype(np.float32)
                for view in (data, data[::-1, ::2]):
                    for perceptual in (True, False):
                        with self.subTest(shape=view.shape, perceptual=perceptual):
                            before = view.copy()
                            expected = reference_histograms(view, perceptual)
                            with patch.object(histogram, 'linear_to_srgb', wraps=linear_to_srgb) as transfer:
                                result = histogram.compute_rgb_histograms(view, perceptual=perceptual)
                                if perceptual:
                                    self.assertLessEqual(transfer.call_args.args[0].shape[0], 250)
                            for actual, reference in zip(result, expected):
                                np.testing.assert_array_equal(actual, reference)
                            np.testing.assert_array_equal(view, before)

    def test_merge_resizes_luminance_once_with_identical_result(self):
        rng = np.random.default_rng(15)
        for gray in (True, False):
            main = rng.uniform(0, .9, (32, 44) if gray else (32, 44, 3)).astype(np.float32)
            for shape in ((25, 31), (25, 31, 1), (25, 31, 3)):
                secondary = rng.uniform(0, .9, shape).astype(np.float32)
                before = secondary.copy()
                def old_resize(channel, width, height):
                    rgb = np.stack([channel] * 3, axis=-1)
                    return scale_image(rgb, width, height)[..., 0]
                scale_image = scale.scale_image
                with patch.object(scale, 'scale_image', side_effect=old_resize):
                    expected = merge_wavelet_detail(main, secondary, gray)
                with patch.object(scale, '_resize_channel', wraps=scale._resize_channel) as resize:
                    actual = merge_wavelet_detail(main, secondary, gray)
                    self.assertEqual(resize.call_count, 1)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(secondary, before)

    def test_disabled_merge_skips_secondary_processing(self):
        main = np.ones((8, 12, 3), dtype=np.float32)
        secondary = np.zeros((40, 60, 3), dtype=np.float32)
        with patch.object(scale, 'scale_image') as resize, patch('planetary_tools.filters.wavelet._wavelet_decompose') as decompose:
            np.testing.assert_array_equal(merge_wavelet_detail(main, secondary, False, 0), main)
            resize.assert_not_called()
            decompose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
