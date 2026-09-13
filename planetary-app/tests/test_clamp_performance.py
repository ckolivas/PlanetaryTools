"""Single-buffer postprocessing preserves pixels, tolerances and ownership."""
import itertools
import unittest

import numpy as np

from planetary_tools.core import brightness


def reference(data, gray, clip_black, clamp_high, clamp_low):
    out = np.asarray(data, dtype=np.float32)
    def selected():
        return out[..., 0] if gray and out.ndim != 2 else out
    if clip_black and float(selected().min()) < -1e-6:
        if float(out.min()) < 0:
            out = np.maximum(out, 0.)
        out = out.astype(np.float32)
    if clamp_high and float(selected().max()) > 1 + 1e-6:
        if clamp_low:
            lo, hi = float(selected().min()), float(selected().max())
            span = hi-lo
            if hi > 1 + 1e-6 and span > 1e-6:
                out = (out-lo)/span
        else:
            peak = float(selected().max())
            if peak > 1 + 1e-6:
                out = out/peak
        out = out.astype(np.float32)
    return out.astype(np.float32)


class ClampPerformanceTests(unittest.TestCase):
    def test_all_options_match_original_and_own_their_result(self):
        source = np.random.default_rng(91).uniform(-.2, 1.6, (17, 23, 3))
        for data in (source.astype(np.float32), source, source.astype(np.float32)[::-1, ::2], source[..., 0].astype(np.float32), np.ones((5, 7, 3), dtype=np.uint16)):
            for gray in (False, True):
                for black, high, low in itertools.product((False, True), repeat=3):
                    with self.subTest(shape=data.shape, gray=gray, options=(black,high,low)):
                        before = data.copy()
                        data.setflags(write=False)
                        expected = reference(data, gray, black, high, low)
                        actual = brightness.apply_channel_post_process(data, gray, clip_black=black, clamp_high=high, clamp_low=low)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(data, before)
                        self.assertFalse(np.shares_memory(data, actual))
                        self.assertEqual(actual.dtype, np.float32)
                        actual[:] = 42
                        np.testing.assert_array_equal(data, before)

    def test_thresholds_nonfinite_and_grayscale_channel_gates(self):
        fixtures = [
            [[-1e-6, 1+1e-6]], [[-1.001e-6, 1.0000011]], [[3., 3.]],
            [[-np.inf, np.inf]], [[np.nan, 2.]], [[-3e38, 3e38]],
            [[[.5, -1., 2.], [.75, 2., -1.]]],
            [[[-.5, np.nan, 2.], [.75, 2., -1.]]],
        ]
        for values in fixtures:
            data = np.array(values, dtype=np.float32)
            for gray, black, high, low in itertools.product((False, True), repeat=4):
                with self.subTest(values=values, gray=gray, options=(black,high,low)), np.errstate(all='ignore'):
                    np.testing.assert_array_equal(
                        brightness.apply_channel_post_process(data, gray, clip_black=black, clamp_high=high, clamp_low=low),
                        reference(data, gray, black, high, low),
                    )

    def test_standalone_helpers_preserve_original_math_and_noop_copies(self):
        rng = np.random.default_rng(7)
        for source in (rng.uniform(-.1, 2, (11, 19, 3)), np.full((5, 7, 3), .5), np.array([[[-1e-8, .5, 2.]]])):
            data = source.astype(np.float32)
            before = data.copy()
            for gray in (False, True):
                selected = data[..., 0] if gray else data
                clipped = np.maximum(data, 0) if float(data.min()) < 0 else data.copy()
                peak = float(selected.max())
                hi_expected = data/peak if peak > 1+1e-6 else data.copy()
                lo, hi = float(selected.min()), float(selected.max())
                range_expected = (data-lo)/(hi-lo) if hi > 1+1e-6 and hi-lo > 1e-6 else data.copy()
                for fn, expected in ((brightness.clip_black_channels, clipped), (brightness.clamp_high_channels, hi_expected), (brightness.clamp_range_channels, range_expected)):
                    actual = fn(data, gray)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertFalse(np.shares_memory(actual, data))
                    np.testing.assert_array_equal(data, before)


if __name__ == '__main__':
    unittest.main()
