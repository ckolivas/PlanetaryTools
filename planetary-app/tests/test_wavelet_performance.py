"""Auto-sharpen reuses Gaussian detail without changing trial results."""
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.core.colour import srgb_to_linear, linear_to_srgb
from planetary_tools.filters.wavelet import _unsharp_mask, gaussian_filter, wavelet_sharpen
from planetary_tools.filters.wavelet_auto import _SharpenTrialEngine


class WaveletPerformanceTests(unittest.TestCase):
    def test_unsharp_preserves_original_math_and_zero_amount(self):
        layer = np.random.default_rng(7).uniform(-.1, 1.3, (27,36)).astype(np.float32)
        linear = srgb_to_linear(layer, clamp=False).astype(np.float64)
        blur = gaussian_filter(linear, 16)
        for amount in (0, .1, 4, 40):
            expected = layer if amount == 0 else linear_to_srgb(linear+amount*(linear-blur), clamp=False)
            np.testing.assert_array_equal(_unsharp_mask(layer,16,amount), expected)

    def test_auto_trials_match_direct_sharpen_and_blur_only_once_per_scale(self):
        source = np.random.default_rng(3).uniform(.001, .9, (32,44,3)).astype(np.float32)
        for gray, luminance in ((False,False),(False,True),(True,False)):
            with self.subTest(gray=gray,luminance=luminance):
                image = source[...,0] if gray else source
                engine = _SharpenTrialEngine(image,gray,texture_scale=2,chromatic=False,luminance=luminance)
                with patch('planetary_tools.filters.wavelet.gaussian_filter',wraps=gaussian_filter) as blur:
                    results = [engine.apply(fine,2,1,0) for fine in (1,2,4,8,16,1,0)]
                    self.assertEqual(blur.call_count, 3 if gray or luminance else 9)
                for fine, result in zip((1,2,4,8,16,1,0),results):
                    np.testing.assert_array_equal(result,wavelet_sharpen(image,gray,fine,2,1,0,luminance=luminance))
                self.assertLessEqual(len(engine._usm_cache),12)
                self.assertLessEqual(len(engine._unsharp_prepared),12)


if __name__ == '__main__':
    unittest.main()
