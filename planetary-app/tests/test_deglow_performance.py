"""Median reuse keeps Deglow's channel models and compact-source behavior."""
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.filters import deglow as module
from test_deglow import scene


class DeglowPerformanceTests(unittest.TestCase):
    def test_grayscale_matches_independently_filtered_colour_first_channel(self):
        mono, _, _ = scene()
        mono[64, 130] += 2.  # Isolated source brighter than the planet.
        mono[95, 191] += .01
        for source in (mono, mono[::-1, ::2], np.asfortranarray(mono)):
            for radius in (2, 30, 80):
                rgb = np.stack([source, source*.7, source*.4], axis=-1)
                # Use first-channel luminance but the colour model path,
                # which independently median-filters every colour channel.
                with patch.object(module, 'linear_luminance', side_effect=lambda data: data[...,0]):
                    expected = module.deglow(rgb, False, radius=radius)
                for data in (source, source[...,None], rgb):
                    before = data.copy()
                    data.setflags(write=False)
                    actual = module.deglow(data, True, radius=radius)
                    target = expected[...,0] if data.ndim == 2 else expected[...,:data.shape[2]]
                    np.testing.assert_array_equal(actual, target)
                    np.testing.assert_array_equal(data, before)

    def test_only_first_mono_channel_reuses_peak_median(self):
        mono, _, _ = scene()
        rgba = np.stack([mono, mono*.7, mono*.4, np.full_like(mono,.8)], axis=-1)
        for data, gray, full_count in ((mono,False,1), (mono[...,None],False,1),
                                      (rgba,True,3), (rgba,False,4)):
            with patch.object(module.ndimage, 'median_filter', wraps=module.ndimage.median_filter) as median:
                actual = module.deglow(data, gray)
            full_calls = [call for call in median.call_args_list if call.args[0].shape == mono.shape]
            self.assertEqual(len(full_calls), full_count)
            if data.ndim == 3 and data.shape[2] == 4:
                np.testing.assert_array_equal(actual[...,3], data[...,3])

    def test_identical_rgb_channels_share_model_but_near_gray_does_not(self):
        mono, _, _ = scene()
        for changed in ('identical', 'one_ulp', 'signed_zero'):
            rgb = np.stack([mono, mono, mono], axis=-1)
            if changed == 'one_ulp':
                # One ULP in one pixel must prevent sharing that channel.
                rgb[100,150,1] = np.nextafter(rgb[100,150,1], np.float32(1))
            elif changed == 'signed_zero':
                rgb[100,150] = [0., -0., 0.]
            with patch.object(module.np, 'array_equal', return_value=False):
                expected = module.deglow(rgb, False)
            with patch.object(module.ndimage, 'median_filter', wraps=module.ndimage.median_filter) as median:
                actual = module.deglow(rgb, False)
            full_calls = [call for call in median.call_args_list if call.args[0].shape == mono.shape]
            self.assertEqual(len(full_calls), 2 if changed == 'identical' else 4)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == '__main__':
    unittest.main()
