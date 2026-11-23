"""Correct Dispersion rescales red and blue to green along the dispersion axis."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import affine_transform

import planetary_tools.core.align as align
from planetary_tools.core.align import align_rgb, dispersion_axis
from test_alignment import planet


def stretched(plane, scale, shift, axis=(1.0, 0.0)):
    """Magnify ``plane`` by ``scale`` along ``axis`` about the centre, then shift."""
    u = np.asarray(axis, dtype=np.float64)
    inverse = np.eye(2) + (1 / scale - 1) * np.outer(u, u)
    centre = (np.array(plane.shape) - 1) / 2
    return affine_transform(plane.astype(np.float64), inverse,
                            offset=centre - inverse @ (centre + shift), order=3).astype(np.float32)


def dispersed(green, red_scale, blue_scale, axis=(1.0, 0.0)):
    u = np.asarray(axis)
    return np.stack([stretched(green, red_scale, -3.2 * u + [.2, -.1], axis), green,
                     stretched(green, blue_scale, 4.1 * u + [-.1, .2], axis)], axis=-1)


class DispersionTests(unittest.TestCase):
    def fitted(self, data, **kwargs):
        fits = []
        real = align.fit_dispersion_scale
        with patch.object(align, 'fit_dispersion_scale',
                          side_effect=lambda *a: fits.append(real(*a)) or fits[-1]):
            out = align_rgb(data, dispersion=True, **kwargs)
        return out, fits

    def test_axis_follows_red_to_blue_and_needs_a_visible_spread(self):
        np.testing.assert_allclose(dispersion_axis((-3, 0), (4, 0)), (1, 0))
        np.testing.assert_allclose(dispersion_axis((2, -2), (-1, 1)), np.array([-1, 1]) / np.sqrt(2))
        self.assertIsNone(dispersion_axis((.1, .2), (-.1, -.1)))

    def test_red_and_blue_are_resized_to_green_along_the_axis(self):
        green = planet()
        for axis in ((1.0, 0.0), (np.sqrt(.5), np.sqrt(.5))):
            with self.subTest(axis=axis):
                data = dispersed(green, 1.03, .98, axis)
                out, fits = self.fitted(data)
                self.assertAlmostEqual(fits[0][0], 1 / 1.03, delta=.002)
                self.assertAlmostEqual(fits[1][0], 1 / .98, delta=.002)
                np.testing.assert_array_equal(out[..., 1], green)
                rigid = align_rgb(data)
                for c in (0, 2):
                    corrected = np.abs(out[..., c] - green).mean()
                    self.assertLess(corrected, .3 * np.abs(rigid[..., c] - green).mean())

    def test_equal_sizes_are_left_alone(self):
        green = planet()
        data = dispersed(green, 1.0, 1.0)
        out, fits = self.fitted(data)
        for scale, _dy, _dx in fits:
            self.assertAlmostEqual(scale, 1.0, delta=.002)

    def test_no_visible_dispersion_matches_plain_alignment(self):
        green = planet()
        data = np.stack([stretched(green, 1.03, (.1, 0)), green, stretched(green, .98, (-.1, 0))], -1)
        out, fits = self.fitted(data)
        self.assertEqual(fits, [])
        np.testing.assert_array_equal(out, align_rgb(data))


if __name__ == '__main__':
    unittest.main()
