"""Automatic moon exclusion should follow the halo, not just planet size."""
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from planetary_tools.filters import moon_enhance as moon


def planet(sigma=2):
    y, x = np.indices((384, 384))
    disk = (x-192)**2 + (y-192)**2 < 75**2
    return ndimage.gaussian_filter(disk.astype(np.float32)*.6, sigma)


class MoonAutoMarginTests(unittest.TestCase):
    def test_sharp_planet_keeps_close_single_pixel_moon(self):
        image = planet()
        positions = {(45, 192), (192, 45), (192, 302), (335, 192)}
        for y, x in positions:
            image[y, x] = .01
        detections, exclusion = moon.detect_moons(
            image, sensitivity=8, planet_margin=0, max_moons=4,
        )
        self.assertEqual({(d.y, d.x) for d in detections}, positions)
        mask, radius = moon._planet_seed(image, moon.DEFAULT_PLANET_FLOOR)
        distance = ndimage.distance_transform_edt(~mask)
        # The old half-radius padding swallowed this close moon.
        self.assertLess(distance[192, 302], moon._auto_margin_px(radius))
        margin = moon.auto_planet_margin(image, True)
        self.assertGreater(margin, 0)
        self.assertLess(margin, distance[192, 302])
        np.testing.assert_array_equal(exclusion, distance <= margin)

    def test_blurred_halos_need_more_padding_and_moons_do_not_expand_it(self):
        sharp, soft = planet(2), planet(15)
        sharp_margin = moon.auto_planet_margin(sharp, True)
        soft_margin = moon.auto_planet_margin(soft, True)
        self.assertGreater(soft_margin, sharp_margin)
        with_moons = sharp.copy()
        # Bright compact objects in both the halo and sky measurement bands.
        with_moons[192, 295] = .5
        with_moons[192, 320] = .5
        self.assertEqual(moon.auto_planet_margin(with_moons, True), sharp_margin)
        self.assertEqual(moon.auto_planet_margin(sharp*.25, True), sharp_margin)

    def test_manual_margin_is_exact_and_tight_crop_uses_conservative_fallback(self):
        image = planet()
        seed, _ = moon._planet_seed(image, moon.DEFAULT_PLANET_FLOOR)
        distance = ndimage.distance_transform_edt(~seed)
        for margin in (1, 27, 50):
            mask, _ = moon._planet_exclusion(image, moon.DEFAULT_PLANET_FLOOR, margin)
            np.testing.assert_array_equal(mask, distance <= margin)
        tight = image[150:235, 150:235]
        _, radius = moon._planet_seed(tight, moon.DEFAULT_PLANET_FLOOR)
        self.assertEqual(moon.auto_planet_margin(tight, True), moon._auto_margin_px(radius))
        self.assertEqual(moon.auto_planet_margin(np.zeros((32, 32)), True), 20)


ROOT = Path(__file__).resolve().parents[2]


class MoonImageMarginTests(unittest.TestCase):
    def check_detections(self, filename, expected):
        path = ROOT / filename
        if not path.exists():
            self.skipTest(f'Local {filename} regression image is not installed')
        from planetary_tools.io.loader import load_image
        doc = load_image(path, pin_noise=False)
        lum = moon._luminance(doc.data, doc.is_grayscale)
        margin = moon.auto_planet_margin(doc.data, doc.is_grayscale)
        detections, exclusion = moon.detect_moons(
            lum, sensitivity=8, planet_margin=0, max_moons=len(expected),
        )
        self.assertEqual(len(detections), len(expected))
        for y, x in expected:
            self.assertTrue(any(np.hypot(d.y-y, d.x-x) <= 2 for d in detections),
                            (filename, margin, detections))
        manual, _ = moon._planet_exclusion(lum, moon.DEFAULT_PLANET_FLOOR, margin)
        np.testing.assert_array_equal(exclusion, manual)
        output = moon.moon_enhance(doc.data, doc.is_grayscale, 25, 8, .5, 0, len(expected))
        np.testing.assert_array_equal(output[exclusion], doc.data[exclusion])
        for d in detections:
            self.assertGreater(float(output[d.y, d.x].max()), float(doc.data[d.y, d.x].max()))
        return margin

    def test_pt_four_moons_including_innermost(self):
        margin = self.check_detections('PT_4moons.png', ((181, 625), (151, 681), (192, 749), (244, 655)))
        self.assertLessEqual(margin, 27)

    def test_earlier_three_moon_image(self):
        self.check_detections('3moons.png', ((128, 459), (280, 100), (235, 647)))

    def test_earlier_four_moon_image(self):
        self.check_detections('4moons.png', ((210, 357), (195, 621), (139, 1054), (103, 1744)))


if __name__ == '__main__':
    unittest.main()
