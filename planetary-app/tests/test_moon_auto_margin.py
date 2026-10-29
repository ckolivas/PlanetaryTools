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
    def test_ringed_planet_uses_local_padding_and_no_artificial_rim_peaks(self):
        y, x = np.indices((384, 512))
        disk = ((x-256)**2 + (y-192)**2 < 70**2).astype(np.float32)
        rings = (((x-256)/160)**2 + ((y-192)/15)**2 < 1).astype(np.float32)
        image = (ndimage.gaussian_filter(np.maximum(disk*.6, rings*.35), 2)
                 + ndimage.gaussian_filter(disk*.03, 20))
        positions = ((152, 386), (70, 100), (300, 425))
        for pos, peak in zip(positions, (.01, .02, .0004)):
            image[pos] += peak
        seed, _ = moon._planet_seed(image, moon.DEFAULT_PLANET_FLOOR)
        distance = ndimage.distance_transform_edt(~seed)
        margin = moon.auto_planet_margin(image, True)
        # The disk's wider halo would hide the close moon above the rings if
        # its largest padding were imposed on every part of Saturn's outline.
        self.assertGreater(margin, distance[positions[0]])
        detections, exclusion = moon.detect_moons(
            image, sensitivity=10, planet_margin=0, max_moons=12,
        )
        self.assertEqual({(d.y, d.x) for d in detections}, set(positions))
        self.assertTrue(exclusion[seed].all())
        self.assertFalse(any(exclusion[p] for p in positions))
        rotated, rotated_mask = moon.detect_moons(
            np.rot90(image), sensitivity=10, planet_margin=0, max_moons=12,
        )
        self.assertEqual({(d.y, d.x) for d in rotated},
                         {(image.shape[1]-1-x, y) for y, x in positions})
        np.testing.assert_array_equal(rotated_mask, np.rot90(exclusion))

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

    def test_strong_trailed_moon_kept_and_border_peaks_ignored(self):
        y, x = np.indices((384, 384))
        image = planet()
        image += np.random.default_rng(1).normal(1e-3, 1e-4, image.shape).astype(np.float32)
        # A moon smeared along its orbit during a derotated stack.
        image += .003*np.exp(-((x-60)/5)**2/2 - ((y-60)/1.5)**2/2)
        image[300, 383] += .003
        detections, _ = moon.detect_moons(
            image, sensitivity=8, planet_margin=0, max_moons=12,
        )
        self.assertEqual([(d.y, d.x) for d in detections], [(60, 60)])

    def test_globe_split_off_by_ring_shadow_is_still_planet(self):
        y, x = np.indices((384, 512))
        disk = (x-256)**2 + (y-192)**2 < 70**2
        rings = ((x-256)/160)**2 + ((y-192)/15)**2 < 1
        image = np.where(disk, .6, 0.).astype(np.float32)
        image[rings] = .35
        # The ring shadow cuts the globe below the rings off from the rest.
        shadow = ((x-256)/160)**2 + ((y-192)/28)**2 < 1
        image[disk & shadow & ~rings & (y > 192)] = .005
        image = ndimage.gaussian_filter(image, 2)
        image += np.random.default_rng(3).normal(2e-4, 2e-5, image.shape).astype(np.float32)
        image[300, 425] += .004
        seed, _ = moon._planet_seed(image, moon.DEFAULT_PLANET_FLOOR)
        self.assertTrue(seed[255, 256])
        for margin in (0, 20):
            detections, exclusion = moon.detect_moons(
                image, sensitivity=8, planet_margin=margin, max_moons=1,
            )
            self.assertEqual([(d.y, d.x) for d in detections], [(300, 425)])
            self.assertTrue(exclusion[263, 256])

    def test_compact_moon_in_glow_fringe_kept_but_broad_knot_rejected(self):
        y, x = np.indices((384, 512))
        disk = ((x-256)**2 + (y-192)**2 < 70**2).astype(np.float32)
        image = ndimage.gaussian_filter(disk*.6, 2) + ndimage.gaussian_filter(disk*.04, 25)
        image += np.random.default_rng(5).normal(2e-4, 2e-5, image.shape).astype(np.float32)
        exclusion, fringe = moon._exclusion_zones(image, moon.DEFAULT_PLANET_FLOOR, 0)
        self.assertTrue(fringe.any())
        # Pick fringe points on opposite limbs, beyond the 20 px core.
        fys, fxs = np.nonzero(fringe & (np.abs(y-192) < 3))
        compact, broad = (192, int(fxs.min())+3), (192, int(fxs.max())-3)
        self.assertTrue(fringe[compact] and fringe[broad])
        image += .004*np.exp(-((x-compact[1])**2 + (y-compact[0])**2)/(2*1.8**2))
        image += .004*np.exp(-((x-broad[1])**2 + (y-broad[0])**2)/(2*7**2))
        detections, carved = moon.detect_moons(image, sensitivity=8, planet_margin=0, max_moons=12)
        # The glow gradient may shift the smoothed peak by a pixel toward the planet.
        self.assertEqual(len(detections), 1)
        found = (detections[0].y, detections[0].x)
        self.assertLessEqual(np.hypot(found[0]-compact[0], found[1]-compact[1]), 1.5)
        self.assertLess(detections[0].fwhm, 10)
        auto, _ = moon._exclusion_zones(image, moon.DEFAULT_PLANET_FLOOR, 0)
        opened = auto & ~carved
        self.assertTrue(opened[found])
        self.assertLessEqual(np.hypot(*(np.argwhere(opened) - found).T).max(), detections[0].fwhm)
        self.assertTrue(carved[broad])
        output = moon.moon_enhance(image, True, 25, 8, .5, 0, 12)
        np.testing.assert_array_equal(output[carved], image[carved])
        self.assertGreater(float(output[found]), float(image[found])*5)
        # A manual margin is exact and never opens a fringe.
        manual, manual_mask = moon.detect_moons(image, sensitivity=8, planet_margin=40, max_moons=12)
        self.assertEqual(manual, [])
        self.assertTrue(manual_mask[found])


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
        # Auto follows local disk/ring glow; its displayed maximum is a
        # conservative uniform manual margin, not the same mask.
        self.assertFalse(np.any(exclusion & ~manual))
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

    def test_trailed_moon_image(self):
        self.check_detections('2moons.png', ((236, 657), (261, 75)))

    def test_earlier_four_moon_image(self):
        self.check_detections('4moons.png', ((210, 357), (195, 621), (139, 1054), (103, 1744)))

    def test_enceladus_in_ring_glow(self):
        # Enceladus sits in the glow below the left ring, at the edge of Auto's
        # glow-following mask; only manual margins of 16 px or less used to keep it.
        self.check_detections('EnceladusTest.png', ((233, 143), (318, 560), (290, 290)))

    def test_globe_below_ring_shadow_is_not_a_moon(self):
        # The ring shadow splits the lower globe from the rings; its limb was
        # detected ahead of the real moon, which needed Max moons of 2.
        self.check_detections('1moon.png', ((172, 627),))
        path = ROOT / '1moon.png'
        from planetary_tools.io.loader import load_image
        doc = load_image(path, pin_noise=False)
        lum = moon._luminance(doc.data, doc.is_grayscale)
        for margin in (0, 40, 60):
            detections, _ = moon.detect_moons(lum, sensitivity=8, planet_margin=margin, max_moons=12)
            self.assertEqual([(d.y, d.x) for d in detections], [(172, 627)])

    def test_saturn_close_and_faint_moons(self):
        margin = self.check_detections('moontest.png', ((160, 620), (214, 528), (264, 694)))
        self.assertLess(margin, 28)

    def test_saturn_pseudo_relative_lifts_faint_moon_and_protects_planet(self):
        path = ROOT / 'moontest.png'
        if not path.exists():
            self.skipTest('Local moontest.png regression image is not installed')
        from planetary_tools.io.loader import load_image
        doc = load_image(path, pin_noise=False)
        lum = moon._luminance(doc.data, doc.is_grayscale)
        detections, exclusion = moon.detect_moons(
            lum, sensitivity=8, planet_margin=0, max_moons=3,
        )
        strict = moon.moon_enhance(doc.data, doc.is_grayscale, 25, 8, .5, 0, 3,
                                  retain_relative_brightness=True)
        pseudo = moon.moon_enhance(doc.data, doc.is_grayscale, 25, 8, .5, 0, 3,
                                  retain_relative_brightness=True, pseudo_relative_brightness=True)
        strict_lum = moon._luminance(strict, doc.is_grayscale)
        pseudo_lum = moon._luminance(pseudo, doc.is_grayscale)
        ordered = sorted(detections, key=lambda d: d.peak_l, reverse=True)
        peaks = [float(pseudo_lum[d.y-2:d.y+3, d.x-2:d.x+3].max()) for d in ordered]
        self.assertGreater(peaks[0], peaks[1])
        self.assertGreater(peaks[1], peaks[2])
        self.assertGreater(float(pseudo_lum[264, 694]), float(strict_lum[264, 694])*5)
        np.testing.assert_array_equal(pseudo[exclusion], doc.data[exclusion])


if __name__ == '__main__':
    unittest.main()
