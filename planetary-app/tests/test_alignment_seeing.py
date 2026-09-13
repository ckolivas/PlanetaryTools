"""Registration across unequal seeing, including optional realign/ captures."""
from pathlib import Path
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, shift

from planetary_tools.core.align import _refine_alignment, _vertical_limb_correction
from planetary_tools.core.field_derotate import estimate_rigid, luma
from test_alignment import planet


class SeeingAlignmentTests(unittest.TestCase):
    def assert_transform(self, reference, target, angle=1.37, delta=(2.38, -3.71)):
        moved = shift(rotate(target, angle, reshape=False, order=3), delta, order=3)
        match = estimate_rigid(reference, moved, max_angle=3)
        theta = np.deg2rad(-angle)
        matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        self.assertAlmostEqual(match.angle_deg, -angle, delta=.02)
        np.testing.assert_allclose([match.dy, match.dx], -matrix @ delta, atol=.025)
        self.assertEqual(match.status, 'OK')

    def test_unequal_seeing_in_either_reference_direction(self):
        sharp = planet()
        noise = .002 * np.random.default_rng(28).normal(size=sharp.shape)
        for sigma in ((3, .5), (.5, 3), (4, 1), (1, 4), (4, 4)):
            soft = .7 * gaussian_filter(sharp, sigma) + noise
            for reverse in (False, True):
                with self.subTest(sigma=sigma, reverse=reverse):
                    ref, tgt = (soft, sharp) if reverse else (sharp, soft)
                    self.assert_transform(ref, tgt)

    def test_seeing_with_single_pixel_moons_and_large_sky_canvas(self):
        ref = np.pad(planet(), ((75, 75), (90, 90)))
        tgt = gaussian_filter(ref, (4, 1))
        ref[20, 25] = 1
        tgt[27, 38] = 1  # A moving moon must not set the planet's alignment.
        self.assert_transform(ref, tgt)

    def test_fractional_shift_only_with_unequal_seeing(self):
        ref = planet()
        tgt = shift(gaussian_filter(ref, (4, 1)), (.23, -.41), order=3)
        match = estimate_rigid(ref, tgt, rotate=False)
        self.assertEqual(match.angle_deg, 0)
        np.testing.assert_allclose([match.dy, match.dx], [-.23, .41], atol=.02)

    def test_vertical_alignment_follows_disk_when_bright_ring_structure_changes(self):
        disk = planet()
        y, x = np.indices(disk.shape, dtype=float)
        y -= (disk.shape[0] - 1) / 2
        x -= (disk.shape[1] - 1) / 2
        ring = .8 * np.exp(-((y + .16*x) / 2)**2) * np.exp(-(x / 70)**8)
        reference = disk + ring
        for ring_shift in (.8, 1.5, -1.5):
            with self.subTest(ring_shift=ring_shift):
                # Seeing/processing can change the apparent bright ring ridge
                # independently of the disk outline. The disk moves by .65 px.
                target = disk + shift(ring, (ring_shift, 0), order=3)
                target = shift(gaussian_filter(target, (2, 1)), (.65, -.4), order=3)
                match = estimate_rigid(reference, target, rotate=False)
                self.assertAlmostEqual(match.dy, -.65, delta=.02)

    def test_limb_refinement_requires_both_complete_disk_edges(self):
        for reference in (np.zeros((60, 80)), np.ones((60, 80)),
                          planet()[65:95], np.eye(60)):
            with self.subTest(shape=reference.shape):
                self.assertEqual(_vertical_limb_correction(
                    reference, shift(reference, (.7, 0)),
                ), 0)

    def test_limb_refinement_rejects_a_missing_target(self):
        reference = planet()
        self.assertEqual(_vertical_limb_correction(reference, np.zeros_like(reference)), 0)

    def test_refinement_can_leave_integer_seed_with_nonzero_edges(self):
        ref = planet() + .002
        tgt = shift(rotate(ref, 1.37, reshape=False, order=3), (2.38, -3.71), order=3)
        match = _refine_alignment(ref, tgt, (0, -2, 4), max_angle=3, angle_span=2)
        self.assertAlmostEqual(match[0], -1.37, delta=.02)
        np.testing.assert_allclose(match[1:3], [-2.2906, 3.7659], atol=.025)
        identity = _refine_alignment(ref, ref, (0, 0, 0), max_angle=3, angle_span=2)
        np.testing.assert_allclose(identity, [0, 0, 0, 1], atol=1e-8)

    def test_known_transforms_across_real_capture_qualities(self):
        from planetary_tools.io.loader import load_image

        files = sorted((Path(__file__).resolve().parents[2] / 'realign').glob('*.png'))
        if not files:
            self.skipTest('Local realign/ Saturn fixtures are not installed')
        for i, path in enumerate(files):
            with self.subTest(file=path.name):
                ref = luma(load_image(path, pin_noise=False).data)
                # Add known motion and unequal blur to each capture; the
                # supplied inter-frame offsets themselves are not ground truth.
                soft = .8 * gaussian_filter(ref, (3, 1))
                if i % 2:
                    ref, soft = soft, ref
                self.assert_transform(ref, soft)

    def test_asymmetric_glow_does_not_move_the_disk(self):
        disk = planet()
        for amount in (.1, .25, .4):
            for direction in (-1, 1):
                halo = shift(gaussian_filter(disk, 8), (12 * direction, 0), order=3)
                glowing = disk + amount * halo
                for reverse in (False, True):
                    with self.subTest(amount=amount, direction=direction, reverse=reverse):
                        ref, tgt = (glowing, disk) if reverse else (disk, glowing)
                        moved = shift(tgt, (.65, -.4), order=3)
                        match = estimate_rigid(ref, moved, rotate=False)
                        self.assertAlmostEqual(match.dy, -.65, delta=.03)

    def test_real_captures_with_known_motion_and_one_sided_glow(self):
        from planetary_tools.io.loader import load_image

        directory = Path(__file__).resolve().parents[2] / 'realign'
        files = sorted(directory.glob('2026-09-11-*-Sat_*.png'))
        if not files:
            self.skipTest('Local realign/ Saturn fixtures are not installed')
        # Isophote midpoints follow defocused glow; they are not ground truth
        # for the original sequence. Inject a known translation and halo instead.
        for path in files:
            ref = luma(load_image(path, pin_noise=False).data)
            glowing = ref + .3 * shift(gaussian_filter(ref, 10), (12, 0), order=3)
            for reverse in (False, True):
                with self.subTest(file=path.name, reverse=reverse):
                    a, b = (glowing, ref) if reverse else (ref, glowing)
                    match = estimate_rigid(a, shift(b, (.65, -.4), order=3), rotate=False)
                    # Require sub-tenth-pixel recovery despite the added halo;
                    # the original unperturbed captures have no ground truth.
                    self.assertAlmostEqual(match.dy, -.65, delta=.1)


if __name__ == '__main__':
    unittest.main()
