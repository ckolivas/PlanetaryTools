"""Registration regressions; optional real-image checks use ../../align/*.png.

Run: python -m unittest discover -s tests -p 'test_alignment.py'
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, shift

from planetary_tools.core.align import align_channel, align_to_reference
from planetary_tools.core.field_derotate import (
    IDENTITY_MATCH, RigidMatch, apply_rigid, derotate_set, estimate_rigid,
)


def planet(shape=(160, 240)):
    y, x = np.indices(shape, dtype=float)
    y = y - (shape[0] - 1) / 2
    x = x - (shape[1] - 1) / 2
    disk = (x*x / 34**2 + y*y / 29**2 < 1).astype(float)
    disk *= .65 + .08 * np.cos(y / 3) + .03 * x / 34
    ring = np.exp(-((np.sqrt((x / 64)**2 + ((y + .16*x) / 10)**2) - 1) / .08)**2)
    return gaussian_filter(disk + .25 * ring, .8).astype(np.float32)


class AlignmentTests(unittest.TestCase):
    def test_fractional_channel_shifts_and_exposure(self):
        ref = planet()
        for delta in ((.12, -.21), (2.38, -3.71), (-4.25, 3.4)):
            with self.subTest(delta=delta):
                tgt = shift(ref, delta, order=3) * .7 + .02
                aligned = align_channel(ref, tgt)
                # Exposure changes are preserved; only coordinates change.
                expected = shift(tgt, tuple(-v for v in delta), order=3)
                np.testing.assert_allclose(aligned[25:-25, 25:-25],
                                           expected[25:-25, 25:-25], atol=.001)
                self.assertEqual(aligned.dtype, np.float32)

    def test_rgb_and_singleton_planes_remain_registered(self):
        ref = planet()
        tgt = shift(ref, (.34, -.57), order=3)
        rgb = np.stack([tgt, tgt*.7, tgt*.4], axis=-1)
        result = align_to_reference(ref, rgb)
        np.testing.assert_allclose(result[..., 1], result[..., 0]*.7, atol=1e-7)
        np.testing.assert_allclose(result[..., 2], result[..., 0]*.4, atol=1e-7)
        single = align_to_reference(ref, tgt[..., None])
        self.assertEqual(single.shape, tgt.shape + (1,))

    def test_blank_and_identity_frames_are_unchanged(self):
        for ref in (np.zeros((30, 40), np.float32), np.ones((30, 40), np.float32), planet()):
            np.testing.assert_array_equal(align_channel(ref, ref), ref)
        match = estimate_rigid(np.zeros((30, 40)), np.zeros((30, 40)), max_angle=1)
        self.assertEqual((match.angle_deg, match.dy, match.dx), (0, 0, 0))
        self.assertEqual(match.status, 'Weak match')

    def test_invalid_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'matching shapes'):
            align_channel(np.zeros((3, 4)), np.zeros((4, 3)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            align_channel(np.full((3, 4), np.nan), np.zeros((3, 4)))

    def test_joint_rotation_and_translation(self):
        ref = planet()
        angle, delta = 1.37, np.array([2.38, -3.71])
        tgt = shift(rotate(ref, angle, reshape=False, order=3), delta, order=3)
        match = estimate_rigid(ref, tgt, max_angle=3)
        theta = np.deg2rad(-angle)
        forward = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        expected = -forward @ delta
        self.assertAlmostEqual(match.angle_deg, -angle, delta=.015)
        np.testing.assert_allclose([match.dy, match.dx], expected, atol=.02)

    def test_shift_only_recovers_large_translation(self):
        ref = planet()
        match = estimate_rigid(ref, shift(ref, (13.38, -20.71), order=3), rotate=False)
        self.assertEqual(match.angle_deg, 0)
        np.testing.assert_allclose([match.dy, match.dx], [-13.38, 20.71], atol=.02)

    def test_moving_moon_does_not_drive_planet_alignment(self):
        ref = planet()
        y, x = np.indices(ref.shape)
        moon = .12 * np.exp(-((y-25)**2 + (x-25)**2) / 3)
        tgt = shift(ref, (.38, -.71), order=3)
        ref = ref + moon
        tgt = tgt + shift(moon, (7, 13), order=3)
        match = estimate_rigid(ref, tgt, rotate=False)
        np.testing.assert_allclose([match.dy, match.dx], [-.38, .71], atol=.03)

    def test_different_frame_sizes_share_reference_coordinates(self):
        ref = planet((161, 241))
        tgt = ref[10:-11, 12:-13]
        match = estimate_rigid(ref, tgt, rotate=False)
        np.testing.assert_allclose([match.dy, match.dx], [0, 0], atol=.01)
        saved, _ = self.export([(Path('ref.tif'), ref, IDENTITY_MATCH),
                                (Path('tgt.tif'), tgt, match)], subpixel=True)
        np.testing.assert_allclose(saved['ref_derot'], saved['tgt_derot'], atol=.001)

    def test_rotation_respects_search_limit(self):
        ref = planet()
        tgt = rotate(ref, 1.4, reshape=False, order=3)
        match = estimate_rigid(ref, tgt, max_angle=.35)
        self.assertLessEqual(abs(match.angle_deg), .35)
        self.assertEqual(match.status, 'Hit search limit')

    def export(self, items, **kwargs):
        saved = {}
        def save(doc, path, **_):
            saved[Path(path).stem] = doc.data
        with tempfile.TemporaryDirectory() as directory, patch(
            'planetary_tools.io.loader.save_image', side_effect=save,
        ):
            result = derotate_set(items, Path(directory), **kwargs)
        self.assertEqual(result.failed, [])
        self.assertEqual(result.processed, len(items))
        return saved, result

    def test_export_does_not_recentre_frames_by_brightness(self):
        ref = planet()
        tgt = ref.copy()
        # Change the brightness centroid without changing feature positions.
        tgt[:, 120:] *= .5
        saved, _ = self.export([(Path('target.tif'), tgt, IDENTITY_MATCH),
                                (Path('reference.tif'), ref, IDENTITY_MATCH)],
                               subpixel=True, ref_index=1)
        a, b = saved['reference_derot'], saved['target_derot']
        # Every unchanged source pixel must still coincide exactly.
        self.assertTrue(np.all((b == a) | (b == .5 * a)))
        self.assertEqual(np.count_nonzero(a), np.count_nonzero(ref))
        np.testing.assert_array_equal(np.sort(a[a != 0]), np.sort(ref[ref != 0]))

    def test_export_subpixel_and_rotation_use_common_coordinates(self):
        ref = planet()
        angle, delta = 1.37, np.array([2.38, -3.71])
        tgt = shift(rotate(ref, angle, reshape=False, order=3), delta, order=3)
        match = estimate_rigid(ref, tgt, max_angle=3)
        saved, _ = self.export([(Path('ref.tif'), ref, IDENTITY_MATCH),
                                (Path('tgt.tif'), tgt, match)], subpixel=True)
        np.testing.assert_allclose(saved['ref_derot'], saved['tgt_derot'], atol=.018)

    def test_invalid_transform_does_not_abort_other_exports(self):
        ref = planet()
        with tempfile.TemporaryDirectory() as directory, patch(
            'planetary_tools.io.loader.save_image',
        ):
            for subpixel in (False, True):
                result = derotate_set([
                    (Path('ref.tif'), ref, IDENTITY_MATCH),
                    (Path('bad.tif'), ref, RigidMatch(0, float('nan'), 0, 0)),
                ], Path(directory), subpixel=subpixel)
                self.assertEqual(result.processed, 1)
                self.assertEqual(len(result.failed), 1)
                self.assertEqual(result.failed[0][0], 'bad.tif')

    def test_integer_output_and_translation_keep_full_frame(self):
        ref = np.zeros((9, 12), dtype=np.float32)
        ref[0, 0], ref[-1, -1] = 1, .5
        moved = apply_rigid(ref, RigidMatch(0, 4, -5, 1))
        self.assertEqual(np.count_nonzero(moved), 2)
        self.assertEqual(float(moved.sum()), 1.5)
        _, result = self.export([(Path('ref.tif'), ref, IDENTITY_MATCH),
                                 (Path('tgt.tif'), ref, RigidMatch(0, .4, -.7, 1))],
                                subpixel=False)
        self.assertEqual((result.frames[1].match.dy, result.frames[1].match.dx), (0, -1))


class SaturnFixtureTests(unittest.TestCase):
    def test_known_transforms_of_previously_aligned_frames(self):
        from planetary_tools.io.loader import load_image
        files = sorted((Path(__file__).resolve().parents[2] / 'align').glob('*.png'))
        if not files:
            self.skipTest('Local align/ Saturn fixtures are not installed')
        for path in files:
            with self.subTest(file=path.name):
                ref = load_image(path).data
                tgt = shift(ref, (.37, -.64, 0), order=3)
                match = estimate_rigid(ref, tgt, rotate=False)
                np.testing.assert_allclose([match.dy, match.dx], [-.37, .64], atol=.02)


if __name__ == '__main__':
    unittest.main()
