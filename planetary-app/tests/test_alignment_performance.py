"""Reference FFT reuse must preserve correlation, angle choices and output."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import rotate, shift

from planetary_tools.core import field_derotate as align
from test_alignment import planet


def original_correlation(reference, target):
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError('phase_correlation_shift requires matching shapes.')
    a, b = ref - ref.mean(), tgt - tgt.mean()
    na, nb = float(np.sqrt(np.sum(a*a))), float(np.sqrt(np.sum(b*b)))
    if na < 1e-12 or nb < 1e-12:
        return 0, 0, 0.
    corr = np.fft.ifft2(np.fft.fft2(a)*np.conj(np.fft.fft2(b))).real
    corr /= na*nb
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = map(int, peak)
    h, w = corr.shape
    return dy-h if dy > h//2 else dy, dx-w if dx > w//2 else dx, float(corr[peak])


class UncachedCorrelation:
    def __init__(self, reference):
        self.reference = reference

    def shift(self, target):
        return original_correlation(self.reference, target)


class AlignmentPerformanceTests(unittest.TestCase):
    def test_prepared_correlation_preserves_scores_wraps_and_weak_matches(self):
        rng = np.random.default_rng(61)
        for shape in ((31, 47), (32, 48)):
            for dtype in (np.float32, np.float64):
                source = rng.normal(size=shape).astype(dtype)
                for reference in (source, source[::-1, ::2], np.asfortranarray(source), source*1e-15, np.zeros_like(source)):
                    reference.setflags(write=False)
                    prepared = align._PreparedPhaseCorrelation(reference)
                    for delta in ((0,0), (7,-9), (shape[0]//2,shape[1]//2)):
                        target = np.roll(reference, delta, axis=(0,1))*.7 + .02
                        expected = original_correlation(reference, target)
                        self.assertEqual(prepared.shift(target), expected)
                        self.assertEqual(align.phase_correlation_shift(reference, target), expected)
                    with self.assertRaisesRegex(ValueError, 'matching shapes'):
                        prepared.shift(np.zeros((2, 3)))

    def test_angle_sweep_uses_one_reference_fft_without_mutating_source(self):
        source = planet((81, 113))
        before = source.copy()
        target = rotate(source, 1.3, reshape=False, order=1)
        angles = np.arange(-2., 2.1, .5)
        with patch.object(align, '_PreparedPhaseCorrelation', UncachedCorrelation):
            expected = align._best_angle(source, target, angles)
        with patch.object(np.fft, 'fft2', wraps=np.fft.fft2) as fft:
            actual = align._best_angle(source, target, angles)
        self.assertEqual(actual, expected)
        self.assertEqual(fft.call_count, len(angles)+1)
        np.testing.assert_array_equal(source, before)

    def test_rival_angle_and_empty_sweep_keep_original_choices(self):
        source = planet((63, 83))
        # Opposite orientations exercise the same rival ordering/tie logic.
        for target in (source, source[::-1, ::-1], np.zeros_like(source)):
            for angles in (np.array([]), np.array([-180., -179.9, 0., .1, 180.])):
                with patch.object(align, '_PreparedPhaseCorrelation', UncachedCorrelation):
                    expected = align._best_angle(source, target, angles)
                self.assertEqual(align._best_angle(source, target, angles), expected)

    def test_complete_search_match_and_render_remain_exact(self):
        source = planet((101, 143))
        target = shift(rotate(source, 1.37, reshape=False, order=3), (.38, -.71), order=3)
        for rotate_enabled, max_angle in ((False,3.), (True,3.), (True,.35)):
            with patch.object(align, '_PreparedPhaseCorrelation', UncachedCorrelation):
                expected = align.estimate_rigid(source, target, rotate=rotate_enabled, max_angle=max_angle)
            actual = align.estimate_rigid(source, target, rotate=rotate_enabled, max_angle=max_angle)
            self.assertEqual(asdict(actual), asdict(expected))
            np.testing.assert_array_equal(align.apply_rigid(target, actual), align.apply_rigid(target, expected))


if __name__ == '__main__':
    unittest.main()
