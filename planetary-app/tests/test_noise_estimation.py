"""Noise readouts must distinguish smooth structure from amplified grain."""
from pathlib import Path
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from planetary_tools.core import noise
from planetary_tools.filters.registry import output_filter_stats


class NoiseEstimationTests(unittest.TestCase):
    def test_sparse_tail_is_finite_and_increasing_bulk_noise_cannot_reduce_it(self):
        # Fixed rare speckles with progressively stronger bulk noise. The old
        # p99/MAD multiplier exploded at zero MAD and fell as noise was added.
        scores = []
        for amplitude in (0.0, .0001, .001, .01):
            values = np.tile([-amplitude, amplitude], 5000)
            values[:100] = -.1
            values[-100:] = .1
            scores.append(noise._residual_noise_level(values, noise._TAIL_WEIGHT))
        self.assertGreater(scores[0], 0)
        self.assertLess(scores[0], .1)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.diff(scores) > 0), scores)

    def test_smooth_curvature_is_not_large_noise_and_added_grain_is_measured(self):
        y, x = np.mgrid[-1:1:160j, -1:1:192j]
        surface = (.2 + .2*x*x + .15*y*y + .04*x*y).astype(np.float32)
        grain = np.random.default_rng(413).standard_normal(surface.shape).astype(np.float32)
        for texture in (2, 4, 6):
            for chromatic in (False, True):
                with self.subTest(texture=texture, chromatic=chromatic):
                    scores = [noise.absolute_noise(
                        surface + amount*grain, True,
                        texture_scale=texture, chromatic=chromatic,
                    ) for amount in (0, .0005, .001, .003, .01)]
                    # The old estimator reported >100,000 on this noiseless
                    # quadratic surface because its residual MAD approached 0.
                    self.assertLess(scores[0], 1)
                    self.assertTrue(np.all(np.diff(scores) > 0), scores)

    def test_sharpening_raises_noise_but_denoising_can_lower_it(self):
        y, x = np.mgrid[-1:1:160j, -1:1:192j]
        grain = np.random.default_rng(413).standard_normal(x.shape)
        image = (.02 + .3*np.exp(-3*(x*x+y*y))
                 + .07*np.exp(-(y+.12*x)**2/.035-x*x/.8)
                 + .0003*grain).astype(np.float32)
        texture = noise.estimate_texture_scale(image, True)
        scores = []
        for amount in (0, 1, 5, 10, 20):
            stats = output_filter_stats('wavelet_sharpen', image, True, {
                'fine': amount, 'medium': amount/2, 'coarse': 0, 'chunky': 0,
            }, texture_scale=texture, chromatic=False)
            scores.append(stats.noise_level)
            self.assertAlmostEqual(stats.source_noise_level, scores[0])
        self.assertTrue(np.all(np.diff(scores) > 0), scores)
        denoised = noise.absolute_noise(gaussian_filter(image, 1), True,
                                       texture_scale=texture, chromatic=False)
        self.assertLess(denoised, scores[0])

    def test_global_gain_invariance_and_grayscale_rgb_agreement(self):
        image = (.3 + .002*np.random.default_rng(15).standard_normal((80, 90))).astype(np.float32)
        expected = noise.absolute_noise(image, True, texture_scale=4)
        for gain in (.25, 2):
            self.assertAlmostEqual(noise.absolute_noise(image*gain, True, texture_scale=4),
                                   expected, places=6)
        rgb = np.repeat(image[..., None], 3, axis=-1)
        self.assertAlmostEqual(noise.absolute_noise(rgb, False, texture_scale=4),
                               expected, places=4)


STACK = Path(__file__).resolve().parents[2] / 'PT_noise.png'


@unittest.skipUnless(STACK.exists(), 'Local PT_noise.png regression stack is not installed')
class NoiseStackRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from planetary_tools.io.loader import load_image
        cls.doc = load_image(STACK)
        cls.texture, cls.chromatic = cls.doc.noise_context()

    def test_stack_baseline_and_sharpening_readouts(self):
        for filter_id in ('wavelet_sharpen', 'adaptive_deconv'):
            scores = []
            for amount in (0, 5, 10, 20, 50):
                params = ({'fine': amount, 'medium': amount/2, 'coarse': 0, 'chunky': 0}
                          if filter_id == 'wavelet_sharpen' else {'amount': amount})
                stats = output_filter_stats(
                    filter_id, self.doc.data, self.doc.is_grayscale, params,
                    texture_scale=self.texture, chromatic=self.chromatic,
                )
                scores.append(stats.noise_level)
                self.assertAlmostEqual(stats.source_noise_level, scores[0])
            with self.subTest(filter=filter_id):
                self.assertLess(scores[0], 1)
                self.assertTrue(np.all(np.diff(scores) > 0), scores)

    def test_auto_deconvolution_uses_the_same_corrected_score(self):
        from planetary_tools.filters.adaptive_deconv_auto import auto_adaptive_deconv_params
        result = auto_adaptive_deconv_params(
            self.doc.data, self.doc.is_grayscale, target_noise=.22,
            target_contrast=100, max_amount=20,
            texture_scale=self.texture, chromatic=self.chromatic,
        )
        self.assertGreater(result.amount, 0)
        self.assertLess(result.amount, 20)
        stats = output_filter_stats(
            'adaptive_deconv', self.doc.data, self.doc.is_grayscale,
            {'amount': result.amount}, texture_scale=self.texture, chromatic=self.chromatic,
        )
        self.assertAlmostEqual(result.noise, stats.noise_level, places=6)
        self.assertLessEqual(result.noise, .22 + 1e-6)


if __name__ == '__main__':
    unittest.main()
