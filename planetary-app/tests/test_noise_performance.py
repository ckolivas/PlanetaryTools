"""Noise preparation optimizations must preserve scores and signal selection."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.core import noise
from planetary_tools.core.colour import linear_luminance


def reference_luminance(data, gray):
    arr = np.asarray(data, dtype=np.float64)
    if gray or arr.ndim == 2:
        return arr if arr.ndim == 2 else arr[..., 0]
    return linear_luminance(arr).astype(np.float64)


def reference_crop(lum):
    if lum.size == 0:
        return lum
    peak = float(np.percentile(lum, 99.))
    floor = max(noise._SIGNAL_PEAK_FRACTION * peak, noise._SIGNAL_ABS_FLOOR)
    ys, xs = np.where(lum >= floor)
    if ys.size < noise._MIN_SAMPLES:
        return lum
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    py = max(noise._BBOX_PAD_MIN, round((y1-y0+1)*noise._BBOX_PAD_FRAC))
    px = max(noise._BBOX_PAD_MIN, round((x1-x0+1)*noise._BBOX_PAD_FRAC))
    return lum[max(0,y0-py):min(lum.shape[0],y1+py+1), max(0,x0-px):min(lum.shape[1],x1+px+1)]


def reference_chromatic(data, gray):
    if gray:
        return False
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        return False
    r,g,b = arr[...,0],arr[...,1],arr[...,2]
    mx = np.maximum(np.maximum(r,g),b)
    mn = np.minimum(np.minimum(r,g),b)
    mask = mx > .05 * max(float(mx.max()), 1e-12)
    if int(mask.sum()) < noise._MIN_SAMPLES:
        return False
    sat = (mx-mn)/(mx+1e-8)
    return float(np.median(sat[mask])) > noise._CHROMA_SAT_THRESHOLD


class NoisePerformanceTests(unittest.TestCase):
    def test_luminance_preserves_precision_and_bounds_rgb_blocks(self):
        source = np.random.default_rng(64).uniform(-.01, 1.1, (23, 31, 4))
        for data in (source, source.astype(np.float32), source[::-1, ::2], np.asfortranarray(source), source[...,0], source[..., :3].astype(np.float16)):
            for gray in (False, True):
                before = data.copy()
                data.setflags(write=False)
                expected = reference_luminance(data, gray)
                with patch.object(noise, '_LUMINANCE_BLOCK_PIXELS', 97), patch.object(noise, 'linear_luminance', wraps=linear_luminance) as convert:
                    actual = noise._luminance(data, gray)
                    if not gray and data.ndim == 3:
                        self.assertTrue(all(call.args[0].size <= 97*3 for call in convert.call_args_list))
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                np.testing.assert_array_equal(data, before)

    def test_subject_bounds_match_sparse_dense_and_threshold_cases(self):
        rng = np.random.default_rng(22)
        sparse = np.zeros((70, 110), dtype=np.float64)
        sparse[10:18, 40:48] = 1  # Exactly the minimum number of signal pixels.
        almost = sparse.copy(); almost[10,40] = 0
        scattered = np.zeros_like(sparse); scattered[::3, ::5] = 1
        for lum in (sparse, almost, scattered, np.ones_like(sparse), rng.random((70,110)), sparse[::-1, ::-1], np.empty((0,4)), np.full((12,15),np.nan)):
            before = lum.copy()
            np.testing.assert_array_equal(noise._crop_to_subject(lum), reference_crop(lum))
            np.testing.assert_array_equal(lum, before)

    def test_colour_detection_preserves_signal_and_saturation_thresholds(self):
        for dtype in (np.float32, np.float64):
            for peak in (.1, .9, 1.0):
                cutoff = dtype(.05*float(dtype(peak)))
                for level in (np.nextafter(cutoff,dtype(0)), cutoff, np.nextafter(cutoff,dtype(1))):
                    for saturation in (.0199999, .02, .0200001, .1):
                        data = np.zeros((9,9,3),dtype=dtype)
                        data[0,0] = peak
                        data.reshape(-1,3)[1:64] = (level, level*(1-saturation), level)
                        self.assertEqual(noise.is_chromatic(data,False), reference_chromatic(data,False))
        rng = np.random.default_rng(5)
        for data in (rng.random((20,30,4)), rng.integers(0, 65536, (20,30,3), dtype=np.uint16), np.ones((3,4,3)), np.full((10,10,3),np.nan), np.zeros((12,14))):
            for gray in (False, True):
                self.assertEqual(noise.is_chromatic(data,gray), reference_chromatic(data,gray))

    def test_complete_noise_and_texture_scores_match_original_preparation(self):
        rng = np.random.default_rng(4)
        colour = np.zeros((112,140,3),dtype=np.float32)
        colour[15:90,40:110] = rng.uniform(.02,.8,(75,70,3))
        for data, gray in ((colour,False), (colour[...,0],True), (colour,True), (colour[::-1, ::2],False), (np.asfortranarray(colour),False)):
            for texture, chromatic in ((2.,False),(5.,False),(2.,True),(None,None)):
                with ExitStack() as stack:
                    for name, replacement in (('_luminance',reference_luminance),('_crop_to_subject',reference_crop),('is_chromatic',reference_chromatic)):
                        stack.enter_context(patch.object(noise,name,replacement))
                    expected = noise.absolute_noise(data,gray,texture_scale=texture,chromatic=chromatic)
                    expected_texture = noise.estimate_texture_scale(data,gray)
                with patch.object(noise, '_LUMINANCE_BLOCK_PIXELS', 97):
                    self.assertEqual(noise.absolute_noise(data,gray,texture_scale=texture,chromatic=chromatic),expected)
                    self.assertEqual(noise.estimate_texture_scale(data,gray),expected_texture)

    def test_band_tail_percentile_runs_only_when_it_contributes_to_score(self):
        lum = np.random.default_rng(2).uniform(.2,.5,(35,41))
        for texture, chromatic, expected_calls in ((2.,False,1),(5.,False,2),(5.,True,1)):
            with patch.object(noise, '_p99_abs', wraps=noise._p99_abs) as percentile:
                score = noise._hybrid_noise_level(lum, float(lum.max()), texture, chromatic=chromatic)
                self.assertIsNotNone(score)
                self.assertEqual(percentile.call_count,expected_calls)


if __name__ == '__main__':
    unittest.main()
