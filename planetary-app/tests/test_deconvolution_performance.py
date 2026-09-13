"""Preparation reuse must preserve deconvolution pixels and auto-search results."""
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import convolve

from planetary_tools.core.colour import clamp01, linear_luminance, rgb_to_oklab, oklab_to_rgb
from planetary_tools.filters import adaptive_deconv as adaptive
from planetary_tools.filters import adaptive_deconv_auto as auto
from planetary_tools.filters import wiener_deconv as wiener


def reference_adaptive(data, gray, amount, use_adaptive, luminance):
    """Original single-evaluation equations, without prepared state."""
    src = np.asarray(data, dtype=np.float32)
    lum = (src if src.ndim == 2 else src[..., 0]) if gray else linear_luminance(src)
    contrast = adaptive._std_windowed(lum)
    normalized = (contrast - float(contrast.min())) / (float(contrast.max()) - float(contrast.min()) + 1e-10)
    weight = np.sqrt(normalized)
    channels = [lum] if gray or luminance else [src[..., c] for c in range(3)]
    output = []
    for channel in channels:
        conv = convolve(channel, adaptive._PSF, mode='reflect')
        relative = channel / (conv + 1e-12)
        correction = convolve(relative, adaptive._PSF_MIRROR, mode='reflect') - 1.0
        strength = amount / math.pi
        damped = 1.0 + strength * weight * correction if use_adaptive else 1.0 + strength * correction
        output.append(channel * damped * damped * damped)
    if gray:
        return output[0]
    if luminance:
        return src + (output[0] - lum)[..., None]
    return np.stack(output, axis=-1)


def reference_wiener(data, gray, amount, use_adaptive, oklab):
    if amount <= 0:
        return np.asarray(data, dtype=np.float32)
    src = clamp01(data)
    nsr = wiener._nsr_from_amount(amount)
    def channel_filter(channel):
        img = channel.astype(np.float64)
        h_freq = np.fft.rfft2(wiener._pad_psf(adaptive._PSF, *img.shape))
        g_freq = np.fft.rfft2(img)
        power = h_freq.real * h_freq.real + h_freq.imag * h_freq.imag
        return np.fft.irfft2(g_freq * (power / (power + nsr)), s=img.shape).astype(np.float32)
    lum = (src if src.ndim == 2 else src[..., 0]) if gray else linear_luminance(src)
    weight = wiener._adaptive_apply_weight(lum) if use_adaptive else None
    def filtered(channel):
        out = channel_filter(channel)
        return wiener._blend(channel, out, weight) if weight is not None else out
    if gray:
        return filtered(lum)
    if oklab:
        lab = rgb_to_oklab(src)
        lab[..., 0] = filtered(lab[..., 0])
        return oklab_to_rgb(lab)
    return np.stack([filtered(src[..., c]) for c in range(3)], axis=-1)


class DeconvolutionPerformanceTests(unittest.TestCase):
    def test_adaptive_pixels_match_original_equations_for_all_modes(self):
        rng = np.random.default_rng(72)
        colour = rng.uniform(-.02, 1.2, (29, 38, 3)).astype(np.float32)
        for data, gray in ((colour, False), (colour[..., 0], True), (colour, True), (colour[::-1, ::2], False), (np.zeros((13, 17)), True)):
            source = data.copy()
            for use_adaptive in (False, True):
                for luminance in (False, True):
                    engine = adaptive._PreparedDeconvolution(data, gray, use_adaptive, luminance)
                    for amount in (0., .1, 10., 67.3, 100.):
                        with self.subTest(shape=data.shape, gray=gray, adaptive=use_adaptive, luminance=luminance, amount=amount):
                            expected = reference_adaptive(data, gray, amount, use_adaptive, luminance)
                            np.testing.assert_array_equal(engine.apply(amount), expected)
                            np.testing.assert_array_equal(adaptive.adaptive_deconvolution(data, gray, amount, use_adaptive, luminance), expected)
                            self.assertEqual(engine.apply(amount).dtype, np.float32)
            np.testing.assert_array_equal(data, source)

    def test_trials_prepare_convolutions_once_and_do_not_accumulate_results(self):
        source = np.random.default_rng(6).uniform(.01, .8, (29, 38, 3)).astype(np.float32)
        for gray, luminance in ((True, False), (False, True), (False, False)):
            with patch.object(adaptive, '_convolve2d', wraps=adaptive._convolve2d) as conv:
                engine = adaptive._PreparedDeconvolution(source, gray, True, luminance)
                first = engine.apply(10)
                first_copy = first.copy()
                fields = set(vars(engine))
                for amount in (0, 1, 40, 90, 100, 10):
                    engine.apply(amount)
                self.assertEqual(conv.call_count, 6 if not gray and not luminance else 2)
                self.assertEqual(set(vars(engine)), fields)
                self.assertEqual(len(engine.corrections), 3 if not gray and not luminance else 1)
                np.testing.assert_array_equal(first, first_copy)
                np.testing.assert_array_equal(engine.apply(10), first_copy)

    def test_auto_prepares_only_once_and_preserves_search_progress(self):
        source = np.random.default_rng(27).uniform(.01, .3, (29, 38, 3)).astype(np.float32)
        class ReferenceEngine:
            def __init__(self, data, gray, adaptive, luminance):
                self.args = data, gray, adaptive, luminance
            def apply(self, amount):
                data, gray, adaptive, luminance = self.args
                return reference_adaptive(data, gray, amount, adaptive, luminance)
        for gray, luminance, use_adaptive in ((False, True, True), (False, False, False), (True, False, True)):
            for target_noise, target_contrast, max_amount in ((100., 100., 10.), (0., 0., 100.), (100., 100., 0.)):
                kwargs = dict(adaptive=use_adaptive, luminance=luminance, texture_scale=2, chromatic=False, target_noise=target_noise, target_contrast=target_contrast, max_amount=max_amount)
                before, after = [], []
                with patch.object(auto, '_PreparedDeconvolution', ReferenceEngine):
                    expected = auto.auto_adaptive_deconv_params(source, gray, progress=lambda *p: before.append(p), **kwargs)
                with patch.object(adaptive, '_convolve2d', wraps=adaptive._convolve2d) as conv:
                    actual = auto.auto_adaptive_deconv_params(source, gray, progress=lambda *p: after.append(p), **kwargs)
                    self.assertEqual(conv.call_count, 6 if not gray and not luminance else 2)
                self.assertEqual(actual, expected)
                self.assertEqual(after, before)

    def test_nonadaptive_skips_unused_contrast_map(self):
        with patch.object(adaptive, '_std_windowed') as std:
            adaptive.adaptive_deconvolution(np.ones((13, 17, 3), dtype=np.float32), False, adaptive=False)
            std.assert_not_called()

    def test_wiener_pixels_match_original_for_all_modes(self):
        source = np.random.default_rng(38).uniform(-.1, 1.2, (29, 38, 3)).astype(np.float32)
        for data, gray in ((source, False), (source[::-1, ::2], False), (source[..., 0], True), (source, True)):
            before = data.copy()
            for use_adaptive in (False, True):
                for oklab in (False, True):
                    for amount in (0., .1, 10., 100.):
                        expected = reference_wiener(data, gray, amount, use_adaptive, oklab)
                        actual = wiener.wiener_deconvolution(data, gray, amount, use_adaptive, oklab)
                        np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(data, before)

    def test_wiener_shares_one_kernel_fft_per_rgb_image(self):
        source = np.ones((29, 38, 3), dtype=np.float32)
        with patch.object(wiener, '_pad_psf', wraps=wiener._pad_psf) as kernel, patch.object(wiener.np.fft, 'rfft2', wraps=np.fft.rfft2) as fft:
            wiener.wiener_deconvolution(source, False, oklab=False)
            self.assertEqual(kernel.call_count, 1)
            self.assertEqual(fft.call_count, 4)  # One kernel and three image channels.


if __name__ == '__main__':
    unittest.main()
