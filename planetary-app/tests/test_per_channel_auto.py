"""Auto in Wavelet Sharpen and Adaptive Deconvolution can search each RGB channel."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication, QProgressDialog
from scipy.ndimage import gaussian_filter

from planetary_tools.batch.pipeline import PipelineStep, _resolve_auto_params
from planetary_tools.filters.adaptive_deconv import _PreparedDeconvolution, adaptive_deconvolution
from planetary_tools.filters.adaptive_deconv_auto import (
    auto_adaptive_deconv_channel_params, auto_adaptive_deconv_params,
)
from planetary_tools.filters.registry import FILTERS, apply_filter
from planetary_tools.filters.wavelet import wavelet_sharpen, wavelet_sharpen_channels
from planetary_tools.filters.wavelet_auto import (
    auto_wavelet_sharpen_channel_params, auto_wavelet_sharpen_params,
)
from planetary_tools.ui.dialogs import (
    PER_CHANNEL_NOTE, PER_CHANNEL_READOUT_NOTE, AdaptiveDeconvDialog, WaveletSharpenDialog,
)
from test_sticky_auto import Isolated


def disk():
    """A soft planet whose channels differ in blur and noise."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:64, :80]
    base = ((yy - 32) ** 2 + (xx - 40) ** 2 < 22 ** 2).astype(np.float64) * .6 + .05
    base += .1 * (np.sin(yy / 2.5) > 0) * (base > .1)
    planes = [gaussian_filter(base, sigma) + rng.normal(0, noise, base.shape)
              for sigma, noise in ((2.4, .002), (1.4, .004), (1.8, .008))]
    return np.clip(np.stack(planes, -1), 0, 1).astype(np.float32)


WAVELET = [(10.0, 2.0, 0.5, 0.0), (4.0, 6.0, 0.0, 0.0), (0.0, 3.0, 1.5, 0.2)]


class PerChannelFilterTests(unittest.TestCase):
    def test_equal_amounts_match_rgb_and_each_channel_uses_its_own(self):
        data = disk()
        same = wavelet_sharpen_channels(data, [WAVELET[0]] * 3)
        np.testing.assert_array_equal(same, wavelet_sharpen(data, False, *WAVELET[0]))
        mixed = wavelet_sharpen_channels(data, WAVELET)
        for c in range(3):
            np.testing.assert_array_equal(mixed[..., c], wavelet_sharpen(data, False, *WAVELET[c])[..., c])
        engine = _PreparedDeconvolution(data, False, True, False)
        np.testing.assert_array_equal(engine.apply([4.0] * 3),
                                      adaptive_deconvolution(data, False, 4.0, True, luminance=False))
        mixed = engine.apply([1.0, 4.0, 9.0])
        for c, amount in enumerate((1.0, 4.0, 9.0)):
            np.testing.assert_array_equal(
                mixed[..., c], adaptive_deconvolution(data, False, amount, True, luminance=False)[..., c])
        with self.assertRaises(ValueError):
            _PreparedDeconvolution(data, False, True, True).apply([1.0, 2.0, 3.0])

    def test_channel_amounts_apply_only_with_auto_and_per_channel(self):
        data = disk()
        base = {'fine': 1.0, 'medium': 1.0, 'coarse': 0.0, 'chunky': 0.0, 'luminance': True,
                'channel_amounts': [list(a) for a in WAVELET]}
        sharpen = FILTERS['wavelet_sharpen'].apply
        per_channel = sharpen(data, False, {**base, 'auto': True, 'per_channel': True})
        np.testing.assert_array_equal(per_channel, wavelet_sharpen_channels(data, WAVELET))
        shared = wavelet_sharpen(data, False, 1.0, 1.0, 0.0, 0.0, luminance=True)
        for params in ({**base, 'auto': False, 'per_channel': True},
                       {**base, 'auto': True, 'per_channel': False}, {**base, 'per_channel': True}):
            # Released Auto or unticked per-channel: the shared amounts, with Luminance.
            np.testing.assert_array_equal(sharpen(data, False, params), shared)
        deconvolve = FILTERS['adaptive_deconv'].apply
        deconv = {'amount': 2.0, 'auto': True, 'per_channel': True, 'channel_amounts': [1.0, 4.0, 9.0]}
        np.testing.assert_array_equal(deconvolve(data, False, deconv),
                                      _PreparedDeconvolution(data, False, True, False).apply([1.0, 4.0, 9.0]))
        grey = np.repeat(data[..., 1:2], 3, -1)
        np.testing.assert_array_equal(deconvolve(grey, True, deconv),
                                      adaptive_deconvolution(grey, True, 2.0, True))

    def test_searches_score_each_channel_against_the_targets(self):
        data = disk()
        started = []
        results = auto_wavelet_sharpen_channel_params(
            data, 6.0, 10.0, channel_started=started.append, texture_scale=2.0, chromatic=True)
        self.assertEqual(started, [0, 1, 2])
        for c, result in enumerate(results):
            alone = auto_wavelet_sharpen_params(data[..., c], True, 6.0, 10.0,
                                                texture_scale=2.0, chromatic=True)
            self.assertEqual(result, alone)
            self.assertLessEqual(result.noise, 6.0 + 1e-6)
            self.assertLessEqual(result.contrast_pct, 10.0 + 1e-6)
        self.assertGreater(len({(r.fine, r.medium, r.coarse, r.chunky) for r in results}), 1)
        started.clear()
        results = auto_adaptive_deconv_channel_params(
            data, 6.0, 10.0, channel_started=started.append, texture_scale=2.0, chromatic=True)
        self.assertEqual(started, [0, 1, 2])
        for result in results:
            self.assertLessEqual(result.noise, 6.0 + 1e-6)
            self.assertLessEqual(result.contrast_pct, 10.0 + 1e-6)
        self.assertGreater(len({r.amount for r in results}), 1)
        # The shared search is unchanged by the refactor.
        shared = auto_adaptive_deconv_params(data, False, 6.0, 10.0, luminance=False,
                                             texture_scale=2.0, chromatic=True)
        self.assertGreater(shared.amount, 0.0)


def fake_channel_search(calls, values, make):
    def search(data, **kwargs):
        calls.append(kwargs)
        labels = []
        for index in range(3):
            kwargs['channel_started'](index)
            dialogs = [w for w in QApplication.topLevelWidgets()
                       if isinstance(w, QProgressDialog) and w.isVisible()]
            labels.append(dialogs[0].labelText() if dialogs else None)
        calls[-1]['labels'] = labels
        return [make(v) for v in values]
    return search


class PerChannelDialogTests(Isolated):
    def wavelet(self, calls, grey=False):
        panel = WaveletSharpenDialog(grey)
        self.addCleanup(panel.close)
        panel._auto_wavelet_sharpen_channel_params = fake_channel_search(
            calls, WAVELET, lambda v: SimpleNamespace(fine=v[0], medium=v[1], coarse=v[2], chunky=v[3]))
        panel._auto_wavelet_sharpen_params = lambda *a, **k: SimpleNamespace(
            fine=7.0, medium=1.0, coarse=0.0, chunky=0.0)
        panel.set_input_brightness(disk(), grey)
        return panel

    def test_per_channel_auto_reports_amounts_and_warns_about_manual_edits(self):
        calls = []
        panel = self.wavelet(calls)
        self.assertFalse(panel.per_channel.isChecked())
        self.assertIn(PER_CHANNEL_NOTE, panel.per_channel.toolTip())
        panel.per_channel.setChecked(True)
        panel.auto_apply.click()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['labels'][0].split('\n')[0], 'Calculating the red channel (1 of 3)…')
        self.assertTrue(all(PER_CHANNEL_NOTE in label for label in calls[0]['labels']))
        self.assertIn('blue channel (3 of 3)', calls[0]['labels'][2])
        self.assertTrue(panel.auto_apply.isChecked())
        self.assertEqual((panel.fine.value(), panel.medium.value()), WAVELET[1][:2])
        self.assertFalse(panel.channel_readout.isHidden())
        text = panel.channel_readout.text()
        self.assertIn('Red: 10.0 / 2.0 / 0.5 / 0.0', text)
        self.assertIn('Blue: 0.0 / 3.0 / 1.5 / 0.2', text)
        self.assertIn(PER_CHANNEL_READOUT_NOTE, text)
        self.assertFalse(panel.luminance.isEnabled())
        params = panel.get_params()
        self.assertEqual((params['auto'], params['per_channel']), (True, True))
        self.assertEqual(params['channel_amounts'], [list(a) for a in WAVELET])
        func = panel.build_filter_func()
        np.testing.assert_array_equal(func(disk(), False),
                                      apply_filter('wavelet_sharpen', disk(), False, params))
        # Editing an amount applies one set of amounts to every channel.
        panel.medium.setValue(3.0)
        self.assertFalse(panel.auto_apply.isChecked())
        self.assertTrue(panel.channel_readout.isHidden())
        self.assertTrue(panel.luminance.isEnabled())
        self.assertIsNone(panel.get_params()['channel_amounts'])

    def test_cancel_keeps_previous_results_and_loading_searches_again(self):
        calls = []
        panel = self.wavelet(calls)
        panel.per_channel.setChecked(True)
        panel.auto_apply.click()
        saved = panel.get_params()

        def cancel(data, **kwargs):
            kwargs['channel_started'](0)
            kwargs['progress'](50.0, 50.0, 50.0, 50.0, 0.0, 0.0)
        with patch.object(QProgressDialog, 'wasCanceled', return_value=True):
            panel._auto_wavelet_sharpen_channel_params = cancel
            panel.auto_apply.click()
        self.assertEqual(panel.get_params(), saved)
        self.assertFalse(panel.channel_readout.isHidden())

        other = self.wavelet(calls)
        other.set_params(saved)
        self.assertEqual(len(calls), 2)
        self.assertEqual(other.get_params()['channel_amounts'], saved['channel_amounts'])
        self.assertTrue(other.per_channel.isChecked())
        # Without a per-channel tick, Auto runs the shared search.
        other.per_channel.setChecked(False)
        other.auto_apply.click()
        self.assertEqual(len(calls), 2)
        self.assertIsNone(other.get_params()['channel_amounts'])
        self.assertTrue(other.channel_readout.isHidden())

    def test_greyscale_has_no_per_channel_search(self):
        calls = []
        panel = self.wavelet(calls, grey=True)
        self.assertFalse(panel.per_channel.isEnabled())
        panel.per_channel.setChecked(True)
        panel.auto_apply.click()
        self.assertEqual(calls, [])
        self.assertEqual(panel.fine.value(), 7.0)

    def test_deconvolution_per_channel_auto(self):
        calls = []
        panel = AdaptiveDeconvDialog(False)
        self.addCleanup(panel.close)
        panel._auto_adaptive_deconv_channel_params = fake_channel_search(
            calls, (1.0, 4.0, 9.0), lambda v: SimpleNamespace(amount=v))
        panel.set_input_brightness(disk(), False)
        panel.per_channel.setChecked(True)
        panel.auto_apply.click()
        self.assertEqual(calls[0]['adaptive'], panel.adaptive.isChecked())
        self.assertTrue(all(PER_CHANNEL_NOTE in label for label in calls[0]['labels']))
        self.assertEqual(panel.amount.value(), 4.0)
        self.assertIn('Green: 4.0', panel.channel_readout.text())
        self.assertFalse(panel.luminance.isEnabled())
        self.assertEqual(panel.get_params()['channel_amounts'], [1.0, 4.0, 9.0])
        panel.amount.setValue(5.0)
        self.assertIsNone(panel.get_params()['channel_amounts'])
        self.assertTrue(panel.luminance.isEnabled())


class PerChannelBatchTests(unittest.TestCase):
    def test_batch_searches_each_channel_of_each_image(self):
        data = disk()
        step = {'auto': True, 'per_channel': True, 'target_noise': 3.0, 'target_contrast': 10.0}
        with patch('planetary_tools.filters.wavelet_auto.auto_wavelet_sharpen_channel_params',
                   return_value=[SimpleNamespace(fine=f, medium=m, coarse=c, chunky=h)
                                 for f, m, c, h in WAVELET]) as search:
            params = _resolve_auto_params('wavelet_sharpen', data, False, step)
        self.assertEqual(search.call_args.kwargs, {'target_noise': 3.0, 'target_contrast': 10.0})
        self.assertEqual(params['channel_amounts'], [list(a) for a in WAVELET])
        self.assertEqual(params['fine'], WAVELET[1][0])
        with patch('planetary_tools.filters.adaptive_deconv_auto.auto_adaptive_deconv_channel_params',
                   return_value=[SimpleNamespace(amount=a) for a in (1.0, 4.0, 9.0)]):
            params = _resolve_auto_params('adaptive_deconv', data, False, step)
        self.assertEqual((params['channel_amounts'], params['amount']), ([1.0, 4.0, 9.0], 4.0))
        with patch('planetary_tools.filters.wavelet_auto.auto_wavelet_sharpen_params',
                   return_value=SimpleNamespace(fine=1.0, medium=2.0, coarse=3.0, chunky=0.0)) as shared:
            params = _resolve_auto_params('wavelet_sharpen', data, True, step)
        shared.assert_called_once()
        self.assertEqual(PipelineStep('wavelet_sharpen', step).label,
                         'Wavelet Sharpen — Auto per channel')


if __name__ == '__main__':
    unittest.main()
