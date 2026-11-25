"""Levels can preserve highlight headroom through preview, presets and batch."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog

from planetary_tools.core.colour import linear_to_srgb, rgb_to_oklab_L, srgb_to_linear
from planetary_tools.core.presets import load_presets, save_presets
from planetary_tools.filters.levels import (
    apply_levels, apply_levels_curve, default_levels_params, identity_levels,
    levels_params_for_preset,
)
from planetary_tools.filters.registry import FILTERS, apply_filter
from planetary_tools.ui.dialogs import LevelsDialog, edit_filter_params


def parameters(channel='R', clamp=True):
    channels = default_levels_params()
    channels[channel]['in_max'] = .5
    return {'channels': channels, 'clamp': clamp}


class LevelsClampTests(unittest.TestCase):
    def test_gamma_extends_above_white_but_black_stays_finite(self):
        levels = dict(identity_levels(), in_min=.1, in_max=.5, gamma=2.)
        values = np.array([0., .2, .5, .9], dtype=np.float32)
        np.testing.assert_allclose(apply_levels_curve(values, levels), [0., .5, 1., 1.])
        np.testing.assert_allclose(apply_levels_curve(values, levels, clamp=False),
                                   [0., .5, 1., np.sqrt(2.)])

    def test_output_mapping_preserves_existing_headroom_when_disabled(self):
        values = np.array([.5, 1., 2.], dtype=np.float32)
        levels = dict(identity_levels(), out_max=.5)
        np.testing.assert_allclose(apply_levels_curve(values, levels), [.25, .5, .5])
        np.testing.assert_allclose(apply_levels_curve(values, levels, clamp=False),
                                   [.25, .5, 1.])

    def test_rgb_headroom_survives_transfer_and_other_channel_edits(self):
        source = np.array([[[.25, 1.3, .2], [.8, .4, .1]]], dtype=np.float32)
        original = source.copy()
        source.setflags(write=False)
        for channel in ('R', 'G', 'B'):
            with self.subTest(channel=channel):
                params = parameters(channel, clamp=False)
                encoded = linear_to_srgb(source, clamp=False)
                encoded[..., ('R', 'G', 'B').index(channel)] /= .5
                np.testing.assert_allclose(apply_levels(source, params),
                                           srgb_to_linear(encoded), rtol=1e-6)
                params['clamp'] = True
                self.assertLessEqual(float(apply_levels(source, params).max()), 1.)
        np.testing.assert_array_equal(source, original)

    def test_luminance_and_combined_rgb_edits_keep_highlights(self):
        source = np.array([[[.1, .2, .3], [.4, .6, .8]]], dtype=np.float32)
        params = parameters('L', clamp=False)
        self.assertGreater(float(apply_levels(source, params).max()), 1.)
        params['channels']['R']['gamma'] = 1.2
        self.assertGreater(float(apply_levels(source, params).max()), 1.)
        params['clamp'] = True
        self.assertLessEqual(float(apply_levels(source, params).max()), 1.)

    def test_luminance_on_a_stretched_red_image_brightens_from_100_percent(self):
        # A red planet stretched so red reaches 1, with an OKLab L peak far below 1.
        yy, xx = np.mgrid[:40, :40]
        level = np.clip(1 - ((yy - 20) ** 2 + (xx - 20) ** 2) / 400, 0, 1)
        image = (level[..., None] * np.array([1., .45, .16])).astype(np.float32)
        peak = float(rgb_to_oklab_L(image).max())
        self.assertLess(peak, .85)
        for clamp in (True, False):
            means = [image.mean()]
            for in_max in (.999, .98, .95, .9, peak):
                params = default_levels_params()
                params['L']['in_max'] = in_max
                out = apply_levels(image, dict(params, clamp=clamp))
                means.append(out.mean())
                if in_max == .999:
                    np.testing.assert_allclose(out, image, atol=.004)
            self.assertTrue(all(b > a for a, b in zip(means, means[1:])), (clamp, means))
        # Input max at the L peak still matches a 100% stretch.
        params = default_levels_params()
        params['L']['in_max'] = peak
        np.testing.assert_allclose(
            apply_levels(image, params),
            apply_filter('stretch_contrast', image, False, {'amount': 100.0}), atol=1e-6)

    def test_luminance_output_max_sets_the_peak_lightness(self):
        for image in (np.full((4, 4, 3), .5, np.float32),
                      np.tile(np.array([1., .45, .16], np.float32), (4, 4, 1))):
            peak = float(rgb_to_oklab_L(image).max())
            for out_max in (.999, .9, .5):
                params = default_levels_params()
                params['L']['out_max'] = out_max
                out = apply_levels(image, params)
                self.assertAlmostEqual(float(rgb_to_oklab_L(out).max()), out_max * peak, places=4)

    def test_legacy_params_default_to_clamped_and_identity_preserves_data(self):
        source = np.array([[[.2, .4, 1.2]]], dtype=np.float32)
        params = parameters()
        self.assertTrue(FILTERS['levels'].default_params['clamp'])
        expected = apply_levels(source, params)
        del params['clamp']
        np.testing.assert_array_equal(apply_levels(source, params), expected)
        for clamp in (True, False):
            np.testing.assert_array_equal(apply_levels(source, {'clamp': clamp}), source)


class LevelsClampDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = patch('planetary_tools.core.presets.PRESET_DIR', Path(temporary.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_toggle_changes_preview_and_matches_apply_and_batch_filter(self):
        panel = LevelsDialog()
        self.addCleanup(panel.close)
        panel.set_params(parameters())
        source = np.array([[[.8, .4, .1]]], dtype=np.float32)
        self.assertTrue(panel._clamp_high.isChecked())
        changes = []
        panel.params_changed.connect(lambda: changes.append(True))
        for enabled in (False, True):
            panel._clamp_high.setChecked(enabled)
            self.assertEqual(panel.get_params()['clamp'], enabled)
            preview = panel.build_preview_func()(source, False)
            expected = apply_filter('levels', source, False, panel.get_params())
            np.testing.assert_array_equal(panel.build_filter_func()(source, False), expected)
            np.testing.assert_array_equal(preview.data, expected)
            self.assertEqual(float(expected.max()) > 1., not enabled)
        self.assertEqual(len(changes), 2)

    def test_preset_roundtrip_and_legacy_default(self):
        panel = LevelsDialog()
        self.addCleanup(panel.close)
        panel.set_params(parameters(clamp=False))
        saved = levels_params_for_preset(panel.get_params())
        save_presets('levels', {'Unclamped': saved})
        restored = load_presets('levels', FILTERS['levels'].default_params)['Unclamped']
        panel.set_params(restored)
        self.assertFalse(panel.get_params()['clamp'])
        panel.set_params({'channels': default_levels_params()})
        self.assertTrue(panel.get_params()['clamp'])

    def test_batch_editor_defaults_on_and_returns_toggled_value(self):
        def accept(dialog):
            box, = [box for box in dialog.findChildren(QCheckBox)
                    if box.text() == 'Clamp to 100%']
            self.assertTrue(box.isChecked())
            box.setChecked(False)
            return QDialog.DialogCode.Accepted

        with patch.object(QDialog, 'exec', accept):
            result, _ = edit_filter_params('levels', {'channels': default_levels_params()}, False)
        self.assertFalse(result['clamp'])


if __name__ == '__main__':
    unittest.main()
