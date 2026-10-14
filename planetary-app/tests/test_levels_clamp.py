"""Levels can preserve highlight headroom through preview, presets and batch."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog

from planetary_tools.core.colour import linear_to_srgb, srgb_to_linear
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
