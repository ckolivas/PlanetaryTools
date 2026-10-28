"""Relative moon brightness, backward compatibility, and parameter persistence."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog, QDoubleSpinBox

from planetary_tools.batch.pipeline import (
    BatchWorkflow, PipelineStep, apply_pipeline, workflow_from_dict, workflow_to_dict,
)
from planetary_tools.filters.moon_enhance import _DETECT_CACHE, moon_enhance
from planetary_tools.filters.registry import apply_filter
from planetary_tools.ui.dialogs import MoonEnhanceDialog, edit_filter_params


MOONS = ((40, 40), (40, 210), (150, 40))


def scene():
    y, x = np.indices((192, 256))
    image = np.zeros((192, 256), dtype=np.float32)
    planet = (x - 128)**2 + (y - 96)**2 < 16**2
    image[planet] = .6
    for (my, mx), brightness in zip(MOONS, (.02, .01, .005)):
        image[my, mx] = brightness
    return image, planet


def enhance(image, **kwargs):
    return moon_enhance(image, image.ndim == 2 or image.shape[-1] == 1,
                        25, 8, .5, 10, 12, **kwargs)


class MoonEnhanceTests(unittest.TestCase):
    def setUp(self):
        _DETECT_CACHE.clear()

    def test_single_pixel_moons_retain_ratios_and_brightest_reaches_target(self):
        source, planet = scene()
        for image in (source, source[..., None], np.repeat(source[..., None], 3, axis=2)):
            with self.subTest(shape=image.shape):
                original = image.copy()
                out = enhance(image, retain_relative_brightness=True)
                peaks = np.array([out[y, x] for y, x in MOONS]).reshape(3, -1)
                np.testing.assert_allclose(peaks[:, 0], [.25, .125, .0625], rtol=1e-6)
                np.testing.assert_array_equal(out[planet], image[planet])
                np.testing.assert_array_equal(out[image == 0], image[image == 0])
                np.testing.assert_array_equal(image, original)

    def test_legacy_independent_boost_and_batch_round_trip(self):
        source, _ = scene()
        params = {'brightness': 25, 'planet_margin': 10}
        legacy = apply_filter('moon_enhance', source, True, params)
        np.testing.assert_allclose([legacy[y, x] for y, x in MOONS], [.25]*3)
        params['retain_relative_brightness'] = True
        workflow, warnings = workflow_from_dict(workflow_to_dict(BatchWorkflow(
            steps=[PipelineStep('moon_enhance', params)]
        )))
        self.assertEqual(warnings, [])
        result = apply_pipeline(source, True, workflow.steps)
        np.testing.assert_array_equal(result, enhance(source, retain_relative_brightness=True))

    def test_shared_gain_cap_no_dimming_and_no_detections(self):
        source, _ = scene()
        out = enhance(source, max_gain=2, retain_relative_brightness=True)
        np.testing.assert_allclose([out[y, x] for y, x in MOONS], [.04, .02, .01])
        out = moon_enhance(source, True, 1, 8, .5, 10, 12,
                           retain_relative_brightness=True)
        np.testing.assert_array_equal(out, source)
        empty = np.zeros_like(source)
        np.testing.assert_array_equal(enhance(empty, retain_relative_brightness=True), empty)

    def test_rgb_hue_and_brightness_order_independent_of_detection_order(self):
        from planetary_tools.filters.moon_enhance import _cached_detections, _luminance
        source, _ = scene()
        rgb = source[..., None] * np.array([1, .8, .6], dtype=np.float32)
        lum = _luminance(rgb, False)
        detections, exclusion = _cached_detections(lum, 8, 10, 12, .05)
        # Detections are sorted by local SNR, which need not track peak brightness.
        with patch('planetary_tools.filters.moon_enhance._cached_detections',
                   return_value=(list(reversed(detections)), exclusion)):
            out = enhance(rgb, retain_relative_brightness=True)
        np.testing.assert_allclose(_luminance(out, False)[MOONS[0]], .25, rtol=1e-6)
        for y, x in MOONS:
            np.testing.assert_allclose(out[y, x] / out[y, x, 0], [1, .8, .6], rtol=1e-6)
        np.testing.assert_allclose(out[MOONS[1]], out[MOONS[0]] / 2, rtol=1e-6)

    def test_pseudo_relative_lifts_faint_moon_without_equalising_or_changing_hue(self):
        source, planet = scene()
        source[MOONS[2]] = .0002  # 1% of the brightest moon.
        rgb = source[..., None] * np.array([1, .8, .6], dtype=np.float32)
        original = rgb.copy()
        from planetary_tools.filters.moon_enhance import _luminance
        out = enhance(rgb, retain_relative_brightness=True, pseudo_relative_brightness=True)
        lum = _luminance(out, False)
        np.testing.assert_allclose([lum[p] for p in MOONS],
                                   .25 * np.sqrt([1., .5, .01]), rtol=1e-6)
        for p in MOONS:
            np.testing.assert_allclose(out[p]/out[p][0], [1., .8, .6], rtol=1e-6)
        np.testing.assert_array_equal(out[planet], rgb[planet])
        np.testing.assert_array_equal(out[rgb == 0], rgb[rgb == 0])
        np.testing.assert_array_equal(rgb, original)

    def test_pseudo_requires_relative_and_respects_gain_limit_and_no_dimming(self):
        source, _ = scene()
        np.testing.assert_array_equal(enhance(source, pseudo_relative_brightness=True),
                                       enhance(source))
        out = enhance(source, max_gain=2, retain_relative_brightness=True,
                      pseudo_relative_brightness=True)
        np.testing.assert_allclose([out[p] for p in MOONS], [.04, .02, .01])
        out = moon_enhance(source, True, .1, 8, .5, 10, 12,
                           retain_relative_brightness=True, pseudo_relative_brightness=True)
        np.testing.assert_array_equal(out, source)

    def test_pseudo_batch_roundtrip_and_gain_changes_reuse_detection(self):
        source, _ = scene()
        enhance(source, retain_relative_brightness=True)
        params = {'brightness': 25, 'planet_margin': 10,
                  'retain_relative_brightness': True, 'pseudo_relative_brightness': True}
        workflow, warnings = workflow_from_dict(workflow_to_dict(BatchWorkflow(
            steps=[PipelineStep('moon_enhance', params)]
        )))
        self.assertEqual(warnings, [])
        with patch('planetary_tools.filters.moon_enhance.detect_moons',
                   side_effect=AssertionError('Brightness options must use cached detections')):
            result = apply_pipeline(source, True, workflow.steps)
            expected = enhance(source, retain_relative_brightness=True,
                               pseudo_relative_brightness=True)
        np.testing.assert_array_equal(result, expected)


class MoonEnhanceRangeTests(unittest.TestCase):
    def setUp(self):
        _DETECT_CACHE.clear()

    def test_never_lifts_a_channel_past_100_or_below_0_and_ignores_old_clamp_params(self):
        mono, planet = scene()
        # An orange moon: at a 100% luminance target, red would reach ~2.5.
        image = np.stack([mono, mono, mono], axis=-1)
        image[40, 40] = (.05, .015, .005)
        image[40, 210] = (.01, .01, .01)
        out = moon_enhance(image, False, 100, 8, .5, 10, 12)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        np.testing.assert_allclose(out[40, 40].max(), 1.0, rtol=1e-6)
        np.testing.assert_allclose(out[40, 40] / out[40, 40].max(),
                                   image[40, 40] / image[40, 40].max(), rtol=1e-5)
        np.testing.assert_array_equal(out[planet], image[planet])
        # A clipped pixel is never dimmed, and stale clamp flags change nothing.
        image[150, 40] = (1.2, .9, .9)
        raw = moon_enhance(image, False, 100, 8, .5, 10, 12)
        self.assertGreaterEqual(float(raw[150, 40, 0]), 1.2)
        for stale in ({}, {'clamp': True, 'clamp_low': True}, {'clamp': False}, {'rescale': True}):
            with self.subTest(stale=stale):
                params = {'brightness': 100, 'planet_margin': 10, **stale}
                np.testing.assert_array_equal(apply_filter('moon_enhance', image, False, params), raw)


class MoonEnhanceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        presets = patch('planetary_tools.core.presets.PRESET_DIR', Path(directory.name))
        presets.start()
        self.addCleanup(presets.stop)

    def test_preview_params_presets_and_old_preset_compatibility(self):
        panel = MoonEnhanceDialog()
        self.addCleanup(panel.close)
        self.assertFalse(panel.get_params()['retain_relative_brightness'])
        changes = []
        panel.params_changed.connect(lambda: changes.append(True))
        panel.retain_relative_brightness.setChecked(True)
        self.assertTrue(changes)
        panel.set_params({'retain_relative_brightness': True, 'planet_margin': 10})
        source, _ = scene()
        np.testing.assert_array_equal(panel.build_filter_func()(source, True),
                                      enhance(source, retain_relative_brightness=True))
        panel.save_last_preset()
        reopened = MoonEnhanceDialog()
        self.addCleanup(reopened.close)
        self.assertTrue(reopened.get_params()['retain_relative_brightness'])
        reopened.set_params({'brightness': 30})
        self.assertFalse(reopened.get_params()['retain_relative_brightness'])

    def test_batch_editor_retains_option_and_changes_brightness(self):
        def accept():
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QDialog) and widget.windowTitle() == 'Moon Enhance':
                    for checkbox in widget.findChildren(QCheckBox):
                        if checkbox.text() == 'Retain relative brightness':
                            checkbox.setChecked(True)
                    widget.findChildren(QDoubleSpinBox)[0].setValue(40)
                    widget.accept()
        QTimer.singleShot(0, accept)
        result = edit_filter_params('moon_enhance', {}, True)
        self.assertTrue(result[0]['retain_relative_brightness'])
        self.assertEqual(result[0]['brightness'], 40)
        self.assertNotIn('clamp', result[0])
        self.assertNotIn('clamp_low', result[0])

    def test_dialog_has_no_clamp_options(self):
        panel = MoonEnhanceDialog()
        self.addCleanup(panel.close)
        self.assertIsNone(panel.clamp_channels)
        self.assertIsNone(panel.clamp_low)
        self.assertFalse(any('Clamp' in box.text() for box in panel.findChildren(QCheckBox)))
        self.assertNotIn('clamp', panel.get_params())

    def test_pseudo_control_dependency_preview_and_preset_restore(self):
        panel = MoonEnhanceDialog()
        self.addCleanup(panel.close)
        self.assertFalse(panel.pseudo_relative_brightness.isEnabled())
        self.assertFalse(panel.get_params()['pseudo_relative_brightness'])
        panel.retain_relative_brightness.setChecked(True)
        self.assertTrue(panel.pseudo_relative_brightness.isEnabled())
        panel.pseudo_relative_brightness.setChecked(True)
        panel.set_params({**panel.get_params(), 'planet_margin': 10})
        source, _ = scene()
        np.testing.assert_array_equal(panel.build_preview_func()(source, True).data,
            enhance(source, retain_relative_brightness=True, pseudo_relative_brightness=True))
        panel.save_last_preset()
        reopened = MoonEnhanceDialog()
        self.addCleanup(reopened.close)
        self.assertTrue(reopened.pseudo_relative_brightness.isEnabled())
        self.assertTrue(reopened.get_params()['pseudo_relative_brightness'])
        reopened.retain_relative_brightness.setChecked(False)
        self.assertFalse(reopened.pseudo_relative_brightness.isEnabled())
        self.assertFalse(reopened.get_params()['pseudo_relative_brightness'])
        reopened.set_params({'pseudo_relative_brightness': True})
        self.assertFalse(reopened.get_params()['pseudo_relative_brightness'])

    def test_batch_pseudo_dependency_and_preset_switch(self):
        from planetary_tools.ui.dialogs import _apply_params_to_batch_widgets
        from planetary_tools.filters.registry import FILTERS

        def accept(dialog):
            boxes = {b.text(): b for b in dialog.findChildren(QCheckBox)}
            relative = boxes['Retain relative brightness']
            pseudo = boxes['Pseudo-relative brightness']
            self.assertFalse(pseudo.isEnabled())
            relative.setChecked(True)
            self.assertTrue(pseudo.isEnabled())
            pseudo.setChecked(True)
            relative.setChecked(False)
            self.assertFalse(pseudo.isChecked())
            # Applying a preset blocks widget signals, so dependencies also
            # need explicit synchronisation after all controls are restored.
            def sync(enabled):
                pseudo.setEnabled(enabled)
                if not enabled:
                    pseudo.setChecked(False)
            widgets = {'retain_relative_brightness': relative,
                       'pseudo_relative_brightness': pseudo, '_sync_relative': sync}
            _apply_params_to_batch_widgets('moon_enhance', widgets,
                {'retain_relative_brightness': True, 'pseudo_relative_brightness': True},
                FILTERS['moon_enhance'].default_params)
            self.assertTrue(pseudo.isEnabled())
            self.assertTrue(pseudo.isChecked())
            return QDialog.DialogCode.Accepted

        with patch.object(QDialog, 'exec', accept):
            result, _ = edit_filter_params('moon_enhance', {}, True)
        self.assertTrue(result['retain_relative_brightness'])
        self.assertTrue(result['pseudo_relative_brightness'])


if __name__ == '__main__':
    unittest.main()
