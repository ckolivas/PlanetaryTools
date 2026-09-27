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


if __name__ == '__main__':
    unittest.main()
