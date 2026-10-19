"""Deglow regression tests: background removal, subject protection and UI lifecycle."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox

from planetary_tools.core.document import ImageDocument
from planetary_tools.core.colour import srgb_to_linear
from planetary_tools.filters.curves import apply_curves
from planetary_tools.filters.deglow import DEFAULT_DEGLOW_PARAMS, deglow
from planetary_tools.filters.registry import apply_filter, batch_filters
from planetary_tools.batch.pipeline import PipelineStep, apply_pipeline, BatchWorkflow, workflow_to_dict, workflow_from_dict
from planetary_tools.ui.dialogs import DeglowDialog, edit_filter_params


def scene():
    y, x = np.indices((256, 256))
    radius = np.hypot(x-128, y-128)
    sky = .001 + .02*np.exp(-radius**2/(2*55**2))
    disk = radius < 25
    image = sky + .6*disk
    image += .012*np.exp(-((x-188)**2 + (y-128)**2)/(2*2**2))
    return image.astype(np.float32), disk, radius


class DeglowTests(unittest.TestCase):
    def test_single_pixel_sources_receive_glow_correction(self):
        base, disk, _ = scene()
        for scale in (2, 7, 10, 30, 80):
            for rgb in (False, True):
                with self.subTest(scale=scale, rgb=rgb):
                    source = np.stack([base, base*.8, base*.6], axis=-1) if rgb else base
                    stars = source.copy()
                    # Include faint moons and an isolated star brighter than
                    # the planet, at different phases of the downsample grid.
                    for y, x, strength in ((128, 210, .008), (95, 191, .01), (64, 130, 2.0)):
                        stars[y, x] += np.array([strength, strength*.7, strength*.4]) if rgb else strength
                    result = deglow(stars, not rgb, radius=scale, blackpoint_compensation=False)
                    for y, x in ((128, 210), (95, 191), (64, 130)):
                        self.assertTrue(np.all(result[y, x] < stars[y, x]))
                    np.testing.assert_array_equal(result[disk], stars[disk])

    def test_single_bright_pixel_is_not_excluded_from_glow_estimate(self):
        base, disk, _ = scene()
        for rgb in (False, True):
            with self.subTest(rgb=rgb):
                source = np.stack([base, base*.8, base*.6], axis=-1) if rgb else base
                stars = source.copy()
                stars[64, 130] += np.array([2., 1.4, .8]) if rgb else 2.
                clean = deglow(source, not rgb, radius=30, blackpoint_compensation=False)
                result = deglow(stars, not rgb, radius=30, blackpoint_compensation=False)
                # Previously the pre-sampling median discarded this source,
                # preserving its signal to within float32 rounding. It now
                # contributes to the normal background estimate instead.
                lost_signal = (stars - source) - (result - clean)
                self.assertTrue(np.all(lost_signal[64, 130] > 5e-5))
                np.testing.assert_array_equal(result[disk], stars[disk])

    def test_removes_halo_preserves_disk_and_moon_contrast(self):
        source, disk, radius = scene()
        # Use a background scale larger than this scene's several-pixel moon.
        out = deglow(source, True, radius=30, threshold=5, margin=3, blackpoint_compensation=False)
        np.testing.assert_array_equal(out[disk], source[disk])
        halo = (radius > 45) & (radius < 85)
        self.assertLess(float(out[halo].mean()), float(source[halo].mean())*.6)
        # Subtracting a smooth background should preserve compact moon contrast.
        original_contrast = source[128,188] - source[128,198]
        contrast = out[128,188] - out[128,198]
        self.assertAlmostEqual(float(contrast), float(original_contrast), delta=.002)
        self.assertTrue(np.all(out <= source))
        self.assertGreaterEqual(float(out.min()), 0)
        np.testing.assert_array_equal(source, scene()[0])

    def test_strength_shapes_alpha_hdr_and_flat_frames(self):
        source, disk, _ = scene()
        for image in (source, source[...,None], np.stack([source, source*.8, source*.6], axis=-1)):
            result = deglow(image, image.ndim == 2 or image.shape[-1] == 1, blackpoint_compensation=False)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(deglow(image, amount=0, blackpoint_compensation=False), image)
        rgba = np.stack([source*3, source*2, source, np.full_like(source, .7)], axis=-1)
        out = deglow(rgba, blackpoint_compensation=False)
        np.testing.assert_array_equal(out[disk], rgba[disk])
        np.testing.assert_array_equal(out[...,3], rgba[...,3])
        self.assertGreater(out.max(), 1)
        for level in (0, .1, -.01):
            flat = np.full((20,30), level, dtype=np.float32)
            np.testing.assert_array_equal(deglow(flat, True, blackpoint_compensation=False), flat)
        full = deglow(source, True, amount=100, blackpoint_compensation=False)
        half = deglow(source, True, amount=50, blackpoint_compensation=False)
        np.testing.assert_allclose(half, (source+full)/2, atol=1e-7)

    def test_validation_and_batch_pipeline(self):
        with self.assertRaises(ValueError):
            deglow(np.zeros((3,3,2), dtype=np.float32))
        with self.assertRaises(ValueError):
            deglow(np.full((3,3), np.nan))
        with self.assertRaises(ValueError):
            deglow(scene()[0], radius=-1)
        self.assertIn('deglow', [item.id for item in batch_filters()])
        step = PipelineStep('deglow', {'amount': 60, 'radius': 20})
        workflow, warnings = workflow_from_dict(workflow_to_dict(BatchWorkflow(steps=[step])))
        self.assertEqual(warnings, [])
        image = scene()[0]
        np.testing.assert_array_equal(apply_pipeline(image, True, workflow.steps),
                                      apply_filter('deglow', image, True, step.params))


class BlackpointCompensationTests(unittest.TestCase):
    def test_default_blackpoint_anchors_in_perceptual_units(self):
        source = srgb_to_linear(np.array([[0., 2., 5., 7., 10., 255.]]) / 255)
        result = deglow(source, True, amount=0)
        np.testing.assert_allclose(result[0, :3], 0, atol=1e-9)
        self.assertGreater(result[0, 3], 0)
        self.assertLess(result[0, 3], source[0, 3])
        np.testing.assert_allclose(result[0, 4:], source[0, 4:], atol=1e-8)
        np.testing.assert_array_equal(
            deglow(source, True, amount=0, blackpoint_compensation=False), source)

    def test_matches_manual_curves_after_glow_removal_for_all_shapes(self):
        source, _, _ = scene()
        rgb = np.stack([source, source*.8, source*.6], axis=-1)
        rgba = np.dstack([rgb, np.full_like(source, .7)])
        for data in (source, source[..., None], rgb, rgba):
            for minimum, maximum in ((5., 10.), (2.5, 17.5), (0., 255.), (254.9, 255.)):
                with self.subTest(shape=data.shape, minimum=minimum, maximum=maximum):
                    before = data.copy()
                    # Equivalent to adding these points in the Curves panel,
                    # retaining its original black and white endpoints.
                    points = [[minimum/255, 0., 'corner'], [maximum/255, maximum/255, 'smooth']]
                    if minimum > 0:
                        points.insert(0, [0., 0., 'smooth'])
                    if maximum < 255:
                        points.append([1., 1., 'smooth'])
                    glow_only = deglow(data, blackpoint_compensation=False)
                    expected = apply_curves(glow_only, {
                        'trc': 'perceptual', 'channels': {'Value': {'mode': 'smooth', 'points': points}},
                    })
                    actual = deglow(data, blackpoint_min=minimum, blackpoint_max=maximum)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(data, before)
                    self.assertEqual(actual.dtype, np.float32)
                    if data is rgba:
                        np.testing.assert_array_equal(actual[..., 3], data[..., 3])

    def test_compensation_runs_even_without_a_detectable_subject(self):
        flat = np.full((20, 30), srgb_to_linear(np.array(3./255)), dtype=np.float32)
        np.testing.assert_array_equal(deglow(flat, True), np.zeros_like(flat))
        np.testing.assert_array_equal(deglow(flat, True, blackpoint_compensation=False), flat)

    def test_validation_and_batch_round_trip(self):
        source, _, _ = scene()
        for minimum, maximum in ((5, 5), (10, 5), (-1, 10), (5, 256), (np.nan, 10), (5, np.inf)):
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                deglow(source, blackpoint_min=minimum, blackpoint_max=maximum)
        for enabled in (True, False):
            params = {'blackpoint_compensation': enabled, 'blackpoint_min': 3.5, 'blackpoint_max': 12.5}
            workflow, warnings = workflow_from_dict(workflow_to_dict(
                BatchWorkflow(steps=[PipelineStep('deglow', params)])))
            self.assertFalse(warnings)
            np.testing.assert_array_equal(apply_pipeline(source, True, workflow.steps),
                                          deglow(source, True, **params))


class DeglowUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        presets = patch('planetary_tools.core.presets.PRESET_DIR', Path(directory.name))
        presets.start()
        self.addCleanup(presets.stop)

    def test_numeric_params_presets_and_batch_editor(self):
        panel = DeglowDialog()
        self.addCleanup(panel.close)
        self.assertEqual(panel.get_params(), DEFAULT_DEGLOW_PARAMS)
        panel.set_params({'amount': 30, 'radius': 18})
        self.assertEqual(panel.get_params()['amount'], 30)
        self.assertFalse(panel._deglow_spins['amount'].keyboardTracking())
        panel.save_last_preset()
        reopened = DeglowDialog()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_params()['radius'], 18)
        def accept():
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QDialog) and widget.windowTitle() == 'Deglow':
                    widget.findChildren(QDoubleSpinBox)[0].setValue(42)
                    widget.accept()
        QTimer.singleShot(0, accept)
        result = edit_filter_params('deglow', dict(DEFAULT_DEGLOW_PARAMS), True)
        self.assertEqual(result[0]['amount'], 42)

    def test_blackpoint_controls_presets_and_batch_editor(self):
        panel = DeglowDialog()
        self.addCleanup(panel.close)
        low, high = (panel._deglow_spins[key] for key in ('blackpoint_min', 'blackpoint_max'))
        self.assertTrue(panel.blackpoint_compensation.isChecked())
        self.assertEqual((low.value(), high.value()), (5., 10.))
        self.assertFalse(low.keyboardTracking())
        changes = []
        panel.params_changed.connect(lambda: changes.append(panel.get_params()))
        panel.blackpoint_compensation.setChecked(False)
        self.assertFalse(low.isEnabled())
        self.assertFalse(high.isEnabled())
        self.assertFalse(changes[-1]['blackpoint_compensation'])
        for minimum, maximum in ((20., 30.), (2., 3.), (5., 10.)):
            panel.set_params({'blackpoint_compensation': False,
                              'blackpoint_min': minimum, 'blackpoint_max': maximum})
            self.assertEqual((low.value(), high.value()), (minimum, maximum))
        panel.save_last_preset()
        reopened = DeglowDialog()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_params(), panel.get_params())
        panel.set_params({'amount': 75})  # Older presets acquire the new defaults.
        self.assertTrue(panel.blackpoint_compensation.isChecked())
        self.assertTrue(low.isEnabled())
        low.setValue(30)
        self.assertLess(low.value(), high.value())
        high.setValue(0)
        self.assertLess(low.value(), high.value())

        # The batch editor must load presets outside the current pair's
        # dynamic bounds without silently clamping their values.
        panel.set_params({'blackpoint_min': 20., 'blackpoint_max': 30.})
        panel.save_last_preset()

        errors = []
        def accept():
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QDialog) and widget.windowTitle() == 'Deglow':
                    try:
                        toggle = next(box for box in widget.findChildren(QCheckBox)
                                      if box.text() == 'Blackpoint compensation')
                        low = widget.findChild(QDoubleSpinBox, 'blackpoint_min')
                        high = widget.findChild(QDoubleSpinBox, 'blackpoint_max')
                        presets = widget.findChild(QComboBox)
                        presets.setCurrentText('Last')
                        self.assertEqual((low.value(), high.value()), (20., 30.))
                        presets.setCurrentText('Default')
                        self.assertEqual((low.value(), high.value()), (5., 10.))
                        high.setValue(15.)
                        low.setValue(6.5)
                        toggle.setChecked(False)
                        self.assertFalse(low.isEnabled())
                        self.assertFalse(high.isEnabled())
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        widget.accept()
        QTimer.singleShot(0, accept)
        result = edit_filter_params('deglow', dict(DEFAULT_DEGLOW_PARAMS), True)
        self.assertFalse(errors, errors)
        self.assertFalse(result[0]['blackpoint_compensation'])
        self.assertEqual((result[0]['blackpoint_min'], result[0]['blackpoint_max']), (6.5, 15.))

    def test_preview_apply_cancel_undo_redo(self):
        from planetary_tools.ui.main_window import MainWindow
        window = MainWindow()
        self.assertFalse(window._deglow_act.isEnabled())
        source = scene()[0]
        window._set_document(ImageDocument(source.copy(), is_grayscale=True))
        self.assertTrue(window._deglow_act.isEnabled())
        errors = []
        def edit(accept):
            try:
                panel = window._active_filter_dlg
                panel.set_params(DEFAULT_DEGLOW_PARAMS)
                np.testing.assert_array_equal(panel.build_filter_func()(source, True), deglow(source, True))
                panel.preview.setChecked(False)
                panel.preview.setChecked(True)
                (panel._accept if accept else panel._reject)()
            except BaseException as exc:
                errors.append(exc)
                window._active_filter_dlg._reject()
        try:
            for accept in (False, True):
                QTimer.singleShot(0, lambda accept=accept: edit(accept))
                window._run_deglow()
                self.assertFalse(errors, errors)
                expected = deglow(source, True) if accept else source
                np.testing.assert_array_equal(window._document.data, expected)
            window._undo_action()
            np.testing.assert_array_equal(window._document.data, source)
            window._redo_action()
            np.testing.assert_array_equal(window._document.data, deglow(source, True))
        finally:
            window._document.modified = False
            window.close()


if __name__ == '__main__':
    unittest.main()
