"""Deglow regression tests: background removal, subject protection and UI lifecycle."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QDoubleSpinBox

from planetary_tools.core.document import ImageDocument
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
    def test_single_pixel_sources_keep_signal_without_protecting_their_glow(self):
        base, disk, _ = scene()
        for scale in (2, 10, 30, 80):
            for rgb in (False, True):
                with self.subTest(scale=scale, rgb=rgb):
                    source = np.stack([base, base*.8, base*.6], axis=-1) if rgb else base
                    stars = source.copy()
                    # Include faint moons and an isolated star brighter than
                    # the planet, at different phases of the downsample grid.
                    for y, x, strength in ((128, 210, .008), (95, 191, .01), (64, 130, 2.0)):
                        stars[y, x] += np.array([strength, strength*.7, strength*.4]) if rgb else strength
                    clean = deglow(source, not rgb, radius=scale)
                    result = deglow(stars, not rgb, radius=scale)
                    signal = stars - source
                    np.testing.assert_allclose(result - clean, signal, atol=8e-5, rtol=.001)
                    np.testing.assert_array_equal(result[disk], stars[disk])

    def test_removes_halo_preserves_disk_and_moon_contrast(self):
        source, disk, radius = scene()
        out = deglow(source, True)
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
            result = deglow(image, image.ndim == 2 or image.shape[-1] == 1)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(deglow(image, amount=0), image)
        rgba = np.stack([source*3, source*2, source, np.full_like(source, .7)], axis=-1)
        out = deglow(rgba)
        np.testing.assert_array_equal(out[disk], rgba[disk])
        np.testing.assert_array_equal(out[...,3], rgba[...,3])
        self.assertGreater(out.max(), 1)
        for level in (0, .1, -.01):
            flat = np.full((20,30), level, dtype=np.float32)
            np.testing.assert_array_equal(deglow(flat, True), flat)
        full = deglow(source, True, amount=100)
        half = deglow(source, True, amount=50)
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
