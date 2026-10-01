"""Frequency response, automatic denoising quality, and UI integration."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox

from planetary_tools.filters.fft_denoise import (
    MAX_FREQUENCY, Spectrum, analyse_spectrum, auto_parameters, fft_denoise, transfer,
)
from planetary_tools.filters.registry import apply_filter, batch_filters
from planetary_tools.ui.fft_denoise_dialog import FftDenoiseDialog


def planet():
    y, x = np.mgrid[:256, :256]
    r = np.hypot(y-128, x-128)
    return (.05 + .5/(1+np.exp(np.clip((r-85)/2, -50, 50)))
            + .06*np.sin(x*.22)*np.exp(-(r/70)**6)).astype(np.float32)


class FftDenoiseTests(unittest.TestCase):
    def test_transfer_has_smooth_monotonic_transition_and_curve_direction(self):
        f = np.linspace(0, MAX_FREQUENCY, 10001)
        for curve in (-5, 0, 5):
            gain = transfer(f, 30, 50, curve)
            self.assertTrue(np.all(gain[f <= 30] == 1))
            self.assertTrue(np.all(gain[f >= 80] == 0))
            self.assertLess(np.diff(gain).max(), 1e-12)
            self.assertLess(abs(transfer(30.00001, 30, 50, curve)-1), 1e-10)
            self.assertLess(abs(transfer(79.99999, 30, 50, curve)), 1e-10)
        self.assertGreater(transfer(55, 30, 50, -3), transfer(55, 30, 50, 3))

    def test_constants_tiny_arrays_alpha_and_identity(self):
        for shape in ((1, 1), (1, 17), (17, 1), (51, 73), (51, 73, 1), (51, 73, 3)):
            arr = np.full(shape, .42, np.float32)
            np.testing.assert_allclose(fft_denoise(arr), arr, atol=2e-7)
            np.testing.assert_array_equal(fft_denoise(arr, auto=True), arr)
        rng = np.random.default_rng(12)
        rgba = rng.random((65, 67, 4), dtype=np.float32)
        out = fft_denoise(rgba)
        np.testing.assert_array_equal(out[..., 3], rgba[..., 3])
        np.testing.assert_array_equal(fft_denoise(rgba, start=141.4), rgba)
        self.assertFalse(np.shares_memory(out, rgba))
        with self.assertRaises(ValueError):
            fft_denoise(np.full((20, 20), np.nan))

    def test_retains_low_frequency_and_attenuates_noise_band(self):
        y, x = np.mgrid[:256, :256]
        low = .1*np.cos(2*np.pi*x/32)
        high = .02*np.cos(2*np.pi*y/3)
        image = (.5+low+high).astype(np.float32)
        out = fft_denoise(image, start=20, feather=30, curve=0)
        region = np.s_[40:-40, 40:-40]
        self.assertLess(np.mean((out[region]-(.5+low)[region])**2), 1e-8)

    def test_negative_start_is_applied_instead_of_clamped_to_zero(self):
        image = np.full((48, 64), .5, np.float32)
        out = fft_denoise(image, start=-20, feather=40, curve=0)
        np.testing.assert_allclose(out, .25, atol=1e-7)
        np.testing.assert_allclose(fft_denoise(image, start=0, feather=40, curve=0), image, atol=1e-7)

    def test_no_wraparound_at_opposite_edge(self):
        image = np.zeros((128, 128), np.float32)
        image[:, :12] = 1
        out = fft_denoise(image, start=30, feather=45)
        self.assertLess(np.max(np.abs(out[:, -16:])), 1e-4)
        self.assertLess(abs(float(out[:, 0].mean())-1), .01)

    def test_auto_reduces_known_noise_and_preserves_detail(self):
        clean = planet()
        rng = np.random.default_rng(42)
        for sigma in (.005, .02, .06):
            noisy = clean + rng.normal(0, sigma, clean.shape).astype(np.float32)
            settings = auto_parameters(analyse_spectrum(noisy))
            out = fft_denoise(noisy, **settings)
            self.assertLess(np.mean((out-clean)**2), np.mean((noisy-clean)**2)*.15)
            # The fine band structure has known amplitude in this central patch.
            region = np.s_[100:150, 90:170]
            self.assertLess(abs(float(out[region].std()/clean[region].std())-1), .08)
            np.testing.assert_allclose(fft_denoise(noisy, auto=True), out)
        clean_out = fft_denoise(clean, auto=True)
        self.assertLess(np.mean((clean_out-clean)**2), 1e-7)

    def test_auto_after_sharpening_preserves_sharpened_reference_detail(self):
        from planetary_tools.filters.wavelet import wavelet_sharpen
        clean = planet()
        noisy = clean + np.random.default_rng(9).normal(0, .01, clean.shape).astype(np.float32)
        for amount in (8, 16, 32):
            reference = wavelet_sharpen(clean, True, amount, amount/2, 1, 0)
            sharpened = wavelet_sharpen(noisy, True, amount, amount/2, 1, 0)
            out = fft_denoise(sharpened, auto=True)
            self.assertLess(np.mean((out-reference)**2), np.mean((sharpened-reference)**2)*.1)
            region = np.s_[100:150, 90:170]
            self.assertLess(abs(float(out[region].std()/reference[region].std())-1), .08)

    def test_auto_suppresses_spectral_upturn_instead_of_preserving_it(self):
        f = np.linspace(0, MAX_FREQUENCY, 257)
        # Signal falls to a valley before a broad correlated-noise peak.
        p = 100*np.exp(-f/7) + .4*np.exp(-((f-70)/10)**2) + .002
        settings = auto_parameters(Spectrum(f, p, .002, (512, 512)))
        gain = transfer(f, **settings)
        self.assertGreater(float(gain[np.argmin(abs(f-10))]), .95)
        self.assertLess(float(gain[np.argmin(abs(f-70))]), .1)

    def test_colour_noise_not_hidden_by_luminance_and_crop_keeps_native_scale(self):
        rng = np.random.default_rng(14)
        noise = rng.normal(0, .02, (80, 90)).astype(np.float32)
        rgb = np.stack((.3+noise, .3-noise, .3+noise), axis=-1)
        out = fft_denoise(rgb, auto=True)
        self.assertLess(np.mean((out-.3)**2), np.mean((rgb-.3)**2)*.15)
        large = np.zeros((1200, 1500), np.float32)
        large[10:266, 10:266] = planet()
        s = analyse_spectrum(large)
        self.assertEqual(s.shape, (1024, 1024))
        self.assertGreater(s.power.max(), 0)

    def test_registered_batch_filter(self):
        self.assertIn('fft_denoise', [f.id for f in batch_filters()])
        image = planet()
        out = apply_filter('fft_denoise', image, True, {'auto': True})
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.isfinite(out).all())


class FftDenoiseUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.presets = patch('planetary_tools.core.presets.PRESET_DIR', Path(self.tmp.name))
        self.presets.start()
        self.addCleanup(self.presets.stop)

    def test_auto_preview_params_presets_markers_and_difference(self):
        panel = FftDenoiseDialog()
        self.addCleanup(panel.close)
        panel.show()
        self.app.processEvents()
        image = planet()+np.random.default_rng(1).normal(0, .02, (256, 256)).astype(np.float32)
        panel.set_input_brightness(image, True)
        panel.auto_button.click()
        before = panel.get_params()
        self.assertLess(before['start'], 60)
        self.assertTrue(before['auto'])
        self.assertFalse(any(c.text() == 'Auto for each image' for c in panel.findChildren(QCheckBox)))
        result = panel.build_preview_func()(image, True)
        panel.update_histogram_display(result.data)
        panel.show_difference.setChecked(True)
        self.assertFalse(panel.difference.pixmap().isNull())
        panel.update_histogram_display(None)
        self.assertIn('Enable preview', panel.difference.text())
        panel.show_difference.setChecked(False)
        self.app.processEvents()
        rect = panel.graph.plot_rect()
        x = int(rect.left()+before['start']/MAX_FREQUENCY*rect.width())
        QTest.mousePress(panel.graph, Qt.MouseButton.LeftButton, pos=QPoint(x, int(rect.center().y())))
        QTest.mouseMove(panel.graph, QPoint(x+20, int(rect.center().y())))
        QTest.mouseRelease(panel.graph, Qt.MouseButton.LeftButton)
        self.assertGreater(panel.start.value(), before['start'])
        self.assertFalse(panel.get_params()['auto'])
        panel.set_params(before)
        panel.save_last_preset()
        reopened = FftDenoiseDialog()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_params(), before)
        self.assertTrue(panel.start.isEnabled())
        observed = []
        panel.params_changed.connect(lambda: observed.append(panel.get_params()))
        panel.start.setValue(panel.start.value()+1)
        self.assertFalse(observed[-1]['auto'])
        self.assertEqual(observed[-1]['start'], panel.start.value())
        panel.auto_button.click()
        self.assertTrue(panel.get_params()['auto'])
        np.testing.assert_allclose(panel.build_filter_func()(image, True),
                                   apply_filter('fft_denoise', image, True, panel.get_params()))
        panel.grab().save('/tmp/fft-denoise-ui.png')

    def test_auto_preset_recalculates_for_each_batch_image_after_sharpening(self):
        from planetary_tools.batch.pipeline import (
            PipelineStep, apply_pipeline, BatchWorkflow, workflow_to_dict, workflow_from_dict,
        )
        panel = FftDenoiseDialog()
        self.addCleanup(panel.close)
        image = planet()
        panel.set_input_brightness(image, True)
        panel.auto_button.click()
        steps = [PipelineStep('wavelet_sharpen', {'fine': 8}),
                 PipelineStep('fft_denoise', panel.get_params())]
        workflow, warnings = workflow_from_dict(workflow_to_dict(BatchWorkflow(steps=steps)))
        self.assertEqual(warnings, [])
        rng = np.random.default_rng(17)
        with patch('planetary_tools.filters.fft_denoise.analyse_spectrum', wraps=analyse_spectrum) as analyse:
            for sigma in (.005, .03):
                source = image + rng.normal(0, sigma, image.shape).astype(np.float32)
                sharpened = apply_filter('wavelet_sharpen', source, True, {'fine': 8})
                apply_pipeline(source, True, workflow.steps)
                np.testing.assert_array_equal(analyse.call_args.args[0], sharpened)
            self.assertEqual(analyse.call_count, 2)

    def test_negative_start_preset_and_marker_drag(self):
        panel = FftDenoiseDialog()
        self.addCleanup(panel.close)
        panel.show()
        self.app.processEvents()
        panel.start.setValue(-15)
        self.assertEqual(panel.get_params()['start'], -15)
        panel.save_last_preset()
        reopened = FftDenoiseDialog()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.start.value(), -15)
        graph = panel.graph
        rect = graph.plot_rect()
        self.assertLess(graph.minimum_frequency(), -15)
        x = int(graph.frequency_x(-15))
        QTest.mousePress(graph, Qt.MouseButton.LeftButton, pos=QPoint(x, int(rect.center().y())))
        QTest.mouseMove(graph, QPoint(int(graph.frequency_x(-30)), int(rect.center().y())))
        QTest.mouseRelease(graph, Qt.MouseButton.LeftButton)
        self.assertAlmostEqual(panel.start.value(), -30, delta=1)
        self.assertFalse(panel.get_params()['auto'])
        panel.grab().save('/tmp/fft-negative-start.png')

    def test_preview_accept_cancel_and_undo(self):
        from planetary_tools.ui.main_window import MainWindow
        from planetary_tools.core.document import ImageDocument
        window = MainWindow()
        self.assertFalse(window._fft_denoise_act.isEnabled())
        source = planet()
        window._set_document(ImageDocument(source.copy(), is_grayscale=True))
        self.assertTrue(window._fft_denoise_act.isEnabled())
        errors = []
        settings = dict(start=20., feather=35., curve=-1., auto=False)
        def edit(accept):
            try:
                panel = window._active_filter_dlg
                panel.set_params(settings)
                panel.params_changed.emit()
                panel.preview.setChecked(False)
                (panel._accept if accept else panel._reject)()
            except BaseException as exc:
                errors.append(exc)
                window._active_filter_dlg._reject()
        try:
            for accept in (False, True):
                QTimer.singleShot(0, lambda accept=accept: edit(accept))
                window._run_fft_denoise()
                self.assertFalse(errors, errors)
                expected = apply_filter('fft_denoise', source, True, settings) if accept else source
                np.testing.assert_array_equal(window._document.data, expected)
            window._undo_action()
            np.testing.assert_array_equal(window._document.data, source)
            window._redo_action()
            np.testing.assert_array_equal(window._document.data, expected)
        finally:
            window._document.modified = False
            window.close()

    def test_batch_parameter_editor_supports_auto(self):
        from planetary_tools.ui.dialogs import edit_filter_params
        def accept():
            dlg = QApplication.activeModalWidget()
            dlg.findChildren(QDoubleSpinBox)[0].setValue(-25)
            for check in dlg.findChildren(QCheckBox):
                if check.text() == 'Auto':
                    check.setChecked(True)
            dlg.accept()
        QTimer.singleShot(20, accept)
        result = edit_filter_params('fft_denoise', {'auto': False}, True)
        self.assertTrue(result[0]['auto'])
        self.assertEqual(result[0]['start'], -25)


if __name__ == '__main__':
    unittest.main()
