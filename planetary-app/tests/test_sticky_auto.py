"""Wavelet Sharpen and Adaptive Deconvolution Auto stays pressed and reaches batch."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.document import ImageDocument
from planetary_tools.core.presets import load_presets
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.dialogs import AdaptiveDeconvDialog, WaveletSharpenDialog
from planetary_tools.ui.main_window import MainWindow


def image():
    return (np.random.default_rng(4).random((48, 64, 3)) * .3).astype(np.float32)


def fake_wavelet_search(calls):
    def search(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(fine=11.0, medium=2.0, coarse=1.0, chunky=0.0)
    return search


class Isolated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)

        def settings():
            return QSettings(str(root/'settings.ini'), QSettings.Format.IniFormat)
        for target, value in (('planetary_tools.core.presets.PRESET_DIR', root/'presets'),
                              ('planetary_tools.ui.batch_dialog.QSettings', settings),
                              ('planetary_tools.ui.recent_files._settings', settings)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class StickyAutoButtonTests(Isolated):
    def wavelet(self, calls):
        panel = WaveletSharpenDialog(False)
        self.addCleanup(panel.close)
        panel._auto_wavelet_sharpen_params = fake_wavelet_search(calls)
        panel.set_input_brightness(image(), False)
        return panel

    def test_wavelet_auto_stays_pressed_until_any_setting_changes(self):
        calls = []
        panel = self.wavelet(calls)
        self.assertTrue(panel.auto_apply.isCheckable())
        self.assertFalse(panel.get_params()['auto'])
        panel.auto_apply.click()
        self.assertTrue(panel.auto_apply.isChecked())
        self.assertEqual((panel.get_params()['auto'], panel.fine.value()), (True, 11.0))
        # Clicking a pressed Auto searches again and stays pressed.
        panel.auto_apply.click()
        self.assertEqual(len(calls), 2)
        self.assertTrue(panel.auto_apply.isChecked())
        for change in (lambda: panel.fine.setValue(5.0), lambda: panel.chunky.setValue(1.0),
                       lambda: panel.luminance.toggle(), lambda: panel.target_noise.setValue(4.0),
                       lambda: panel.target_contrast.setValue(20.0)):
            panel.auto_apply.click()
            change()
            self.assertFalse(panel.auto_apply.isChecked())
            self.assertFalse(panel.get_params()['auto'])

    def test_cancelled_search_keeps_previous_state(self):
        panel = self.wavelet([])

        def cancelled(*args, progress, **kwargs):
            progress(50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        panel.fine.setValue(7.0)
        panel._auto_wavelet_sharpen_params = cancelled
        with patch('planetary_tools.ui.dialogs.QProgressDialog.wasCanceled', return_value=True):
            panel.auto_apply.click()
        self.assertEqual(panel.fine.value(), 7.0)
        self.assertFalse(panel.auto_apply.isChecked())

    def test_loaded_auto_searches_the_current_image(self):
        calls = []
        panel = WaveletSharpenDialog(False)
        self.addCleanup(panel.close)
        panel._auto_wavelet_sharpen_params = fake_wavelet_search(calls)
        # Restored before the image arrives, as the Last preset is on opening.
        panel.set_params({'fine': 99.0, 'auto': True, 'target_noise': 2.5})
        self.assertEqual(calls, [])
        panel.set_input_brightness(image(), False)
        self.assertEqual([c['target_noise'] for c in calls], [2.5])
        self.assertEqual(panel.fine.value(), 11.0)
        self.assertTrue(panel.auto_apply.isChecked())
        # Choosing a preset with Auto while open searches again, without
        # marking the preset as edited.
        panel._presets['Mine'] = {**panel.get_params(), 'fine': 42.0}
        panel._populate_preset_combo()
        panel._set_combo_to('Mine')
        self.assertEqual(len(calls), 2)
        self.assertEqual(panel._preset_combo.currentText(), 'Mine')
        panel.set_params({**panel.get_params(), 'auto': False, 'fine': 3.0})
        self.assertEqual((panel.auto_apply.isChecked(), panel.fine.value(), len(calls)),
                         (False, 3.0, 2))

    def test_adaptive_deconv_auto_stays_pressed_until_any_setting_changes(self):
        panel = AdaptiveDeconvDialog(False)
        self.addCleanup(panel.close)
        panel._auto_adaptive_deconv_params = lambda *a, **k: SimpleNamespace(amount=7.5)
        panel.set_input_brightness(image(), False)
        panel.auto_apply.click()
        self.assertEqual((panel.get_params()['auto'], panel.amount.value()), (True, 7.5))
        for change in (lambda: panel.amount.setValue(3.0), lambda: panel.adaptive.toggle(),
                       lambda: panel.luminance.toggle(), lambda: panel.target_noise.setValue(4.0),
                       lambda: panel.target_contrast.setValue(20.0)):
            panel.auto_apply.click()
            change()
            self.assertFalse(panel.get_params()['auto'])


class AutoReachesBatchTests(Isolated):
    def setUp(self):
        super().setUp()
        self.window = MainWindow()
        self.addCleanup(self.close_window)
        self.window._set_document(ImageDocument(image()))

    def close_window(self):
        self.window._document.modified = False
        self.window.close()

    def apply(self, run, attr, search, after=None):
        errors = []

        def drive():
            panel = self.window._active_filter_dlg
            try:
                setattr(panel, attr, search)
                panel.preview.setChecked(False)
                panel.auto_apply.click()
                if after:
                    after(panel)
                panel._accept()
            except BaseException as exc:
                errors.append(exc)
                panel._reject()
        QTimer.singleShot(0, drive)
        run()
        self.assertFalse(errors, errors)

    def test_unaltered_auto_goes_to_last_values_and_current_workflow(self):
        w = self.window
        self.apply(w._run_wavelet_sharpen, '_auto_wavelet_sharpen_params', fake_wavelet_search([]))
        self.apply(w._run_adaptive_deconv, '_auto_adaptive_deconv_params',
                   lambda *a, **k: SimpleNamespace(amount=7.5))
        for fid in ('wavelet_sharpen', 'adaptive_deconv'):
            self.assertTrue(load_presets(fid, {})['Last']['auto'], fid)
        batch = BatchDialog(current_operations=w._undo.stack.applied_operations())
        self.addCleanup(batch.close)
        batch._current_wf_btn.click()
        self.assertEqual([(s.filter_id, s.params['auto']) for s in batch._steps],
                         [('wavelet_sharpen', True), ('adaptive_deconv', True)])
        self.assertTrue(all(s.label.endswith('Auto') for s in batch._steps))
        batch._steps = []
        batch._filter_combo.setCurrentIndex(batch._filter_combo.findData('wavelet_sharpen'))
        batch._add_step()
        self.assertEqual(batch._steps[0].preset_name, 'Last')
        self.assertTrue(batch._steps[0].params['auto'])

    def test_adjusted_auto_result_is_not_auto_in_batch(self):
        w = self.window
        self.apply(w._run_wavelet_sharpen, '_auto_wavelet_sharpen_params', fake_wavelet_search([]),
                   after=lambda panel: panel.medium.setValue(4.0))
        self.assertFalse(load_presets('wavelet_sharpen', {})['Last']['auto'])
        op = w._undo.stack.applied_operations()[-1]
        self.assertEqual((op.params['auto'], op.params['fine'], op.params['medium']),
                         (False, 11.0, 4.0))


if __name__ == '__main__':
    unittest.main()
