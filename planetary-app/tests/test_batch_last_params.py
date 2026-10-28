"""New batch steps start from the values last applied interactively, for every batch filter."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QDialog

from planetary_tools.core.document import ImageDocument
from planetary_tools.filters.registry import FILTERS, apply_filter, batch_filters
from planetary_tools.ui import dialogs
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.crop_dialog import CropImageDialog
from planetary_tools.ui.curves_dialog import CurvesDialog
from planetary_tools.ui.fft_denoise_dialog import FftDenoiseDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.rotate_dialog import RotateImageDialog
from planetary_tools.ui.scale_dialog import ScaleImageDialog

FILTER_DIALOGS = {
    'wavelet_sharpen': dialogs.WaveletSharpenDialog,
    'fft_denoise': FftDenoiseDialog,
    'wavelet_denoise': dialogs.WaveletDenoiseDialog,
    'adaptive_deconv': dialogs.AdaptiveDeconvDialog,
    'deglow': dialogs.DeglowDialog,
    'moon_enhance': dialogs.MoonEnhanceDialog,
    'stretch_contrast': dialogs.StretchContrastDialog,
    'colour_matrix': dialogs.ColourMatrixDialog,
    'saturation_vibrance': dialogs.SaturationVibranceDialog,
    'curves': CurvesDialog,
    'levels': dialogs.LevelsDialog,
}
GEOMETRY = ('scale_image', 'rotate_image', 'crop_image')


def _edit_levels(params):
    params['channels']['L']['gamma'] = 1.3
    return params


def _edit_curves(params):
    params['trc'] = 'linear' if params['trc'] != 'linear' else 'perceptual'
    return params


# Parameters held in a matrix or per-channel dicts rather than plain numbers.
NESTED_EDITS = {
    'colour_matrix': lambda p: {**p, 'matrix': [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, .9]]},
    'curves': _edit_curves,
    'levels': _edit_levels,
}


def planet():
    y, x = np.indices((90, 120))
    image = np.zeros((90, 120, 3), dtype=np.float32)
    image[(x-70)**2 + (y-40)**2 < 15**2] = .8
    return image


class BatchLastParamsTests(unittest.TestCase):
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

    def batch_step_params(self, fid):
        batch = BatchDialog()
        self.addCleanup(batch.close)
        batch._filter_combo.setCurrentIndex(batch._filter_combo.findData(fid))
        self.assertEqual(batch._preset_combo.currentText(), 'Last')
        batch._add_step()
        step = batch._steps[-1]
        self.assertEqual(step.preset_name, 'Last')
        return step.params

    def test_every_batch_filter_is_covered(self):
        self.assertEqual({f.id for f in batch_filters()}, set(FILTER_DIALOGS) | set(GEOMETRY))

    def test_filter_dialogs_save_last_for_batch(self):
        for fid, cls in FILTER_DIALOGS.items():
            with self.subTest(fid=fid):
                needs_mode = fid in ('wavelet_sharpen', 'adaptive_deconv')
                panel = cls(False) if needs_mode else cls()
                self.addCleanup(panel.close)
                params = panel.get_params()
                changed = None
                nested = NESTED_EDITS.get(fid)
                if nested is not None:
                    panel.set_params(nested(deepcopy(params)))
                    changed = 'nested' if panel.get_params() != params else None
                for key, value in params.items() if nested is None else ():
                    if changed is None and isinstance(value, float):
                        for step in (1.0, -1.0):
                            panel.set_params({**params, key: value + step})
                            if panel.get_params()[key] != value:
                                changed = key
                                break
                self.assertIsNotNone(changed, f'No editable numeric parameter found for {fid}')
                applied = panel.get_params()
                self.assertNotEqual(applied, FILTERS[fid].default_params)
                panel.save_last_preset()
                step = self.batch_step_params(fid)
                self.assertEqual({k: step[k] for k in applied}, applied)

    def window_with_image(self):
        window = MainWindow()

        def close():
            # Never block teardown on the unsaved-changes prompt.
            window._document.modified = False
            window.close()
        self.addCleanup(close)
        window._set_document(ImageDocument(planet()))
        return window

    def test_applied_scale_and_rotate_are_inherited(self):
        window = self.window_with_image()

        def scale_exec(dlg):
            dlg._aspect.setChecked(False)
            dlg._width.setValue(60)
            dlg._height.setValue(30)
            return QDialog.DialogCode.Accepted
        with patch.object(ScaleImageDialog, 'exec', scale_exec):
            window._run_scale_image()
        self.assertEqual((window._document.width, window._document.height), (60, 30))
        step = self.batch_step_params('scale_image')
        self.assertFalse(step['maintain_aspect'])
        self.assertAlmostEqual(step['width_percent'], 50.0)
        self.assertAlmostEqual(step['height_percent'], 100/3)
        self.assertEqual(apply_filter('scale_image', planet(), False, step).shape[:2], (30, 60))

        def rotate_exec(dlg):
            dlg._angle.setValue(30.0)
            dlg._crop.setChecked(True)
            return QDialog.DialogCode.Accepted
        with patch.object(RotateImageDialog, 'exec', rotate_exec):
            window._run_rotate_image()
        step = self.batch_step_params('rotate_image')
        self.assertEqual((step['angle'], step['crop_to_original']), (30.0, True))

    def test_cancelled_or_no_op_scale_keeps_previous_last(self):
        window = self.window_with_image()

        def half(dlg):
            dlg._percent.setValue(50.0)
            return QDialog.DialogCode.Accepted
        with patch.object(ScaleImageDialog, 'exec', half):
            window._run_scale_image()
        for result in (QDialog.DialogCode.Rejected, QDialog.DialogCode.Accepted):
            with patch.object(ScaleImageDialog, 'exec', lambda dlg, r=result: r):
                window._run_scale_image()
        self.assertEqual(self.batch_step_params('scale_image')['percent'], 50.0)

    def test_crop_keeps_autocrop_mode_or_manual_geometry(self):
        image = planet()
        dlg = CropImageDialog(120, 90, image, False)
        self.addCleanup(dlg.close)
        dlg._border.setValue(7)
        dlg._run_autocrop()
        params = dlg.batch_params()
        self.assertTrue(params['autocrop'])
        self.assertEqual(params['border_px'], 7)
        # Batch autocrop reproduces the interactive crop on the same image.
        rect = dlg.crop_rect()
        expected = image[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width]
        np.testing.assert_array_equal(apply_filter('crop_image', image, False, params), expected)
        dlg._width.setValue(dlg._width.value() - 4)
        params = dlg.batch_params()
        self.assertFalse(params['autocrop'])
        self.assertEqual((params['width'], params['height']), (dlg._width.value(), dlg._height.value()))
        self.assertEqual((params['offset_x'], params['offset_y']), (dlg._offset_x.value(), dlg._offset_y.value()))
        window = self.window_with_image()
        window._remember_batch_params('crop_image', params)
        step = self.batch_step_params('crop_image')
        self.assertEqual({k: step[k] for k in params}, params)


if __name__ == '__main__':
    unittest.main()
