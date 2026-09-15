"""Rotation preview preserves the source and shares the final rendering path."""
import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.document import ImageDocument
from planetary_tools.core.rotate import rotate_image
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.rotate_dialog import RotateImageDialog


class RotateDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.data = np.random.default_rng(12).random((18, 30, 3), dtype=np.float32)
        self.window = MainWindow()
        self.window._document = ImageDocument(self.data.copy())
        self.window._canvas.set_document(self.window._document)
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window._preview.finish(apply=False)
        self.window._document.modified = False
        self.window.close()

    def wait_preview(self):
        preview = self.window._preview
        preview.update_now()
        deadline = time.monotonic() + 5
        while preview._result_generation != preview._generation:
            self.assertLess(time.monotonic(), deadline, 'Preview did not finish')
            QTest.qWait(10)
        self.app.processEvents()
        np.testing.assert_array_equal(self.window._document.data, self.data)
        return preview.display_data()

    def test_dialog_fits_controls(self):
        dlg = RotateImageDialog(1920, 1080)
        try:
            dlg.show()
            self.app.processEvents()
            self.assertEqual(dlg.width(), dlg.minimumSizeHint().width())
            self.assertLess(dlg.width(), 500)
            self.assertTrue(dlg.preview.isChecked())
        finally:
            dlg.close()

    def test_preview_angles_crop_toggle_cancel_and_reopen(self):
        def interact(dlg):
            for angle, crop in ((90, False), (-23.5, False), (-23.5, True)):
                dlg._angle.setValue(angle)
                dlg._crop.setChecked(crop)
                shown = self.wait_preview()
                np.testing.assert_array_equal(shown, rotate_image(
                    self.data, angle, crop_to_original=crop))
                pixmap = self.window._canvas._pixmap_item.pixmap()
                self.assertEqual((pixmap.height(), pixmap.width()), shown.shape[:2])
            dlg.preview.setChecked(False)
            np.testing.assert_array_equal(self.window._preview.display_data(), self.data)
            dlg._angle.setValue(70)
            dlg.preview.setChecked(True)
            np.testing.assert_array_equal(self.wait_preview(), rotate_image(
                self.data, 70, crop_to_original=True))
            dlg._angle.setValue(15)  # Cancel with another update pending.
            return dlg.DialogCode.Rejected
        with patch.object(RotateImageDialog, 'exec', interact):
            for _ in range(2):
                self.window._run_rotate_image()
                QTest.qWait(550)
                np.testing.assert_array_equal(self.window._document.data, self.data)
                self.assertFalse(self.window._preview.is_active)
                self.assertFalse(self.window._filter_dialog_open)
                self.assertFalse(self.window._undo.stack.can_undo())
                pixmap = self.window._canvas._pixmap_item.pixmap()
                self.assertEqual((pixmap.height(), pixmap.width()), self.data.shape[:2])

    def test_accept_cached_preview_and_undo(self):
        def interact(dlg):
            dlg._angle.setValue(32.5)
            self.wait_preview()
            return dlg.DialogCode.Accepted
        with patch.object(RotateImageDialog, 'exec', interact), patch(
            'planetary_tools.ui.main_window.rotate_image', wraps=rotate_image
        ) as render:
            self.window._run_rotate_image()
            self.assertEqual(render.call_count, 1)
        expected = rotate_image(self.data, 32.5)
        np.testing.assert_array_equal(self.window._document.data, expected)
        self.window._undo_action()
        np.testing.assert_array_equal(self.window._document.data, self.data)
        self.window._redo_action()
        np.testing.assert_array_equal(self.window._document.data, expected)

    def test_accept_latest_settings_with_preview_disabled(self):
        def interact(dlg):
            dlg._angle.setValue(90)
            self.wait_preview()
            dlg.preview.setChecked(False)
            dlg._angle.setValue(-41)
            dlg._crop.setChecked(True)
            return dlg.DialogCode.Accepted
        with patch.object(RotateImageDialog, 'exec', interact):
            self.window._run_rotate_image()
        np.testing.assert_array_equal(self.window._document.data, rotate_image(
            self.data, -41, crop_to_original=True))

    def test_noop_and_failure_restore_original(self):
        with patch.object(RotateImageDialog, 'exec', return_value=RotateImageDialog.DialogCode.Accepted):
            self.window._run_rotate_image()
        self.assertFalse(self.window._undo.stack.can_undo())

        def interact(dlg):
            dlg.preview.setChecked(False)
            dlg._angle.setValue(35)
            return dlg.DialogCode.Accepted
        with patch.object(RotateImageDialog, 'exec', interact), patch(
            'planetary_tools.ui.main_window.rotate_image', side_effect=ValueError('failed')
        ), patch('planetary_tools.ui.main_window.QMessageBox.critical') as error:
            self.window._run_rotate_image()
            error.assert_called_once()
        np.testing.assert_array_equal(self.window._document.data, self.data)
        self.assertFalse(self.window._undo.stack.can_undo())
        self.assertFalse(self.window._preview.is_active)
        self.assertFalse(self.window._filter_dialog_open)

    def test_flip_axes_after_rotation_and_crop_preserve_pixels(self):
        for source in (self.data, self.data[..., 0]):
            original = source.copy()
            for angle in (0, 90, 32.5):
                for crop in (False, True):
                    rotated = rotate_image(source, angle, crop_to_original=crop)
                    for horizontal, vertical in ((True, False), (False, True), (True, True)):
                        with self.subTest(shape=source.shape, angle=angle, crop=crop,
                                          horizontal=horizontal, vertical=vertical):
                            expected = rotated
                            if horizontal:
                                expected = expected[:, ::-1]
                            if vertical:
                                expected = expected[::-1]
                            actual = rotate_image(source, angle, crop_to_original=crop,
                                                  flip_horizontal=horizontal, flip_vertical=vertical)
                            np.testing.assert_array_equal(actual, expected)
                            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(source, original)

    def test_flip_only_preview_apply_and_undo(self):
        def interact(dlg):
            self.assertFalse(dlg.flip_horizontal())
            self.assertFalse(dlg.flip_vertical())
            dlg._flip_horizontal.setChecked(True)
            np.testing.assert_array_equal(self.wait_preview(), self.data[:, ::-1])
            dlg.preview.setChecked(False)
            np.testing.assert_array_equal(self.window._preview.display_data(), self.data)
            dlg._flip_vertical.setChecked(True)
            return dlg.DialogCode.Accepted
        with patch.object(RotateImageDialog, 'exec', interact):
            self.window._run_rotate_image()
        np.testing.assert_array_equal(self.window._document.data, self.data[::-1, ::-1])
        self.window._undo_action()
        np.testing.assert_array_equal(self.window._document.data, self.data)
        self.window._redo_action()
        np.testing.assert_array_equal(self.window._document.data, self.data[::-1, ::-1])

    def test_cancel_rotation_with_flips_restores_source(self):
        def interact(dlg):
            dlg._angle.setValue(90)
            dlg._flip_vertical.setChecked(True)
            np.testing.assert_array_equal(self.wait_preview(), np.rot90(self.data)[::-1])
            dlg._flip_horizontal.setChecked(True)
            return dlg.DialogCode.Rejected
        with patch.object(RotateImageDialog, 'exec', interact):
            self.window._run_rotate_image()
        np.testing.assert_array_equal(self.window._document.data, self.data)
        self.assertFalse(self.window._undo.stack.can_undo())


if __name__ == '__main__':
    unittest.main()
