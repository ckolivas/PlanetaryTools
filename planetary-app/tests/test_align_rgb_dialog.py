"""RGB alignment dock settings, immutable previews and edit lifecycle."""
import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from scipy.ndimage import rotate, shift

from planetary_tools.core.align import align_channel
from planetary_tools.core.document import ImageDocument
from planetary_tools.core.field_derotate import apply_rigid, estimate_rigid, mask_alignment_background
from planetary_tools.ui.align_rgb_dialog import AlignRgbDialog
from planetary_tools.ui.main_window import MainWindow
from test_alignment import planet


class AlignRgbDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_defaults_control_dependencies_and_preview_snapshot(self):
        panel = AlignRgbDialog()
        self.addCleanup(panel.close)
        self.assertTrue(panel._mask_enabled.isChecked())
        self.assertEqual(panel._mask_percent.value(), 25)
        self.assertTrue(panel._derotate.isChecked())
        self.assertEqual(panel._max_angle.value(), 45)
        events = []
        panel.params_changed.connect(lambda: events.append(True))
        default = panel.build_filter_func()
        panel._mask_percent.setValue(50)
        panel._max_angle.setValue(3)
        changed = panel.build_filter_func()
        panel._mask_enabled.setChecked(False)
        panel._derotate.setChecked(False)
        disabled = panel.build_filter_func()
        self.assertFalse(panel._mask_percent.isEnabled())
        self.assertFalse(panel._max_angle.isEnabled())
        self.assertEqual(len(events), 4)
        data = np.random.default_rng(25).random((10, 12, 3), dtype=np.float32)
        for func, expected in ((default, (.25, True, 45)), (changed, (.5, True, 3)),
                               (disabled, (None, False, 3))):
            with patch('planetary_tools.ui.align_rgb_dialog.align_channel',
                       side_effect=lambda ref, target, **kwargs: target.copy()) as align:
                np.testing.assert_array_equal(func(data, False), data)
                self.assertEqual(align.call_count, 2)
                for call in align.call_args_list:
                    np.testing.assert_array_equal(call.args[0], data[..., 1])
                    self.assertEqual(call.kwargs, dict(zip(('mask_fraction', 'rotate', 'max_angle'), expected)))
        panel._mask_enabled.setChecked(True)
        panel._derotate.setChecked(True)
        self.assertTrue(panel._mask_percent.isEnabled())
        self.assertTrue(panel._max_angle.isEnabled())

    def test_rotation_and_custom_mask_match_derotate(self):
        ref = planet()
        target = shift(rotate(ref, 1.37, reshape=False, order=3), (2.38, -3.71), order=3)
        original = target.copy()
        for fraction in (None, .25, .5):
            with self.subTest(fraction=fraction):
                a, b = [mask_alignment_background(p, fraction) if fraction is not None else p
                        for p in (ref, target)]
                match = estimate_rigid(a, b, rotate=True, max_angle=3)
                actual = align_channel(ref, target, mask_fraction=fraction, rotate=True, max_angle=3)
                np.testing.assert_array_equal(actual, apply_rigid(target, match, expand=False))
                self.assertAlmostEqual(match.angle_deg, -1.37, delta=.04)
                self.assertEqual(actual.shape, ref.shape)
        np.testing.assert_array_equal(target, original)

    def test_preview_toggle_cancel_apply_and_undo_in_detachable_dock(self):
        green = planet()
        data = np.stack([shift(rotate(green, .8, reshape=False), (2.2, -3.1)),
                         green, shift(rotate(green, -.6, reshape=False), (-1.7, 2.3))], axis=-1)
        window = MainWindow()
        window._set_document(ImageDocument(data.copy()))
        window.show()
        errors = []
        expected = []
        def interact(accept):
            panel = window._active_filter_dlg
            try:
                self.assertIsInstance(panel, AlignRgbDialog)
                panel._max_angle.setValue(2)
                panel._mask_percent.setValue(35)
                window._filter_dock.setFloating(True)
                window._filter_dock.setFloating(False)
                window._preview.update_now()
                deadline = time.monotonic() + 10
                while window._preview._result_generation != window._preview._generation:
                    self.assertLess(time.monotonic(), deadline, 'Preview did not finish')
                    QTest.qWait(10)
                shown = window._preview.display_data().copy()
                expected[:] = [shown]
                np.testing.assert_array_equal(window._document.data, data)
                np.testing.assert_array_equal(shown[..., 1], green)
                self.assertFalse(np.array_equal(shown[..., 0], data[..., 0]))
                panel.preview.setChecked(False)
                np.testing.assert_array_equal(window._preview.display_data(), data)
                # Commit the latest settings even with preview disabled.
                (panel._accept if accept else panel._reject)()
            except BaseException as exc:
                errors.append(exc)
                panel._reject()
        try:
            for accept in (False, True):
                QTimer.singleShot(0, lambda accept=accept: interact(accept))
                window._run_align_rgb()
                self.assertFalse(errors, errors)
                self.assertFalse(window._preview.is_active)
                self.assertFalse(window._filter_dialog_open)
                np.testing.assert_array_equal(window._document.data, expected[0] if accept else data)
                self.assertEqual(window._undo.stack.can_undo(), accept)
            window._undo_action()
            np.testing.assert_array_equal(window._document.data, data)
            window._redo_action()
            np.testing.assert_array_equal(window._document.data, expected[0])
        finally:
            window._document.modified = False
            window.close()

    def test_failed_apply_preserves_document_and_closes_preview(self):
        data = np.zeros((10, 12, 3), dtype=np.float32)
        window = MainWindow()
        window._set_document(ImageDocument(data.copy()))
        def accept():
            window._active_filter_dlg.preview.setChecked(False)
            window._active_filter_dlg._accept()
        try:
            with patch('planetary_tools.ui.align_rgb_dialog.align_channel',
                       side_effect=ValueError('Alignment failed')), \
                 patch('PyQt6.QtWidgets.QMessageBox.critical') as error:
                QTimer.singleShot(0, accept)
                window._run_align_rgb()
                error.assert_called_once()
            np.testing.assert_array_equal(window._document.data, data)
            self.assertFalse(window._preview.is_active)
            self.assertFalse(window._filter_dialog_open)
            self.assertFalse(window._undo.stack.can_undo())
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
