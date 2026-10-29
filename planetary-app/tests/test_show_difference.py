"""Enhance filters can show the preview's difference from the original on the canvas."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QCheckBox

from planetary_tools.core.document import ImageDocument
from planetary_tools.ui import dialogs
from planetary_tools.ui.fft_denoise_dialog import FftDenoiseDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.preview import difference_display_rgb, display_srgb

ENHANCE_DIALOGS = (
    lambda: dialogs.WaveletSharpenDialog(False),
    dialogs.WaveletDenoiseDialog,
    FftDenoiseDialog,
    lambda: dialogs.AdaptiveDeconvDialog(False),
    dialogs.MoonEnhanceDialog,
    dialogs.DeglowDialog,
)


def planet():
    y, x = np.indices((64, 80))
    image = np.full((64, 80, 3), .01, dtype=np.float32)
    image[(x-40)**2 + (y-32)**2 < 18**2] = (.5, .4, .3)
    return image + np.random.default_rng(2).normal(0, .01, image.shape).astype(np.float32)


class ShowDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_checkbox_follows_preview_on_enhance_filters_only(self):
        for make in ENHANCE_DIALOGS:
            panel = make()
            self.addCleanup(panel.close)
            with self.subTest(panel=type(panel).__name__):
                box = panel.show_difference
                rows = [panel._form.itemAt(i).widget() for i in range(panel._form.count())]
                self.assertEqual(rows.index(box), rows.index(panel.preview) + 1)
                self.assertTrue(box.isEnabled())
                box.setChecked(True)
                self.assertTrue(panel.difference_shown())
                panel.preview.setChecked(False)
                self.assertFalse(box.isEnabled())
                self.assertFalse(panel.difference_shown())
                panel.preview.setChecked(True)
                self.assertTrue(box.isEnabled())
        self.assertFalse(dialogs.MergeWaveletDetailDialog.supports_difference)
        levels = dialogs.LevelsDialog()
        self.addCleanup(levels.close)
        self.assertIsNone(levels.show_difference)
        self.assertFalse(levels.difference_shown())
        fft = FftDenoiseDialog()
        self.addCleanup(fft.close)
        self.assertEqual([c.text() for c in fft.findChildren(QCheckBox) if 'removed' in c.text().lower()], [])

    def test_difference_is_signed_on_mid_grey(self):
        original = np.full((4, 4, 3), .2, dtype=np.float32)
        modified = original.copy()
        modified[0, 0] = .4
        modified[3, 3] = .1
        rgb = difference_display_rgb(display_srgb(original), modified)
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue((rgb[1:3, 1:3] == 128).all())
        self.assertTrue((rgb[0, 0] == 255).all())
        self.assertTrue((rgb[3, 3] < 128).all())
        mono = difference_display_rgb(display_srgb(original[..., 0]), modified[..., 0])
        self.assertEqual(mono.shape, (4, 4, 3))
        np.testing.assert_array_equal(mono, rgb)

    def test_canvas_shows_difference_while_checked(self):
        window = MainWindow()

        def close():
            window._document.modified = False
            window.close()
        self.addCleanup(close)
        window._set_document(ImageDocument(planet()))
        shown = {}

        def drive():
            panel = window._active_filter_dlg
            try:
                window._preview.update_now()
                window._preview._worker.wait()
                self.app.processEvents()
                original = window._preview.original_data()
                preview = window._preview.display_data()
                shown['preview'] = window._canvas._pixmap_item.pixmap().toImage().pixelColor(40, 32)
                panel.show_difference.setChecked(True)
                expected = difference_display_rgb(display_srgb(original), preview)
                shown['difference'] = window._canvas._pixmap_item.pixmap().toImage().pixelColor(40, 32)
                shown['expected'] = tuple(int(v) for v in expected[32, 40])
                panel.preview.setChecked(False)
                shown['off'] = window._canvas._pixmap_item.pixmap().toImage().pixelColor(40, 32)
                panel.preview.setChecked(True)
                window._preview._worker.wait()
                self.app.processEvents()
                shown['back'] = window._canvas._pixmap_item.pixmap().toImage().pixelColor(40, 32)
            finally:
                panel._reject()
        QTimer.singleShot(0, drive)
        window._run_wavelet_denoise()
        rgb = lambda c: (c.red(), c.green(), c.blue())
        self.assertEqual(rgb(shown['difference']), shown['expected'])
        self.assertNotEqual(rgb(shown['difference']), rgb(shown['preview']))
        self.assertEqual(rgb(shown['back']), shown['expected'])
        self.assertNotEqual(rgb(shown['off']), shown['expected'])
        # Cancelling restores the document display and drops the cached original.
        self.assertIsNone(window._difference_base)
        document = window._document.to_display_rgb()
        self.assertEqual(rgb(window._canvas._pixmap_item.pixmap().toImage().pixelColor(40, 32)),
                         tuple(int(v) for v in document[32, 40]))


if __name__ == '__main__':
    unittest.main()
