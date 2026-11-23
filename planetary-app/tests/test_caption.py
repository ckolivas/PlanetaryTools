"""Caption placement, pixel precision and preview/undo lifecycle."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from planetary_tools.core.colour import srgb_to_linear
from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.caption_dialog import (
    CaptionDialog, POSITIONS, WEB_COLOURS, apply_caption, make_caption_overlay,
)
from planetary_tools.ui.main_window import MainWindow


def isolate_settings(test):
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    path = str(Path(directory.name) / 'settings.ini')
    test.settings = lambda: QSettings(path, QSettings.Format.IniFormat)
    settings_patch = patch('planetary_tools.ui.caption_dialog.QSettings', side_effect=test.settings)
    settings_patch.start()
    test.addCleanup(settings_patch.stop)
    test.preset_dir = Path(directory.name) / 'presets'
    preset_patch = patch('planetary_tools.core.presets.PRESET_DIR', test.preset_dir)
    preset_patch.start()
    test.addCleanup(preset_patch.stop)


class CaptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        isolate_settings(self)

    def font(self, size=24):
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        font.setPixelSize(size)
        return font

    def test_six_positions_and_border_match_visible_ink(self):
        source = np.zeros((240, 400, 3), dtype=np.float32)
        for position in POSITIONS:
            for border in (0, 9, 25):
                with self.subTest(position=position, border=border):
                    overlay = make_caption_overlay(400, 240, 'Saturn', self.font(),
                                                   QColor('white'), position, border)
                    result = apply_caption(source, overlay)
                    y, x = np.nonzero(np.any(result != source, axis=2))
                    if position.startswith('Top'):
                        self.assertEqual(y.min(), border)
                    else:
                        self.assertEqual(239-y.max(), border)
                    if position.endswith('left'):
                        self.assertEqual(x.min(), border)
                    elif position.endswith('right'):
                        self.assertEqual(399-x.max(), border)
                    else:
                        self.assertLessEqual(abs((x.min()+x.max())/2 - 199.5), .5)
                    self.assertTrue(np.all(source == 0))

    def test_named_colour_composites_in_linear_light_without_quantizing_source(self):
        source = np.random.default_rng(5).uniform(-.1, 1.4, (240, 400, 3)).astype(np.float32)
        before = source.copy()
        colour = QColor('cornflowerblue')
        overlay = make_caption_overlay(400, 240, 'Jupiter', self.font(40), colour, 'Top left', 12)
        result = apply_caption(source, overlay)
        h, w = overlay.coverage.shape
        region = result[12:12+h, 12:12+w]
        solid = overlay.coverage == 1
        self.assertTrue(solid.any())
        expected_colour = srgb_to_linear(np.array(colour.getRgbF()[:3], dtype=np.float32))
        np.testing.assert_array_equal(region[solid], np.broadcast_to(expected_colour, region[solid].shape))
        mask = np.zeros(source.shape[:2], bool)
        mask[12:12+h, 12:12+w] = overlay.coverage > 0
        np.testing.assert_array_equal(result[~mask], source[~mask])
        np.testing.assert_array_equal(source, before)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.any((overlay.coverage > 0) & (overlay.coverage < 1)))

    def test_font_styles_multiline_unicode_and_wrapping(self):
        font = self.font()
        plain = make_caption_overlay(240, 240, 'Saturn', font, QColor('white'), 'Top left', 10)
        for style in ('setBold', 'setUnderline', 'setItalic'):
            styled = QFont(font)
            getattr(styled, style)(True)
            overlay = make_caption_overlay(240, 240, 'Saturn', styled, QColor('white'), 'Top left', 10)
            self.assertFalse(np.array_equal(plain.coverage, overlay.coverage))
        multiline = make_caption_overlay(240, 240, 'Saturn\nΔ 2026 — α', font, QColor('white'), 'Top centre', 10)
        self.assertGreater(multiline.coverage.shape[0], plain.coverage.shape[0])
        wrapped = make_caption_overlay(140, 240, 'Saturn Jupiter Mars', font, QColor('white'), 'Bottom right', 10)
        self.assertLessEqual(wrapped.coverage.shape[1], 120)
        self.assertGreater(wrapped.coverage.shape[0], plain.coverage.shape[0])
        larger = make_caption_overlay(400, 240, 'Saturn', self.font(40), QColor('white'), 'Top left', 10)
        self.assertGreater(larger.coverage.shape[0], plain.coverage.shape[0])

    def test_empty_oversized_and_grayscale_captions(self):
        self.assertIsNone(make_caption_overlay(100, 100, ' \n ', self.font(), QColor('white'), 'Top left', 10))
        with self.assertRaises(ValueError):
            make_caption_overlay(100, 50, 'A\nB\nC', self.font(60), QColor('white'), 'Top left', 10)
        source = np.full((240, 400), .2, dtype=np.float32)
        for name in ('white', 'red'):
            overlay = make_caption_overlay(400, 240, 'Saturn', self.font(), QColor(name), 'Top left', 10)
            result = apply_caption(source, overlay)
            self.assertEqual(result.ndim, 2 if name == 'white' else 3)
            np.testing.assert_array_equal(result[0], source[0] if name == 'white' else np.repeat(source[0, :, None], 3, axis=1))

    def test_dialog_controls_and_snapshot(self):
        dialog = CaptionDialog(400, 240)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.font_size.value(), 12)
        self.assertEqual([dialog.position.itemText(i) for i in range(6)], list(POSITIONS))
        self.assertGreater(dialog.font_family.count(), 0)
        self.assertGreater(len(WEB_COLOURS), 100)
        self.assertIn('red', WEB_COLOURS)
        ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(ok.isEnabled())
        dialog.text.setPlainText('Saturn')
        self.assertTrue(ok.isEnabled())
        snapshot = dialog.build_filter_func()
        source = np.zeros((240, 400, 3), np.float32)
        expected = snapshot(source, False)
        dialog.colour.setCurrentIndex(dialog.colour.findData('red'))
        np.testing.assert_array_equal(snapshot(source, False), expected)
        self.assertFalse(np.array_equal(dialog.build_filter_func()(source, False), expected))
        dialog.font_size.setValue(1000)
        self.assertFalse(ok.isEnabled())
        self.assertIn('Reduce', dialog.message.text())
        np.testing.assert_array_equal(dialog.build_filter_func()(source, False), source)

    def test_presets_save_load_and_delete(self):
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        dialog = CaptionDialog(400, 240)
        self.addCleanup(dialog.close)
        combo = dialog.preset_combo
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ['(None)', 'Default'])
        dialog.text.setPlainText('C. Kolivas\n16" LX200')
        dialog.position.setCurrentText('Top right')
        dialog.font_size.setValue(18)
        dialog.italic.setChecked(True)
        dialog.colour.setCurrentIndex(dialog.colour.findData('gold'))
        dialog.border.setValue(25)
        saved = dialog.get_params()
        with patch.object(QInputDialog, 'getText', return_value=(' Signature ', True)):
            dialog._save_preset()
        self.assertEqual(combo.currentText(), 'Signature')
        self.assertTrue((self.preset_dir / 'caption.json').exists())
        # A hand edit leaves the preset; Default resets everything.
        dialog.bold.setChecked(True)
        self.assertEqual(combo.currentText(), '(None)')
        combo.setCurrentText('Default')
        self.assertEqual(dialog.get_params(), dialog.default_params())
        self.assertEqual(combo.currentText(), 'Default')
        # A new dialog (another session) loads the saved preset exactly.
        other = CaptionDialog(400, 240)
        self.addCleanup(other.close)
        other.preset_combo.setCurrentText('Signature')
        self.assertEqual(other.get_params(), saved)
        self.assertTrue(other.has_caption())
        # Reserved names are refused, an existing name asks before replacing.
        with patch.object(QInputDialog, 'getText', return_value=('Default', True)), \
                patch.object(QMessageBox, 'warning') as warning:
            other._save_preset()
        warning.assert_called_once()
        other.font_size.setValue(30)
        with patch.object(QInputDialog, 'getText', return_value=('Signature', True)), \
                patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            other._save_preset()
        other.preset_combo.setCurrentText('Signature')
        self.assertEqual(other.font_size.value(), 18)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            other._delete_preset()
        self.assertEqual(other.preset_combo.findText('Signature'), -1)
        fresh = CaptionDialog(400, 240)
        self.addCleanup(fresh.close)
        self.assertEqual(fresh.preset_combo.findText('Signature'), -1)

    def test_settings_round_trip_and_smaller_image(self):
        dialog = CaptionDialog(400, 240)
        self.addCleanup(dialog.close)
        dialog.text.setPlainText('Saturn\nΔ 2026')
        dialog.position.setCurrentText('Top right')
        dialog.font_family.setCurrentIndex(0)
        dialog.font_size.setValue(18)
        for box in (dialog.bold, dialog.underline, dialog.italic):
            box.setChecked(True)
        dialog.colour.setCurrentIndex(dialog.colour.findData('cornflowerblue'))
        dialog.border.setValue(30)
        dialog.preview.setChecked(False)
        expected = dialog.build_filter_func()(np.zeros((240, 400, 3), np.float32), False)
        dialog.save_settings()
        # A new dialog reads a new QSettings instance from the saved INI file.
        reopened = CaptionDialog(400, 240)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.text.toPlainText(), 'Saturn\nΔ 2026')
        self.assertEqual(reopened.position.currentText(), 'Top right')
        self.assertEqual(reopened.font_family.currentFont().family(), dialog.font_family.currentFont().family())
        self.assertEqual(reopened.font_size.value(), 18)
        self.assertTrue(all(box.isChecked() for box in (reopened.bold, reopened.underline, reopened.italic)))
        self.assertEqual(reopened.colour.currentData(), 'cornflowerblue')
        self.assertEqual(reopened.border.value(), 30)
        self.assertFalse(reopened.preview.isChecked())
        np.testing.assert_array_equal(reopened.build_filter_func()(np.zeros((240, 400, 3), np.float32), False), expected)
        smaller = CaptionDialog(40, 30)
        self.addCleanup(smaller.close)
        self.assertEqual(smaller.border.value(), 14)
        self.assertEqual(self.settings().value('caption/border', type=int), 30)

    def test_unavailable_font_and_invalid_settings_use_defaults(self):
        settings = self.settings()
        for key, value in {'font_family': 'This font does not exist', 'font_size': 'bad',
                           'position': 'invalid', 'colour': 'invalid'}.items():
            settings.setValue(f'caption/{key}', value)
        dialog = CaptionDialog(400, 240)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.font_size.value(), 12)
        self.assertEqual(dialog.position.currentText(), 'Bottom left')
        self.assertEqual(dialog.colour.currentData(), 'white')
        self.assertNotEqual(dialog.font_family.currentFont().family(), 'This font does not exist')


class CaptionLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        isolate_settings(self)
        self.window = MainWindow()
        self.assertFalse(self.window._caption_act.isEnabled())
        self.source = np.full((180, 300), .1, dtype=np.float32)
        self.window._set_document(ImageDocument(self.source.copy(), is_grayscale=True))
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
            self.assertLess(time.monotonic(), deadline)
            QTest.qWait(10)
        self.app.processEvents()
        return preview.display_data()

    def test_preview_cancel_apply_and_undo_restore_mono_mode(self):
        self.assertTrue(self.window._caption_act.isEnabled())
        tools = next(action.menu() for action in self.window.menuBar().actions() if action.text() == '&Tools')
        labels = [action.text().replace('&', '') for action in tools.actions()]
        self.assertEqual(labels, sorted(labels, key=str.casefold))
        expected = []
        for accept in (False, True):
            def interact(dialog):
                dialog.text.setPlainText('Jupiter\n2026')
                dialog.colour.setCurrentIndex(dialog.colour.findData('red'))
                expected[:] = [dialog.build_filter_func()(self.source, True)]
                np.testing.assert_array_equal(self.wait_preview(), expected[0])
                np.testing.assert_array_equal(self.window._document.data, self.source)
                self.assertTrue(self.window._document.is_grayscale)
                pixmap = self.window._canvas._pixmap_item.pixmap()
                self.assertEqual((pixmap.width(), pixmap.height()), (300, 180))
                dialog.preview.setChecked(False)
                np.testing.assert_array_equal(self.window._preview.display_data(), self.source)
                return dialog.DialogCode.Accepted if accept else dialog.DialogCode.Rejected
            with patch.object(CaptionDialog, 'exec', interact):
                self.window._run_caption()
            self.assertFalse(self.window._preview.is_active)
            if not accept:
                np.testing.assert_array_equal(self.window._document.data, self.source)
                self.assertTrue(self.window._document.is_grayscale)
                self.assertFalse(self.window._undo.stack.can_undo())
                self.assertFalse(self.settings().contains('caption/text'))
        self.assertEqual(self.settings().value('caption/text'), 'Jupiter\n2026')
        self.assertEqual(self.settings().value('caption/colour'), 'red')
        np.testing.assert_array_equal(self.window._document.data, expected[0])
        self.assertFalse(self.window._document.is_grayscale)
        self.assertEqual(self.window._undo.stack.undo_label(), 'Caption')
        self.window._undo_action()
        np.testing.assert_array_equal(self.window._document.data, self.source)
        self.assertTrue(self.window._document.is_grayscale)
        self.window._redo_action()
        np.testing.assert_array_equal(self.window._document.data, expected[0])
        self.assertFalse(self.window._document.is_grayscale)

    def test_empty_caption_does_not_make_an_edit(self):
        with patch.object(CaptionDialog, 'exec', return_value=CaptionDialog.DialogCode.Accepted):
            self.window._run_caption()
        self.assertFalse(self.window._undo.stack.can_undo())
        np.testing.assert_array_equal(self.window._document.data, self.source)


if __name__ == '__main__':
    unittest.main()
