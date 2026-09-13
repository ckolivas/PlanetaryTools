"""Multi-file inputs accumulate visibly and only process the retained list."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QPushButton

from planetary_tools.batch.pipeline import PipelineStep
from planetary_tools.ui.animate_dialog import AnimateDialog
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.compose_dialog import RGBComposeDialog
from planetary_tools.ui.field_derotate_dialog import FieldDerotateDialog


class MultipleFileDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.first = self.root/'first'/'image.png'
        self.second = self.root/'second'/'image.png'
        for path in (self.first, self.second):
            path.parent.mkdir()
            path.touch()
        def settings():
            return QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        for target, replacement in (
            ('planetary_tools.ui.batch_dialog.QSettings', settings),
            ('planetary_tools.ui.recent_files._settings', settings),
            ('planetary_tools.core.presets.PRESET_DIR', self.root/'presets'),
        ):
            mock = patch(target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def add_via_button(self, dialog, paths):
        buttons = [b for b in dialog.findChildren(QPushButton) if b.text() == 'Add files…']
        self.assertEqual(len(buttons), 1)
        self.assertFalse(any(b.text() == 'Select files…' for b in dialog.findChildren(QPushButton)))
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileNames', return_value=([str(p) for p in paths], '')):
            buttons[0].click()

    def test_batch_align_and_animate_append_and_display_all_files(self):
        for cls in (BatchDialog, FieldDerotateDialog, AnimateDialog):
            with self.subTest(dialog=cls.__name__):
                dialog = cls()
                self.addCleanup(dialog.close)
                self.add_via_button(dialog, [self.first])
                self.add_via_button(dialog, [self.second, self.first])
                if isinstance(dialog, BatchDialog):
                    self.assertEqual(dialog._input_files, [self.first, self.second])
                    self.assertEqual(dialog._input_list.count(), 2)
                    shown = [dialog._input_list.item(i).toolTip() for i in range(2)]
                else:
                    self.assertEqual(dialog._table.rowCount(), 2)
                    shown = [dialog._table.item(i, 0).toolTip() for i in range(2)]
                self.assertCountEqual(shown, [str(self.first), str(self.second)])

    def test_batch_folder_list_removal_and_worker_use_the_same_inputs(self):
        dialog = BatchDialog()
        self.addCleanup(dialog.close)
        for path in (self.first, self.second):
            with patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory', return_value=str(path.parent)):
                dialog._pick_folder()
        self.assertEqual(dialog._input_files, [self.first, self.second])
        self.assertEqual(dialog._input_list.count(), 2)
        dialog._input_list.item(0).setSelected(True)
        dialog._remove_input_files()
        self.assertEqual(dialog._input_files, [self.second])
        self.assertIsNone(dialog._input_folder)
        self.assertFalse(dialog._settings.contains('batch/inputFolder'))
        dialog._steps = [PipelineStep('curves')]
        dialog._output_dir.setText(str(self.root/'out'))
        with patch('planetary_tools.ui.batch_dialog._BatchWorker') as worker:
            worker.return_value.isRunning.return_value = False
            dialog._run_batch()
            self.assertEqual(worker.call_args.args[0], [self.second])
            worker.return_value.start.assert_called_once()
        dialog._worker = None
        dialog._clear_input_files()
        self.assertEqual(dialog._input_files, [])
        self.assertEqual(dialog._input_list.count(), 0)
        self.assertEqual(dialog._input_start_directory(), str(self.second.parent))
        reopened = BatchDialog()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened._input_files, [])

    def test_batch_restores_folder_files_and_recursion_into_visible_list(self):
        nested = self.first.parent/'nested'/'moon.png'
        nested.parent.mkdir()
        nested.touch()
        dialog = BatchDialog()
        self.addCleanup(dialog.close)
        with patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory', return_value=str(self.first.parent)):
            dialog._pick_folder()
        self.assertEqual(dialog._input_list.count(), 1)
        dialog._recursive.setChecked(True)
        self.assertEqual(dialog._input_list.count(), 2)
        dialog.reject()
        reopened = BatchDialog()
        self.addCleanup(reopened.close)
        self.assertTrue(reopened._recursive.isChecked())
        self.assertCountEqual(reopened._input_files, [self.first, nested])
        self.assertEqual(reopened._input_list.count(), 2)
        for i in range(2):
            reopened._input_list.item(i).setSelected(True)
        reopened._remove_input_files()
        self.assertEqual(reopened._input_list.count(), 0)
        reopened._recursive.setChecked(False)
        self.assertEqual(reopened._input_files, [])

    def test_animate_folder_add_keeps_frames_and_custom_output(self):
        dialog = AnimateDialog()
        self.addCleanup(dialog.close)
        self.add_via_button(dialog, [self.first])
        custom = str(self.root/'custom.gif')
        dialog._output.setText(custom)
        dialog._on_output_edited(custom)
        with patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory', return_value=str(self.second.parent)):
            dialog._pick_folder()
            dialog._pick_folder()
        self.assertCountEqual(dialog._paths, [self.first, self.second])
        self.assertEqual(dialog._table.rowCount(), 2)
        self.assertEqual(dialog._output.text(), custom)

    def test_compose_add_keeps_assignments_skips_duplicates_and_lists_paths(self):
        dialog = RGBComposeDialog()
        self.addCleanup(dialog.close)
        red = self.first.with_name('planet_R.png')
        another_red = self.second.with_name('planet_R.png')
        green = self.second.with_name('planet_G.png')
        blue = self.second.with_name('planet_B.png')
        self.add_via_button(dialog, [red])
        self.add_via_button(dialog, [red, green])
        with patch('PyQt6.QtWidgets.QMessageBox.information') as info:
            self.add_via_button(dialog, [another_red, blue])
            info.assert_called_once()
        self.assertEqual(dialog.channel_assignment(), {'Red': red, 'Green': green, 'Blue': blue})
        for channel, path in dialog.channel_assignment().items():
            self.assertEqual(dialog._edits[channel].text(), str(path))
            self.assertEqual(dialog._browse_buttons[channel].text(), 'Change…')
        dialog._set_channel('Red', None)
        self.add_via_button(dialog, [another_red])
        self.assertEqual(dialog.channel_assignment()['Red'], another_red)


if __name__ == '__main__':
    unittest.main()
