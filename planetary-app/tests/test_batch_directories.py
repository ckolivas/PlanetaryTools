"""Batch folder history, including restoration in a fresh application process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from planetary_tools.batch.pipeline import BatchWorkflow
from planetary_tools.ui.batch_dialog import BatchDialog


class BatchDirectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input_dir = self.root/'inputs'
        self.output_dir = self.root/'outputs'
        self.input_dir.mkdir()
        self.output_dir.mkdir()
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

    def dialog(self):
        dialog = BatchDialog()
        self.addCleanup(dialog.close)
        return dialog

    def select_input(self, dialog):
        with patch('planetary_tools.ui.batch_dialog.QFileDialog.getExistingDirectory',
                   return_value=str(self.input_dir)):
            dialog._pick_folder()

    def test_selected_folders_and_picker_starts_are_restored(self):
        dialog = self.dialog()
        self.select_input(dialog)
        with patch('planetary_tools.ui.batch_dialog.QFileDialog.getExistingDirectory',
                   return_value=str(self.output_dir)):
            dialog._pick_output()
        dialog.reject()
        reopened = self.dialog()
        self.assertEqual(reopened._input_folder, self.input_dir)
        self.assertEqual(reopened._input_label.text(), str(self.input_dir))
        self.assertEqual(reopened._output_dir.text(), str(self.output_dir))
        with patch('planetary_tools.ui.batch_dialog.QFileDialog.getExistingDirectory',
                   return_value='') as picker:
            reopened._pick_folder()
            self.assertEqual(picker.call_args.args[2], str(self.input_dir))
            reopened._pick_output()
            self.assertEqual(picker.call_args.args[2], str(self.output_dir))
        with patch('planetary_tools.ui.batch_dialog.QFileDialog.getOpenFileNames',
                   return_value=([], '')) as picker:
            reopened._pick_files()
            self.assertEqual(picker.call_args.args[2], str(self.input_dir))
        # Cancelling a picker must leave saved selections intact.
        after_cancel = self.dialog()
        self.assertEqual(after_cancel._input_folder, self.input_dir)
        self.assertEqual(after_cancel._output_dir.text(), str(self.output_dir))

    def test_typed_and_workflow_output_paths_persist_before_they_exist(self):
        dialog = self.dialog()
        future = self.output_dir/'new'/'processed'
        dialog._output_dir.setText(str(future))
        dialog.reject()
        reopened = self.dialog()
        self.assertEqual(reopened._output_dir.text(), str(future))
        with patch('planetary_tools.ui.batch_dialog.QFileDialog.getExistingDirectory',
                   return_value='') as picker:
            reopened._pick_output()
            self.assertEqual(picker.call_args.args[2], str(self.output_dir))
        workflow_output = self.root/'workflow-output'
        reopened._apply_workflow(BatchWorkflow(output_dir=str(workflow_output)))
        reopened.reject()
        again = self.dialog()
        self.assertEqual(again._output_dir.text(), str(workflow_output))
        again._output_dir.clear()
        again.reject()
        self.assertEqual(self.dialog()._output_dir.text(), '')

    def test_file_picker_history_does_not_select_the_whole_folder(self):
        dialog = self.dialog()
        self.select_input(dialog)
        other_dir = self.root/'other'
        other_dir.mkdir()
        path = other_dir/'saturn.png'
        path.touch()
        with patch('planetary_tools.ui.batch_dialog.QFileDialog.getOpenFileNames',
                   return_value=([str(path)], '')):
            dialog._pick_files()
        dialog.reject()
        # Workflows and ordinary Open share lastOpenDir; batch history is separate.
        from planetary_tools.ui.recent_files import remember_open_path
        remember_open_path(self.input_dir)
        reopened = self.dialog()
        self.assertIsNone(reopened._input_folder)
        self.assertEqual(reopened._input_files, [])
        self.assertEqual(reopened._input_start_directory(), str(other_dir))

    def test_missing_input_folder_falls_back_without_selecting_it(self):
        dialog = self.dialog()
        self.select_input(dialog)
        self.input_dir.rmdir()
        with patch('planetary_tools.ui.batch_dialog.last_open_directory', return_value=str(self.root)):
            reopened = self.dialog()
            self.assertIsNone(reopened._input_folder)
            self.assertEqual(reopened._input_start_directory(), str(self.root))

    def test_folder_history_survives_a_fresh_process(self):
        script = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from planetary_tools.ui.batch_dialog import BatchDialog
app = QApplication([])
app.setApplicationName("Planetary Tools")
app.setOrganizationName("PlanetaryTools")
root = Path(sys.argv[2])
with patch("planetary_tools.core.presets.PRESET_DIR", root/"child-presets"):
    dialog = BatchDialog()
    if sys.argv[1] == "write":
        with patch("planetary_tools.ui.batch_dialog.QFileDialog.getExistingDirectory",
                   return_value=str(root/"inputs")):
            dialog._pick_folder()
        dialog._output_dir.setText(str(root/"outputs"))
        dialog.reject()
    else:
        print(json.dumps([str(dialog._input_folder), dialog._output_dir.text()]))
        dialog.reject()
'''
        env = dict(os.environ, XDG_CONFIG_HOME=str(self.root/'child-config'),
                   QT_QPA_PLATFORM='offscreen',
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        for phase in ('write', 'read'):
            result = subprocess.run([sys.executable, '-c', script, phase, str(self.root)],
                                    env=env, capture_output=True, text=True, check=True, timeout=20)
        self.assertEqual(json.loads(result.stdout), [str(self.input_dir), str(self.output_dir)])


if __name__ == '__main__':
    unittest.main()
