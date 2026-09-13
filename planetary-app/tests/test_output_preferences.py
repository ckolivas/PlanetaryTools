"""Export choices persist, and selected image formats control output filenames."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox

from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.animate_dialog import AnimateDialog
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.field_derotate_dialog import FieldDerotateDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.recent_files import remember_save_filter, last_save_filter


class OutputPreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        def settings():
            return QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        self.settings = settings
        for target, replacement in (
            ('planetary_tools.ui.recent_files._settings', settings),
            ('planetary_tools.ui.batch_dialog.QSettings', settings),
            ('planetary_tools.core.presets.PRESET_DIR', self.root/'presets'),
        ):
            mock = patch(target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def window(self):
        window = MainWindow()
        window._set_document(ImageDocument(np.full((8,8,3), .1, dtype=np.float32),
                                            path=self.root/'source.tif'))
        self.addCleanup(window.close)
        return window

    def test_save_as_and_decompose_share_last_type_and_use_its_extension(self):
        window = self.window()
        selected = 'PNG 16-bit (*.png)'
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(str(self.root/'output.tif'), selected)), patch.object(window, '_write_document') as save:
            self.assertTrue(window._try_save_document_as())
            save.assert_called_once_with(str(self.root/'output.png'), 16)
        self.assertEqual(last_save_filter(), selected)
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(str(self.root/'channels.png'), 'TIFF float32 (*.tif *.tiff)')) as picker, \
             patch('planetary_tools.ui.main_window.save_channel') as save:
            window._run_rgb_decompose()
            self.assertEqual(picker.call_args.args[4], selected)
            self.assertEqual([call.args[1].suffix for call in save.call_args_list], ['.tif']*3)
            self.assertTrue(all(call.kwargs['bit_depth']==32 for call in save.call_args_list))
        reopened = self.window()
        self.assertEqual(reopened._default_save_as_filter(), 'TIFF float32 (*.tif *.tiff)')
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName', return_value=('', 'JPEG (*.jpg *.jpeg)')):
            self.assertFalse(reopened._try_save_document_as())
        self.assertEqual(last_save_filter(), 'TIFF float32 (*.tif *.tiff)')

    def test_changed_suffix_does_not_overwrite_without_confirmation(self):
        window = self.window()
        destination = self.root/'output.png'
        destination.write_bytes(b'keep')
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName', return_value=(str(self.root/'output.tif'), 'PNG 8-bit (*.png)')), \
             patch('PyQt6.QtWidgets.QMessageBox.question', return_value=QMessageBox.StandardButton.No), \
             patch.object(window, '_write_document') as save:
            self.assertFalse(window._try_save_document_as())
            save.assert_not_called()
        self.assertEqual(destination.read_bytes(), b'keep')

    def test_animation_picker_and_format_control_remember_each_other(self):
        dialog = AnimateDialog()
        self.addCleanup(dialog.close)
        dialog._format.setCurrentIndex(dialog._format.findData('webp'))
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName',
                   return_value=(str(self.root/'movie.webp'), 'Animated PNG (*.png)')) as picker:
            dialog._browse_output()
            self.assertEqual(picker.call_args.args[4], 'WebP (*.webp)')
        self.assertEqual(dialog._fmt(), 'apng')
        self.assertEqual(Path(dialog._output.text()).suffix, '.png')
        reopened = AnimateDialog()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened._fmt(), 'apng')
        self.assertFalse(reopened._gif_quality.isEnabled())

    def test_invalid_saved_options_fall_back(self):
        settings = self.settings()
        for key in ('batchDepth', 'alignDepth', 'animationFormat'):
            settings.setValue('outputOptions/'+key, 'invalid')
        remember_save_filter('obsolete filter')
        dialogs = [BatchDialog(), FieldDerotateDialog(), AnimateDialog()]
        for dialog in dialogs:
            self.addCleanup(dialog.close)
        self.assertEqual(dialogs[0]._bit_depth.currentData(), 32)
        self.assertEqual(dialogs[1]._bit_depth.currentData(), 32)
        self.assertEqual(dialogs[2]._fmt(), 'gif')
        self.assertEqual(self.window()._default_save_as_filter(), 'PNG 16-bit (*.png)')

    def test_output_options_survive_a_fresh_process(self):
        script = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from planetary_tools.ui.animate_dialog import AnimateDialog
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.field_derotate_dialog import FieldDerotateDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.recent_files import remember_save_filter
app = QApplication([])
app.setOrganizationName('PlanetaryTools')
app.setApplicationName('Planetary Tools')
with patch('planetary_tools.core.presets.PRESET_DIR', Path(sys.argv[2])/'presets'):
    batch, align, animate, window = BatchDialog(), FieldDerotateDialog(), AnimateDialog(), MainWindow()
    if sys.argv[1] == 'write':
        batch._bit_depth.setCurrentIndex(batch._bit_depth.findData(16))
        align._bit_depth.setCurrentIndex(align._bit_depth.findData(8))
        animate._format.setCurrentIndex(animate._format.findData('webp'))
        remember_save_filter('PNG 8-bit (*.png)')
    else:
        print(json.dumps([batch._bit_depth.currentData(), align._bit_depth.currentData(),
                          animate._fmt(), window._default_save_as_filter()]))
    for dialog in (batch, align, animate, window): dialog.close()
'''
        env = dict(os.environ, XDG_CONFIG_HOME=str(self.root/'config'),
                   QT_QPA_PLATFORM='offscreen', PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        for phase in ('write', 'read'):
            result = subprocess.run([sys.executable, '-c', script, phase, str(self.root)],
                                    env=env, capture_output=True, text=True, check=True, timeout=20)
        self.assertEqual(json.loads(result.stdout), [16, 8, 'webp', 'PNG 8-bit (*.png)'])


if __name__ == '__main__':
    unittest.main()
