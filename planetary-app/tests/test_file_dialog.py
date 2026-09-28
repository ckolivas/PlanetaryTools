"""File-picker sections, navigation, persistent history and static API parity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl
from PyQt6.QtWidgets import QApplication, QFrame, QLineEdit, QListView

from planetary_tools.ui.file_dialog import QFileDialog
from planetary_tools.ui import recent_files


class FileDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folders = [self.root / name for name in ('first', 'second', 'third')]
        for folder in self.folders:
            folder.mkdir()
        self.ini = self.root / 'settings.ini'
        settings = patch.object(recent_files, '_settings', self.settings)
        settings.start()
        self.addCleanup(settings.stop)

    def settings(self):
        return QSettings(str(self.ini), QSettings.Format.IniFormat)

    def dialog(self, directory=None):
        dialog = QFileDialog(directory=str(directory or self.folders[0]))
        original = dialog.sidebarUrls()
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.reject)
        self.addCleanup(lambda: dialog.setSidebarUrls(original))
        return dialog

    def test_history_is_bounded_deduplicated_and_skips_missing_folders(self):
        for folder in self.folders:
            recent_files.remember_directory(folder)
        recent_files.remember_directory(self.folders[0] / '.')
        self.folders[1].rmdir()
        self.assertEqual(recent_files.list_recent_directories(),
                         [str(self.folders[0]), str(self.folders[2])])
        for i in range(15):
            folder = self.root / str(i)
            folder.mkdir()
            recent_files.remember_directory(folder)
        history = recent_files.list_recent_directories()
        self.assertEqual(len(history), 10)
        self.assertEqual(history[0], str(self.root / '14'))

    def test_legacy_history_seeds_sidebar_and_open_save_paths_share_it(self):
        settings = self.settings()
        settings.setValue('lastOpenDir', str(self.folders[0]))
        settings.setValue('lastSaveDir', str(self.folders[1]))
        settings.setValue('recentFiles', [str(self.folders[2]/'old.png')])
        self.assertEqual(recent_files.list_recent_directories(), list(map(str, self.folders)))
        source = self.folders[2]/'image.png'
        source.touch()
        recent_files.remember_open_path(source)
        recent_files.remember_save_path(self.folders[1]/'new.png')
        self.assertEqual(recent_files.list_recent_directories(),
                         list(map(str, (self.folders[1], self.folders[2], self.folders[0]))))

    def test_sections_keep_qt_favourites_and_recent_click_navigates(self):
        recent_files.remember_directory(self.folders[1])
        dialog = self.dialog()
        sidebar = dialog.findChild(QListView, 'sidebar')
        self.assertIsNotNone(sidebar)
        separator = dialog.findChild(QFrame, 'recentDirectoriesSeparator')
        self.assertEqual(separator.frameShape(), QFrame.Shape.HLine)
        layout = sidebar.parentWidget().layout()
        self.assertLess(layout.indexOf(sidebar), layout.indexOf(separator))
        self.assertLess(layout.indexOf(separator), layout.indexOf(dialog.recent_directories))
        item = dialog.recent_directories.item(0)
        self.assertEqual(item.toolTip(), str(self.folders[1]))
        self.assertEqual(item.data(Qt.ItemDataRole.UserRole), str(self.folders[1]))
        dialog.recent_directories.itemClicked.emit(item)
        self.assertEqual(dialog.directory().absolutePath(), str(self.folders[1]))
        dialog.reject()
        self.assertEqual(recent_files.list_recent_directories(), [str(self.folders[1])])

    def test_favourites_and_recents_are_separate_and_survive_restart(self):
        dialog = self.dialog()
        favourites = [QUrl.fromLocalFile(str(self.folders[0]))]
        dialog.setSidebarUrls(favourites)
        dialog.reject()
        recent_files.remember_directory(self.folders[2])
        reopened = self.dialog()
        self.assertEqual(reopened.sidebarUrls(), favourites)
        self.assertEqual(reopened.recent_directories.item(0).toolTip(), str(self.folders[2]))
        script = '''
import json, sys
from PyQt6.QtCore import QSettings
from planetary_tools.ui import recent_files
recent_files._settings = lambda: QSettings(sys.argv[1], QSettings.Format.IniFormat)
print(json.dumps([recent_files.list_recent_directories(), recent_files._settings().value('fileDialog/favourites')]))
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.ini)],
                                capture_output=True, text=True, check=True)
        paths, urls = json.loads(result.stdout)
        self.assertEqual(paths[0], str(self.folders[2]))
        self.assertEqual(urls, [favourites[0].toString()])

    def choose(self, method, edit, *args, **kwargs):
        errors = []
        def interact():
            dialog = QApplication.activeModalWidget()
            try:
                self.assertIsInstance(dialog, QFileDialog)
                self.assertTrue(dialog.testOption(QFileDialog.Option.DontUseNativeDialog))
                self.assertIsNotNone(dialog.findChild(QFrame, 'recentDirectoriesSeparator'))
                edit(dialog)
            except BaseException as exc:
                errors.append(exc)
                dialog.reject()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: QApplication.activeModalWidget().reject())
        timer.start(3000)
        QTimer.singleShot(0, interact)
        try:
            result = method(*args, **kwargs)
        finally:
            timer.stop()
        self.assertFalse(errors, errors)
        return result

    def test_open_multiple_save_and_directory_wrappers_keep_selections_and_filters(self):
        a, b = self.folders[1]/'a.png', self.folders[1]/'b.png'
        a.touch(); b.touch()
        filters = 'PNG 8-bit (*.png);;PNG 16-bit (*.png);;All files (*)'
        def accept_open(dialog):
            self.assertEqual(dialog.fileMode(), QFileDialog.FileMode.ExistingFile)
            dialog.selectFile(str(a))
            dialog.accept()
        selected, selected_filter = self.choose(QFileDialog.getOpenFileName, accept_open,
            None, 'Open', str(a.parent), filters, 'PNG 16-bit (*.png)')
        self.assertEqual(selected, str(a))
        self.assertEqual(selected_filter, 'PNG 16-bit (*.png)')
        self.assertEqual(recent_files.list_recent_directories()[0], str(a.parent))
        def accept_multiple(dialog):
            self.assertEqual(dialog.fileMode(), QFileDialog.FileMode.ExistingFiles)
            dialog.findChild(QLineEdit, 'fileNameEdit').setText('"a.png" "b.png"')
            dialog.accept()
        selected, _ = self.choose(QFileDialog.getOpenFileNames, accept_multiple,
                                  None, 'Add files', str(a.parent), filters)
        self.assertCountEqual(selected, [str(a), str(b)])
        destination = self.folders[2]/'new.png'
        def accept_save(dialog):
            self.assertEqual(dialog.acceptMode(), QFileDialog.AcceptMode.AcceptSave)
            self.assertEqual(dialog.selectedFiles(), [str(destination)])
            dialog.accept()
        selected, _ = self.choose(QFileDialog.getSaveFileName, accept_save,
                                  None, 'Save', str(destination), filters)
        self.assertEqual(selected, str(destination))
        self.assertEqual(recent_files.list_recent_directories()[0], str(destination.parent))
        selected = self.choose(QFileDialog.getExistingDirectory, lambda d: d.accept(),
                                None, 'Folder', str(self.folders[0]))
        self.assertEqual(selected, str(self.folders[0]))
        self.assertEqual(recent_files.list_recent_directories()[0], selected)
        before = recent_files.list_recent_directories()
        result = self.choose(QFileDialog.getSaveFileName, lambda d: d.reject(),
                              None, 'Cancel', str(a), filters)
        self.assertEqual(result, ('', ''))
        self.assertEqual(recent_files.list_recent_directories(), before)


if __name__ == '__main__':
    unittest.main()
