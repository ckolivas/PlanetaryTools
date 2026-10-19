"""Native URL drops and reordering must update the inputs actually processed."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QApplication

from planetary_tools.ui.animate_dialog import AnimateDialog
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.compose_dialog import RGBComposeDialog
from planetary_tools.ui.field_derotate_dialog import FieldDerotateDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.file_drop import image_paths, row_order


def mime_for(paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


class DragDropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.paths = [self.root/'one'/'frame1.PNG', self.root/'two'/'frame2.tif', self.root/'two'/'frame3.png']
        for path in self.paths:
            path.parent.mkdir(exist_ok=True)
            path.touch()
        def settings():
            return QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        for target, value in (
            ('planetary_tools.ui.batch_dialog.QSettings', settings),
            ('planetary_tools.ui.recent_files._settings', settings),
            ('planetary_tools.core.presets.PRESET_DIR', self.root/'presets'),
        ):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)

    def dialog(self, cls):
        dialog = cls()
        self.addCleanup(dialog.close)
        dialog.show()
        self.app.processEvents()
        return dialog

    def drop(self, target, paths, accepted=True):
        mime = mime_for(paths)
        actions = Qt.DropAction.CopyAction | Qt.DropAction.MoveAction
        args = (actions, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
        enter = QDragEnterEvent(QPoint(5, 5), *args)
        QApplication.sendEvent(target, enter)
        self.assertEqual(enter.isAccepted(), accepted)
        if accepted:
            move = QDragMoveEvent(QPoint(5, 5), *args)
            QApplication.sendEvent(target, move)
            self.assertTrue(move.isAccepted())
            event = QDropEvent(QPointF(5, 5), *args)
            QApplication.sendEvent(target, event)
            self.assertTrue(event.isAccepted())
            self.assertEqual(event.dropAction(), Qt.DropAction.CopyAction)

    def test_local_images_only_and_aliases_deduplicated(self):
        invalid = self.root/'readme.txt'
        invalid.touch()
        mime = mime_for([*self.paths, self.paths[0], invalid, self.root, self.root/'missing.png'])
        mime.setUrls(mime.urls()+[QUrl('https://example.org/image.png')])
        self.assertEqual(image_paths(mime), self.paths)
        text = QMimeData()
        text.setText(str(self.paths[0]))
        self.assertEqual(image_paths(text), [])

    def test_drop_on_dialog_and_viewport_accumulates_and_remembers_directory(self):
        for cls in (BatchDialog, FieldDerotateDialog, AnimateDialog):
            with self.subTest(dialog=cls.__name__):
                dialog = self.dialog(cls)
                self.drop(dialog, [self.paths[0]])
                view = dialog._input_list if cls is BatchDialog else dialog._table
                self.drop(view.viewport(), [self.paths[1], self.paths[0]])
                self.assertEqual(view.model().rowCount(), 2)
                if cls is BatchDialog:
                    self.assertEqual(dialog._input_files, self.paths[:2])
                    self.assertEqual(dialog._input_start_directory(), str(self.paths[1].parent))
                else:
                    self.assertEqual([view.item(i, 0).toolTip() for i in range(2)],
                                     [str(p) for p in self.paths[:2]])
                self.drop(view.viewport(), [self.root], accepted=False)
                dialog._worker = Mock()
                dialog._worker.isRunning.return_value = True
                self.drop(view.viewport(), [self.paths[2]], accepted=False)
                self.drop(dialog, [self.paths[2]], accepted=False)
                dialog._worker = None

    def test_main_canvas_reuses_open_and_unsaved_guard_and_rejects_multi(self):
        window = self.dialog(MainWindow)
        with patch.object(window, '_open_path') as open_image:
            self.drop(window._canvas.viewport(), [self.paths[0]])
            open_image.assert_called_once_with(self.paths[0])
            self.drop(window._canvas.viewport(), self.paths[:2], accepted=False)
        with patch.object(window, '_confirm_unsaved_changes', return_value=False) as confirm, \
             patch('planetary_tools.ui.main_window.load_image') as load:
            self.drop(window._canvas.viewport(), [self.paths[1]])
            confirm.assert_called_once()
            load.assert_not_called()
        with patch.object(window, '_guard_filter_dialog', return_value=True) as guard, \
             patch('planetary_tools.ui.main_window.load_image') as load:
            self.drop(window, [self.paths[1]])
            guard.assert_called_once()
            load.assert_not_called()

    def test_compose_assigns_names_and_explicit_channel_drops_replace(self):
        red, green, blue = (self.root/name for name in ('saturn_R.png', 'saturn_G.png', 'saturn_B.png'))
        for path in (red, green, blue):
            path.touch()
        dialog = self.dialog(RGBComposeDialog)
        self.drop(dialog, [blue, red, green])
        self.assertEqual(dialog.channel_assignment(), {'Red': red, 'Green': green, 'Blue': blue})
        self.drop(dialog._edits['Red'], [self.paths[0]])
        self.assertEqual(dialog.channel_assignment()['Red'], self.paths[0])
        self.assertEqual(dialog._edits['Red'].text(), str(self.paths[0]))

    def reorder(self, view, rows, destination):
        view.clearSelection()
        for row in rows:
            if hasattr(view, 'selectRow'):
                view.selectRow(row)
            else:
                view.item(row).setSelected(True)
        last = view.visualRect(view.model().index(view.model().rowCount()-1, 0))
        point = QPointF(5, last.bottom()+5)
        if destination < view.model().rowCount():
            rect = view.visualRect(view.model().index(destination, 0))
            point = QPointF(rect.left()+3, rect.top()+1)
        event = Mock()
        event.source.return_value = view
        event.position.return_value = point
        event.mimeData.return_value = QMimeData()
        event.isAccepted.return_value = True
        view.dropEvent(event)
        return event

    def test_internal_drag_uses_own_move_and_keeps_processing_order(self):
        for cls in (BatchDialog, FieldDerotateDialog, AnimateDialog):
            with self.subTest(dialog=cls.__name__):
                dialog = self.dialog(cls)
                dialog._receive_files(self.paths)
                if cls is AnimateDialog:
                    dialog._motion_interpolation.setChecked(False)
                view = dialog._input_list if cls is BatchDialog else dialog._table
                if cls is FieldDerotateDialog:
                    dialog._ref_index = 0
                    reference = dialog._rows[0]
                self.reorder(view, [0], 3)
                expected = [self.paths[1], self.paths[2], self.paths[0]]
                actual = (dialog._input_files if cls is BatchDialog else
                          dialog._paths if cls is AnimateDialog else [r.path for r in dialog._rows])
                self.assertEqual(actual, expected)
                self.assertEqual(view.currentIndex().row(), 2)
                if cls is FieldDerotateDialog:
                    self.assertIs(dialog._rows[dialog._ref_index], reference)
                    self.assertTrue(view.item(2, 0).text().startswith('★'))
                with patch('planetary_tools.ui.file_drop.QDrag') as drag:
                    view.startDrag(Qt.DropAction.MoveAction)
                    drag.return_value.exec.assert_called_once_with(Qt.DropAction.MoveAction)
                    self.assertEqual(view.model().rowCount(), 3)

    def test_animation_interpolation_disables_manual_reorder(self):
        dialog = self.dialog(AnimateDialog)
        dialog._receive_files(self.paths)
        dialog._motion_interpolation.setChecked(True)
        with patch('planetary_tools.ui.file_drop.QDrag') as drag:
            dialog._table.startDrag(Qt.DropAction.MoveAction)
            drag.assert_not_called()
        dialog._reorder_files([2, 1, 0])
        self.assertEqual(dialog._paths, self.paths)

    def test_multi_row_permutations_preserve_items_and_relative_order(self):
        self.assertEqual(row_order(6, [1, 3], 6), [0, 2, 4, 5, 1, 3])
        self.assertEqual(row_order(6, [2, 3], 0), [2, 3, 0, 1, 4, 5])
        self.assertEqual(row_order(6, [2, 3], 3), list(range(6)))
        self.assertEqual(row_order(0, [], 0), [])


if __name__ == '__main__':
    unittest.main()
