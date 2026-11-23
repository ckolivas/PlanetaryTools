"""Batch "Current workflow" rebuilds the edits applied to the open image, in order."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from planetary_tools.batch.pipeline import apply_pipeline, load_workflow
from planetary_tools.core.document import ImageDocument
from planetary_tools.core.undo import AppliedOperation, UndoStack
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.rotate_dialog import RotateImageDialog
from planetary_tools.ui.scale_dialog import ScaleImageDialog


def planet():
    y, x = np.indices((80, 112))
    image = np.full((80, 112, 3), .01, dtype=np.float32)
    disk = (x-60)**2 + (y-38)**2 < 22**2
    image[disk] = (.55, .4, .25)
    image[disk & (np.abs(y-38) < 3)] = (.3, .25, .2)
    return image


def op(label, fid=None, **params):
    return AppliedOperation(label, fid, params)


class AppliedOperationStackTests(unittest.TestCase):
    def test_follows_undo_redo_clear_and_outlives_snapshot_depth(self):
        data = np.zeros((2, 2), dtype=np.float32)
        stack = UndoStack(max_depth=2)
        for i in range(4):
            stack.push(data, True, f'Step {i}', op(f'Step {i}', 'levels', n=i))
        self.assertEqual([o.label for o in stack.applied_operations()],
                         ['Step 0', 'Step 1', 'Step 2', 'Step 3'])
        stack.undo(data, True)
        self.assertEqual([o.label for o in stack.applied_operations()][-1], 'Step 2')
        stack.redo(data, True)
        self.assertEqual([o.label for o in stack.applied_operations()][-1], 'Step 3')
        stack.undo(data, True)
        stack.push(data, True, 'Caption')
        self.assertEqual([(o.label, o.filter_id) for o in stack.applied_operations()][-2:],
                         [('Step 2', 'levels'), ('Caption', None)])
        self.assertFalse(stack.can_redo())
        copied = stack.applied_operations()
        copied[0].params['n'] = 99
        self.assertEqual(stack.applied_operations()[0].params['n'], 0)
        stack.clear()
        self.assertEqual(stack.applied_operations(), [])


class CurrentWorkflowTests(unittest.TestCase):
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
        self.window = MainWindow()
        self.addCleanup(self.close_window)
        self.source = planet()
        self.window._set_document(ImageDocument(self.source.copy()))

    def close_window(self):
        # Never block teardown on the unsaved-changes prompt.
        self.window._document.modified = False
        self.window.close()

    def apply_panel(self, run, params):
        errors = []

        def accept():
            try:
                panel = self.window._active_filter_dlg
                panel.set_params({**panel.get_params(), **params})
                panel.params_changed.emit()
                panel.preview.setChecked(False)
                panel._accept()
            except BaseException as exc:
                errors.append(exc)
                self.window._active_filter_dlg._reject()
        QTimer.singleShot(0, accept)
        run()
        self.assertFalse(errors, errors)

    def scale(self, percent):
        def run(dlg):
            dlg._percent.setValue(percent)
            return QDialog.DialogCode.Accepted
        with patch.object(ScaleImageDialog, 'exec', run):
            self.window._run_scale_image()

    def rotate(self, angle, flip=False):
        def run(dlg):
            dlg._angle.setValue(angle)
            dlg._flip_horizontal.setChecked(flip)
            return QDialog.DialogCode.Accepted
        with patch.object(RotateImageDialog, 'exec', run):
            self.window._run_rotate_image()

    def imported(self, existing_steps=0):
        batch = BatchDialog(current_operations=self.window._undo.stack.applied_operations())
        self.addCleanup(batch.close)
        for _ in range(existing_steps):
            batch._add_step()
        return batch

    def test_imports_applied_edits_in_order_and_reproduces_the_image(self):
        w = self.window
        self.apply_panel(w._run_wavelet_sharpen, {'fine': 40.0, 'medium': 10.0})
        self.scale(50.0)
        self.rotate(12.0)
        w._undo_action()  # the rotation is no longer applied
        self.apply_panel(w._run_saturation_vibrance, {'saturation': 1.4})
        batch = self.imported()
        self.assertTrue(batch._current_wf_btn.isEnabled())
        with patch.object(QMessageBox, 'information') as info:
            batch._current_wf_btn.click()
        info.assert_not_called()
        self.assertEqual([s.filter_id for s in batch._steps],
                         ['wavelet_sharpen', 'scale_image', 'saturation_vibrance'])
        self.assertEqual(batch._step_list.count(), 3)
        rebuilt = apply_pipeline(self.source.copy(), False, batch._steps)
        np.testing.assert_allclose(rebuilt, w._document.data, atol=1e-6)

    def test_skips_edits_without_batch_steps_and_confirms_replacing(self):
        w = self.window
        self.rotate(90.0, flip=True)
        w._commit_filter('Caption', w._document.data.copy())
        batch = self.imported()
        with patch.object(QMessageBox, 'information') as info:
            batch._current_wf_btn.click()
        self.assertEqual(batch._steps, [])
        self.assertIn('Rotate Image', info.call_args.args[2])
        self.assertIn('Caption', info.call_args.args[2])
        self.scale(50.0)
        batch = self.imported(existing_steps=2)
        for reply, expected in ((QMessageBox.StandardButton.Cancel, 2),
                                (QMessageBox.StandardButton.Yes, 1)):
            with patch.object(QMessageBox, 'question', return_value=reply), \
                    patch.object(QMessageBox, 'information') as info:
                batch._current_wf_btn.click()
            self.assertEqual(len(batch._steps), expected)
        self.assertEqual(batch._steps[0].filter_id, 'scale_image')
        self.assertIn('Caption', info.call_args.args[2])

    def save_as(self, path, filter_name, tick):
        """Save As through the dialog, ticking its workflow box or not."""
        seen = {}

        def choose(*args, option=None, **kwargs):
            seen['initial'] = option.isChecked()
            option.setChecked(tick)
            return str(path), filter_name
        with patch('planetary_tools.ui.file_dialog.QFileDialog.getSaveFileName',
                   side_effect=choose):
            self.assertTrue(self.window._try_save_document_as())
        return seen['initial']

    def test_save_writes_the_current_workflow_beside_the_image(self):
        w = self.window
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [p.unlink() for p in root.iterdir()] and root.rmdir())
        self.apply_panel(w._run_wavelet_sharpen, {'fine': 40.0, 'medium': 10.0})
        self.scale(50.0)
        w._commit_filter('Caption', w._document.data.copy())
        # Off by default: the image alone.
        self.assertFalse(self.save_as(root/'saturn.png', 'PNG 16-bit (*.png)', tick=False))
        self.assertEqual(sorted(p.name for p in root.iterdir()), ['saturn.png'])
        self.assertFalse(w._save_workflow_act.isChecked())
        # Ticked: the workflow beside it, as Current workflow would import it.
        self.assertFalse(self.save_as(root/'saturn.png', 'PNG 16-bit (*.png)', tick=True))
        self.assertTrue(w._save_workflow_act.isChecked())
        saved = root/'saturn.ptbatch'
        workflow, warnings = load_workflow(saved)
        self.assertEqual(warnings, [])
        batch = self.imported()
        with patch.object(QMessageBox, 'information'):  # the caption is skipped
            batch._current_wf_btn.click()
        self.assertEqual([(s.filter_id, s.params) for s in workflow.steps],
                         [(s.filter_id, s.params) for s in batch._steps])
        self.assertEqual([s.filter_id for s in workflow.steps], ['wavelet_sharpen', 'scale_image'])
        self.assertEqual(workflow.bit_depth, 16)
        self.assertIn('saturn.ptbatch', w._status.currentMessage())
        self.assertIn('1 edit with no batch step', w._status.currentMessage())
        # The choice is remembered: the next Save As starts ticked, and a
        # plain Save rewrites the workflow after a further edit.
        self.assertTrue(MainWindow()._save_workflow_act.isChecked())
        self.rotate(12.0)
        w._save_file()
        self.assertEqual([s.filter_id for s in load_workflow(saved)[0].steps],
                         ['wavelet_sharpen', 'scale_image', 'rotate_image'])
        # Unticked in the File menu: plain Save leaves the workflow alone.
        w._save_workflow_act.setChecked(False)
        saved.unlink()
        w._save_file()
        self.assertFalse(saved.exists())

    def test_save_without_batch_edits_writes_no_workflow(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [p.unlink() for p in root.iterdir()] and root.rmdir())
        self.window._commit_filter('Caption', self.window._document.data.copy())
        self.save_as(root/'moon.tif', 'TIFF float32 (*.tif *.tiff)', tick=True)
        self.assertEqual(sorted(p.name for p in root.iterdir()), ['moon.tif'])
        self.assertIn('no workflow saved', self.window._status.currentMessage())

    def test_save_as_dialog_shows_the_workflow_box(self):
        from PyQt6.QtWidgets import QCheckBox
        from planetary_tools.ui.file_dialog import QFileDialog
        dialog = QFileDialog(None, 'Save Image As', str(Path.home()), 'PNG (*.png)')
        self.addCleanup(dialog.deleteLater)
        box = QCheckBox('Save workflow with image (.ptbatch)')
        dialog.add_option(box)
        dialog.show()
        self.app.processEvents()
        self.assertIs(box.window(), dialog)
        self.assertTrue(box.isVisible())
        dialog.reject()

    def test_disabled_without_edits_and_cleared_by_open_or_close(self):
        self.assertFalse(self.imported()._current_wf_btn.isEnabled())
        self.scale(50.0)
        self.assertTrue(self.imported()._current_wf_btn.isEnabled())
        self.window._set_document(ImageDocument(planet()))
        self.assertFalse(self.imported()._current_wf_btn.isEnabled())
        self.scale(50.0)
        with patch.object(self.window, '_confirm_unsaved_changes', return_value=True):
            self.window._close_document()
        self.window._set_document(ImageDocument(planet()))
        self.assertFalse(self.imported()._current_wf_btn.isEnabled())
        with patch('planetary_tools.ui.main_window.BatchDialog') as dialog:
            self.scale(50.0)
            self.window._run_batch()
        ops = dialog.call_args.kwargs['current_operations']
        self.assertEqual([(o.filter_id, o.params['percent']) for o in ops], [('scale_image', 50.0)])


if __name__ == '__main__':
    unittest.main()
