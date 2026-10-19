"""Floating graph panels grow and retain accurate mouse-to-value mapping."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QSettings, QSize, Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QWindow
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractButton, QApplication, QMainWindow

from planetary_tools.ui.curves_dialog import CurvesDialog
from planetary_tools.ui.dialogs import FILTER_PANEL_WIDTH, LevelsDialog
from planetary_tools.ui.fft_denoise_dialog import FftDenoiseDialog
from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.main_window import MainWindow, _ToolDock


class ResizableGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        p = patch('planetary_tools.core.presets.PRESET_DIR', Path(folder.name))
        p.start()
        self.addCleanup(p.stop)
        self.settings_path = str(Path(folder.name) / 'settings.ini')
        p = patch('planetary_tools.ui.main_window.QSettings', self.settings)
        p.start()
        self.addCleanup(p.stop)

    def settings(self):
        return QSettings(self.settings_path, QSettings.Format.IniFormat)

    def floating_panel(self, cls):
        window = QMainWindow()
        self.addCleanup(window.close)
        window.resize(1000, 800)
        panel = cls()
        dock = _ToolDock(window)
        dock.setWidget(panel)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.set_panel_width(max(FILTER_PANEL_WIDTH, panel.sizeHint().width()))
        window.show()
        dock.setFloating(True)
        self.app.processEvents()
        return window, dock, panel

    def test_graphs_grow_both_axes_and_shrink_after_redocking(self):
        for cls, attr in ((CurvesDialog, 'editor'), (FftDenoiseDialog, 'graph'),
                          (LevelsDialog, '_histogram')):
            with self.subTest(panel=cls.__name__):
                window, dock, panel = self.floating_panel(cls)
                graph = getattr(panel, attr)
                if cls is LevelsDialog:
                    graph = graph._plot
                params = panel.get_params()
                before = graph.size()
                compact_width = dock.width()
                dock.resize(compact_width + 300, dock.height() + 240)
                self.app.processEvents()
                self.assertGreater(graph.width(), before.width() + 250)
                self.assertGreater(graph.height(), before.height() + 180)
                self.assertEqual(panel.get_params(), params)
                dock.setFloating(False)
                self.app.processEvents()
                self.assertEqual(dock.width(), compact_width)
                self.assertLess(graph.width(), before.width() + 10)
                self.assertEqual(panel.get_params(), params)
                dock.setFloating(True)
                self.app.processEvents()
                self.assertGreater(dock.maximumWidth(), dock.minimumWidth())

    def test_curve_clicks_and_drags_use_enlarged_plot_coordinates(self):
        _, dock, panel = self.floating_panel(CurvesDialog)
        editor = panel.editor
        small_step = 1 / editor.plot_rect().width()
        dock.resize(dock.width() + 400, dock.height() + 250)
        self.app.processEvents()
        self.assertLess(1 / editor.plot_rect().width(), small_step)
        point = editor._position(.4, .7).toPoint()
        QTest.mouseClick(editor, Qt.MouseButton.LeftButton, pos=point)
        x, y, _ = editor.curve['points'][editor.selected]
        self.assertAlmostEqual(x, .4, delta=1/editor.plot_rect().width())
        self.assertAlmostEqual(y, .7, delta=1/editor.plot_rect().height())
        target = editor._position(.55, .6).toPoint()
        QTest.mousePress(editor, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(editor, target)
        QTest.mouseRelease(editor, Qt.MouseButton.LeftButton, pos=target)
        x, y, _ = editor.curve['points'][editor.selected]
        self.assertAlmostEqual(x, .55, delta=1/editor.plot_rect().width())
        self.assertAlmostEqual(y, .6, delta=1/editor.plot_rect().height())

    def test_fft_marker_and_removed_signal_follow_resize_without_reanalysis(self):
        _, dock, panel = self.floating_panel(FftDenoiseDialog)
        source = np.linspace(0, .5, 64*96, dtype=np.float32).reshape(64, 96)
        panel.set_input_brightness(source, True)
        panel.update_histogram_display(source*.9)
        panel.show_difference.setChecked(True)
        self.app.processEvents()
        before = panel.difference.pixmap().size()
        with patch('planetary_tools.ui.fft_denoise_dialog.analyse_spectrum',
                   side_effect=AssertionError('Resize must not reanalyse pixels')):
            dock.resize(dock.width() + 400, dock.height() + 300)
            self.app.processEvents()
        after = panel.difference.pixmap().size()
        self.assertGreater(after.width(), before.width())
        self.assertGreater(after.height(), before.height())
        self.assertAlmostEqual(after.width()/after.height(), 96/64, delta=.02)
        graph = panel.graph
        for frequency in (-10., 0., 60., 100.):
            self.assertAlmostEqual(graph.position_frequency(graph.frequency_x(frequency)), frequency)
        y = int(graph.plot_rect().center().y())
        QTest.mousePress(graph, Qt.MouseButton.LeftButton,
                         pos=QPoint(round(graph.frequency_x(60)), y))
        target = QPoint(round(graph.frequency_x(67)), y)
        QTest.mouseMove(graph, target)
        QTest.mouseRelease(graph, Qt.MouseButton.LeftButton, pos=target)
        self.assertAlmostEqual(panel.start.value(), 67., delta=.2)
        panel.update_histogram_display(None)
        dock.resize(dock.width()-100, dock.height()-100)
        self.app.processEvents()
        self.assertIn('Enable preview', panel.difference.text())

    def test_histogram_hover_bins_use_new_width(self):
        _, dock, panel = self.floating_panel(LevelsDialog)
        panel.set_input_brightness(np.full((16, 16, 3), .25, dtype=np.float32), False)
        dock.resize(dock.width()+400, dock.height()+200)
        self.app.processEvents()
        plot = panel._histogram._plot
        for fraction, expected in ((0., 0), (.5, 128), (1., 255)):
            x = plot._MARGIN_LEFT + fraction*(plot.width()-plot._MARGIN_LEFT-plot._MARGIN_RIGHT)
            self.assertEqual(plot._bin_at_x(x), expected)

    def test_ten_pixel_border_leaves_controls_and_docked_panel_accessible(self):
        _, dock, panel = self.floating_panel(CurvesDialog)
        border = dock._resize_border
        w, h = dock.width(), dock.height()
        for point in (QPoint(9, h//2), QPoint(w-10, h//2),
                      QPoint(w//2, 9), QPoint(w//2, h-10), QPoint(9, h-10)):
            self.assertIs(dock.childAt(point), border)
        for point in (QPoint(10, h//2), QPoint(w-11, h//2),
                      QPoint(w//2, 10), QPoint(w//2, h-11)):
            self.assertIsNot(dock.childAt(point), border)
        editor_center = panel.editor.mapTo(dock, panel.editor.rect().center())
        self.assertIs(dock.childAt(editor_center), panel.editor)
        for name in ('qt_dockwidget_floatbutton', 'qt_dockwidget_closebutton'):
            button = dock.findChild(QAbstractButton, name)
            if button.isVisible():
                for point in (button.rect().topLeft(), button.rect().center()):
                    self.assertIs(dock.childAt(button.mapTo(dock, point)), button)
        dock.setFloating(False)
        self.app.processEvents()
        self.assertTrue(border.isHidden())

    def test_saved_size_survives_redocking_and_new_dock(self):
        _, dock, panel = self.floating_panel(CurvesDialog)
        dock.set_panel_width(max(FILTER_PANEL_WIDTH, panel.sizeHint().width()), 'CurvesDialog')
        size = QSize(640, 720)
        dock.resize(size)
        self.app.processEvents()
        dock.setFloating(False)
        self.app.processEvents()
        self.assertEqual(self.settings().value('floatingPanels/CurvesDialog/size'), size)
        dock.setFloating(True)
        self.app.processEvents()
        self.assertEqual(dock.size(), size)
        dock.close()
        _, new_dock, new_panel = self.floating_panel(CurvesDialog)
        new_dock.set_panel_width(max(FILTER_PANEL_WIDTH, new_panel.sizeHint().width()), 'CurvesDialog')
        self.app.processEvents()
        self.assertEqual(new_dock.size(), size)

    def test_each_tool_reopens_at_own_size_after_cleanup_and_restart(self):
        wanted = {'CurvesDialog': QSize(650, 720), 'FftDenoiseDialog': QSize(570, 690),
                  'CropImageDialog': QSize(590, 710)}
        # Exercise the actual nested tool loop, cleanup, and a new main window.
        for restart in (False, True):
            window = MainWindow()
            window._set_document(ImageDocument(np.full((32, 48, 3), .25, dtype=np.float32)))
            window.show()
            self.addCleanup(window.close)
            for key in list(wanted) * 2:
                errors = []
                def interact():
                    dock = window._filter_dock
                    try:
                        dock.setFloating(True)
                        self.app.processEvents()
                        if restart or self.settings().contains(f'floatingPanels/{key}/size'):
                            self.assertEqual(dock.size(), wanted[key])
                        dock.resize(wanted[key])
                        self.app.processEvents()
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        window._active_filter_dlg._reject()
                QTimer.singleShot(20, interact)
                if key == 'CropImageDialog':
                    window._run_crop_image()
                else:
                    cls = CurvesDialog if key == 'CurvesDialog' else FftDenoiseDialog
                    window._run_filter_dialog(cls(), key)
                self.assertFalse(errors, errors)
                self.assertEqual(self.settings().value(f'floatingPanels/{key}/size'), wanted[key])
            window.close()

    def test_invalid_or_oversized_saved_sizes(self):
        _, dock, panel = self.floating_panel(CurvesDialog)
        width = max(FILTER_PANEL_WIDTH, panel.sizeHint().width())
        for saved in ('bad size', QSize(-1, 100), QSize(0, 0)):
            self.settings().setValue('floatingPanels/Invalid/size', saved)
            dock.set_panel_width(width, 'Invalid')
            self.assertIsNone(dock._floating_size)
            dock.finish_panel()
        self.settings().setValue('floatingPanels/Large/size', QSize(10000, 10000))
        dock.set_panel_width(width, 'Large')
        self.app.processEvents()
        available = dock.screen().availableGeometry().size()
        self.assertLessEqual(dock.width(), available.width())
        self.assertLessEqual(dock.height(), available.height())
        self.assertGreaterEqual(dock.width(), dock.minimumWidth())

    def test_border_requests_native_resize_for_edges_and_corners(self):
        _, dock, _ = self.floating_panel(CurvesDialog)
        border = dock._resize_border
        w, h = border.width(), border.height()
        left, right = Qt.Edge.LeftEdge, Qt.Edge.RightEdge
        top, bottom = Qt.Edge.TopEdge, Qt.Edge.BottomEdge
        points = ((9, h//2, left), (w-10, h//2, right),
                  (w//2, 9, top), (w//2, h-10, bottom),
                  (9, 9, left | top), (w-10, 9, right | top),
                  (9, h-10, left | bottom), (w-10, h-10, right | bottom))
        for x, y, edges in points:
            with self.subTest(edges=edges), patch.object(QWindow, 'startSystemResize', return_value=True) as resize:
                QTest.mouseClick(border, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
                resize.assert_called_once_with(edges)
                self.assertIsNone(border._drag)

    def test_border_fallback_drag_and_minimum_size_keep_opposite_edges_fixed(self):
        _, dock, _ = self.floating_panel(CurvesDialog)
        border = dock._resize_border
        dock.resize(dock.width()+200, dock.height()+200)
        self.app.processEvents()

        def move_from(origin, delta):
            global_pos = origin + delta
            event = QMouseEvent(QEvent.Type.MouseMove,
                               QPointF(border.mapFromGlobal(global_pos)), QPointF(global_pos),
                               Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(border, event)
            self.app.processEvents()

        for left in (False, True):
            for top in (False, True):
                with self.subTest(left=left, top=top), patch.object(QWindow, 'startSystemResize', return_value=False):
                    dock.move(300, 300)
                    self.app.processEvents()
                    initial = dock.geometry()
                    point = QPoint(9 if left else border.width()-10,
                                   9 if top else border.height()-10)
                    origin = border.mapToGlobal(point)
                    QTest.mousePress(border, Qt.MouseButton.LeftButton, pos=point)
                    move_from(origin, QPoint(-50 if left else 50, -40 if top else 40))
                    self.assertEqual(dock.width(), initial.width()+50)
                    self.assertEqual(dock.height(), initial.height()+40)
                    self.assertEqual(dock.geometry().right() if left else dock.geometry().x(),
                                     initial.right() if left else initial.x())
                    self.assertEqual(dock.geometry().bottom() if top else dock.geometry().y(),
                                     initial.bottom() if top else initial.y())
                    move_from(origin, QPoint(5000 if left else -5000, 5000 if top else -5000))
                    self.assertEqual(dock.size(), dock.minimumSize().expandedTo(dock.minimumSizeHint()))
                    self.assertEqual(dock.geometry().right() if left else dock.geometry().x(),
                                     initial.right() if left else initial.x())
                    self.assertEqual(dock.geometry().bottom() if top else dock.geometry().y(),
                                     initial.bottom() if top else initial.y())
                    QTest.mouseRelease(border, Qt.MouseButton.LeftButton)
                    self.assertIsNone(border._drag)


if __name__ == '__main__':
    unittest.main()
