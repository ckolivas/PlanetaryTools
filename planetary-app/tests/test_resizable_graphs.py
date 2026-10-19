"""Floating graph panels grow and retain accurate mouse-to-value mapping."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow

from planetary_tools.ui.curves_dialog import CurvesDialog
from planetary_tools.ui.dialogs import FILTER_PANEL_WIDTH, LevelsDialog
from planetary_tools.ui.fft_denoise_dialog import FftDenoiseDialog
from planetary_tools.ui.main_window import _ToolDock


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


if __name__ == '__main__':
    unittest.main()
