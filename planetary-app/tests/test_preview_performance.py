"""Preview reuse must never apply stale pixels or duplicate filter evaluation."""
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtWidgets import QApplication

from planetary_tools.filters.registry import (
    apply_filter, output_filter_stats, apply_filter_and_output_stats, run_filter_raw,
)
from planetary_tools.ui.preview import PreviewController


class PreviewPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def controller(self):
        preview = PreviewController(debounce_ms=10)
        preview.start(np.ones((20,30), dtype=np.float32), True)
        self.addCleanup(lambda: preview.finish(False))
        return preview

    def wait_for(self, predicate):
        deadline = time.monotonic()+5
        while not predicate() and time.monotonic()<deadline:
            QTest.qWait(5)
        self.assertTrue(predicate())

    def test_preview_toggle_and_apply_reuse_completed_result(self):
        preview = self.controller()
        calls = []
        def filter_func(data, gray):
            calls.append(threading.get_ident())
            return data*2
        preview.set_filter_func(filter_func)
        preview.update_now()
        self.wait_for(lambda: np.all(preview.display_data()==2))
        preview.set_preview_enabled(False)
        np.testing.assert_array_equal(preview.display_data(), 1)
        preview.set_preview_enabled(True)
        preview.update_now()
        result = preview.finish(True)
        np.testing.assert_array_equal(result, 2)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0], threading.get_ident())

    def test_apply_uses_completed_worker_even_before_queued_signal_delivery(self):
        preview = self.controller()
        calls = []
        def func(data, _gray):
            calls.append(1)
            return data*3
        preview.set_filter_func(func)
        preview.update_now()
        result = preview.finish(True)
        np.testing.assert_array_equal(result, 3)
        self.assertEqual(calls, [1])
        self.app.processEvents()
        self.assertIsNone(preview.display_data())

    def test_settings_changed_during_debounce_are_applied(self):
        preview = self.controller()
        preview.set_filter_func(lambda data, _gray: data*2)
        preview.update_now()
        self.wait_for(lambda: np.all(preview.display_data()==2))
        preview.set_filter_func(lambda data, _gray: data*4)
        preview.schedule_update()
        np.testing.assert_array_equal(preview.finish(True), 4)

    def test_running_job_is_immutable_and_newest_job_runs_after_it(self):
        preview = self.controller()
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        calls = []
        def slow(data, gray):
            calls.append('old')
            started.set()
            if not release.wait(3):
                raise RuntimeError('test release timeout')
            return data*2
        preview.set_filter_func(slow)
        preview.update_now()
        self.assertTrue(started.wait(2))
        def latest(data, gray):
            calls.append('new')
            return data*5
        preview.set_filter_func(latest)
        preview.update_now()
        release.set()
        self.wait_for(lambda: np.all(preview.display_data()==5))
        np.testing.assert_array_equal(preview.finish(True), 5)
        self.assertEqual(calls, ['old','new'])

    def test_cancel_and_restart_ignore_queued_results_and_errors(self):
        preview = self.controller()
        errors = QSignalSpy(preview.preview_failed)
        def fail(data, gray):
            raise ValueError('old failure')
        preview.set_filter_func(fail)
        preview.update_now()
        preview.finish(False)
        preview.start(np.zeros((20,30), dtype=np.float32), True)
        preview.set_filter_func(lambda data, gray: data+7)
        preview.update_now()
        self.wait_for(lambda: np.all(preview.display_data()==7))
        self.assertEqual(len(errors), 0)
        np.testing.assert_array_equal(preview.finish(True), 7)

    def test_combined_stats_preserve_pre_clamp_values_with_one_filter_call(self):
        source = np.random.default_rng(2).uniform(0, 1.2, (36,48,3)).astype(np.float32)
        for name, params in (
            ('wavelet_sharpen', {'fine': 4, 'clamp': True}),
            ('adaptive_deconv', {'amount': 2, 'clamp': True}),
            ('deglow', {}),
            ('curves', {}),
        ):
            with self.subTest(filter=name):
                expected = apply_filter(name, source, False, params)
                stats = output_filter_stats(name, source, False, params, texture_scale=2, chromatic=False)
                with patch('planetary_tools.filters.registry.run_filter_raw', wraps=run_filter_raw) as run:
                    pixels, combined = apply_filter_and_output_stats(name, source, False, params,
                                                                     texture_scale=2, chromatic=False)
                self.assertEqual(run.call_count, 1)
                np.testing.assert_array_equal(pixels, expected)
                self.assertEqual(combined, stats)

    def test_main_window_readouts_toggle_and_apply_do_not_rerun_the_filter(self):
        from planetary_tools.core.document import ImageDocument
        from planetary_tools.ui.main_window import MainWindow
        with tempfile.TemporaryDirectory() as directory, \
             patch('planetary_tools.core.presets.PRESET_DIR', Path(directory)):
            window = MainWindow()
            window._set_document(ImageDocument(np.full((32,48,3), .1, dtype=np.float32)))
            calls, errors = [], []
            def measured(*args, **kwargs):
                calls.append(threading.get_ident())
                return run_filter_raw(*args, **kwargs)
            deadline = time.monotonic()+5
            def accept_when_ready():
                try:
                    if window._preview.output_stats() is None:
                        if time.monotonic() >= deadline:
                            raise AssertionError('preview timed out')
                        QTimer.singleShot(5, accept_when_ready)
                        return
                    panel = window._active_filter_dlg
                    self.assertNotIn('preview off', panel._output_label.text())
                    panel.preview.setChecked(False)
                    panel.preview.setChecked(True)
                    panel._accept()
                except BaseException as exc:
                    errors.append(exc)
                    window._active_filter_dlg._reject()
            try:
                with patch('planetary_tools.filters.registry.run_filter_raw', side_effect=measured):
                    QTimer.singleShot(0, accept_when_ready)
                    window._run_deglow()
                self.assertFalse(errors, errors)
                self.assertEqual(len(calls), 1)
                self.assertNotEqual(calls[0], threading.get_ident())
            finally:
                window._document.modified = False
                window.close()


if __name__ == '__main__':
    unittest.main()
