"""Selected Levels/Curves channel ranges accompany the overall readouts."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.brightness import measure_brightness, measure_channel_brightness
from planetary_tools.filters.curves import default_curves_params
from planetary_tools.filters.registry import apply_filter_and_output_stats, run_filter_raw
from planetary_tools.ui.curves_dialog import CurvesDialog
from planetary_tools.ui.dialogs import LevelsDialog


def source():
    return np.array([[[.1, .2, .3], [.4, .5, .6]]], dtype=np.float32)


class ChannelStatsTests(unittest.TestCase):
    def test_individual_channels_include_hdr_negative_and_alpha_levels(self):
        image = np.array([[[-.1, .2, .3, .4], [.5, 1.2, .7, .8]]], dtype=np.float32)
        stats = measure_channel_brightness(image, False)
        for index, channel in enumerate(('Red', 'Green', 'Blue', 'Alpha')):
            self.assertEqual(stats[channel], measure_brightness(image[..., index], True))
        self.assertTrue(stats['Red'].would_clip)
        self.assertTrue(stats['Green'].would_clip)
        self.assertFalse(stats['Blue'].would_clip)
        for gray in (image[..., 0], image[..., :1], image[..., :3]):
            self.assertEqual(measure_channel_brightness(gray, True), {})

    def test_channel_stats_share_the_pre_clip_result_without_reapplying_filter(self):
        image = source()
        raw = image * 3 - .5
        for filter_id in ('levels', 'curves'):
            with self.subTest(filter=filter_id), patch(
                'planetary_tools.filters.registry.run_filter_raw', return_value=raw,
            ) as run:
                _, stats = apply_filter_and_output_stats(filter_id, image, False, {'clamp': True})
                self.assertEqual(run.call_count, 1)
                self.assertEqual(stats.brightness, measure_brightness(raw, False))
                self.assertEqual(stats.channel_brightness, measure_channel_brightness(raw, False))


class ChannelReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        presets = patch('planetary_tools.core.presets.PRESET_DIR', Path(temporary.name))
        presets.start()
        self.addCleanup(presets.stop)

    def panel(self, cls, image=None):
        panel = cls()
        self.addCleanup(panel.close)
        panel.set_input_brightness(source() if image is None else image, False)
        return panel

    def choose_channel(self, panel, index):
        combo = panel._channel_combo if isinstance(panel, LevelsDialog) else panel.channel
        combo.setCurrentIndex(index)

    def update(self, panel, image):
        result = panel.build_preview_func()(image, False)
        panel.update_output_brightness(result.stats.brightness,
                                       channel_info=result.stats.channel_brightness)
        return result.stats

    def test_selection_shows_both_ranges_without_replacing_overall_values(self):
        for cls in (LevelsDialog, CurvesDialog):
            with self.subTest(dialog=cls.__name__):
                panel = self.panel(cls)
                stats = self.update(panel, source())
                overall = (panel._input_label.text(), panel._output_label.text())
                self.assertTrue(panel._channel_input_label.isHidden())
                for index, channel in enumerate(('Red', 'Green', 'Blue'), 1):
                    self.choose_channel(panel, index)
                    self.assertFalse(panel._channel_input_label.isHidden())
                    self.assertFalse(panel._channel_output_label.isHidden())
                    self.assertEqual(panel._channel_input_label.text(),
                        measure_brightness(source()[..., index-1], True).format_line(f'{channel} input — '))
                    self.assertEqual(panel._channel_output_label.text(),
                        stats.channel_brightness[channel].format_line(f'{channel} output — '))
                    self.assertEqual((panel._input_label.text(), panel._output_label.text()), overall)
                self.choose_channel(panel, 0)
                self.assertTrue(panel._channel_input_label.isHidden())
                self.assertTrue(panel._channel_output_label.isHidden())

    def test_edited_channel_outputs_preview_toggle_and_new_source(self):
        for cls in (LevelsDialog, CurvesDialog):
            with self.subTest(dialog=cls.__name__):
                panel = self.panel(cls)
                self.choose_channel(panel, 1)
                if cls is LevelsDialog:
                    panel._out_max.setValue(50)
                else:
                    params = default_curves_params()
                    params['trc'] = 'linear'
                    params['channels']['Red']['points'][-1][1] = .5
                    panel.set_params(params)
                stats = self.update(panel, source())
                raw = run_filter_raw(panel.filter_id, source(), False, panel.get_params())
                self.assertEqual(stats.channel_brightness['Red'], measure_brightness(raw[..., 0], True))
                self.assertLess(stats.channel_brightness['Red'].max_pct, 40)
                self.assertEqual(panel._channel_output_label.text(),
                                 stats.channel_brightness['Red'].format_line('Red output — '))
                panel.update_output_brightness(None)
                self.assertIn('preview off', panel._output_label.text())
                self.assertIn('preview off', panel._channel_output_label.text())
                self.choose_channel(panel, 3)
                self.assertEqual(panel._channel_output_label.text(), 'Blue output — (preview off)')
                self.update(panel, source())
                self.assertNotIn('preview off', panel._channel_output_label.text())
                panel.set_input_brightness(source()[..., 0], True)
                self.assertTrue(panel._channel_input_label.isHidden())
                self.assertTrue(panel._channel_output_label.isHidden())

    def test_alpha_and_batch_without_image(self):
        image = np.concatenate((source(), np.array([[[.2], [.8]]], dtype=np.float32)), axis=-1)
        panel = self.panel(CurvesDialog, image)
        self.choose_channel(panel, 4)
        stats = self.update(panel, image)
        self.assertEqual(panel._channel_output_label.text(),
                         stats.channel_brightness['Alpha'].format_line('Alpha output — '))
        batch = CurvesDialog()
        self.addCleanup(batch.close)
        self.choose_channel(batch, 1)
        self.assertTrue(batch._channel_input_label.isHidden())
        self.assertTrue(batch._channel_output_label.isHidden())

    def test_main_window_delivers_channel_stats_to_both_panels(self):
        from planetary_tools.core.document import ImageDocument
        from planetary_tools.ui.main_window import MainWindow
        window = MainWindow()
        self.addCleanup(window.close)
        window._set_document(ImageDocument(source(), is_grayscale=False))
        window._preview.start(source(), False)
        self.addCleanup(lambda: window._preview.finish(False))
        for cls in (LevelsDialog, CurvesDialog):
            panel = self.panel(cls)
            self.choose_channel(panel, 2)
            result = panel.build_preview_func()(source(), False)
            window._active_filter_dlg = panel
            with patch.object(window._preview, 'output_stats', return_value=result.stats):
                window._update_dialog_brightness()
            self.assertEqual(panel._channel_output_label.text(),
                             result.stats.channel_brightness['Green'].format_line('Green output — '))
        window._active_filter_dlg = None


if __name__ == '__main__':
    unittest.main()
