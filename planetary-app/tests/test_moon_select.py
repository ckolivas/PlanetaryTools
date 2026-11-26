"""Moon Enhance: Enhance shows OKLab L; Select keeps clicked moons first."""
import os
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer

from planetary_tools.core.colour import rgb_to_oklab_L
from planetary_tools.core.document import ImageDocument
from planetary_tools.core.presets import load_presets
from planetary_tools.filters.extract_component import extract_component
from planetary_tools.filters.moon_enhance import locate_selected_moon
from planetary_tools.filters.registry import apply_filter
from planetary_tools.ui.dialogs import MoonEnhanceDialog
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.preview import array_to_display_rgb, luminance_display_rgb
from test_sticky_auto import Isolated

BRIGHT = (60, 50)   # (x, y) of a moon the detector finds first
FAINT = (250, 150)  # a fainter one


def field():
    """Dark noisy sky, a planet disk and two moons."""
    rng = np.random.default_rng(7)
    h, w = 200, 300
    yy, xx = np.mgrid[:h, :w]
    sky = 2e-4 + rng.normal(0.0, 2e-5, (h, w))
    sky += 0.3 * ((yy - 100) ** 2 + (xx - 150) ** 2 <= 30 ** 2)
    for (x, y), peak in ((BRIGHT, 0.02), (FAINT, 0.004)):
        sky += peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2.2 ** 2))
    return np.repeat(np.clip(sky, 0, 1)[..., None], 3, axis=2).astype(np.float32)


def params(**changes):
    return {'brightness': 50.0, 'sensitivity': 8.0, 'radius_scale': 0.5,
            'planet_margin': 0.0, 'max_moons': 1, **changes}


class SelectionFilterTests(Isolated):
    def test_luminance_view_matches_extracted_oklab_l(self):
        data = field()
        rgb = luminance_display_rgb(data)
        extracted = extract_component(data, False, 'oklab_l')
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb, array_to_display_rgb(extracted, False))
        np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])
        np.testing.assert_array_equal(luminance_display_rgb(data[..., 0]), rgb)
        # Brighter than L shown as a screen value: the sky is lifted further.
        self.assertGreater(rgb[185, 150, 0], np.clip(rgb_to_oklab_L(data), 0, 1)[185, 150] * 255 + 20)

    def test_click_snaps_to_the_moon_and_not_to_the_planet(self):
        data = field()
        self.assertEqual(locate_selected_moon(data, False, FAINT[0] + 7, FAINT[1] - 6), FAINT)
        self.assertIsNone(locate_selected_moon(data, False, 150, 100))  # planet
        self.assertIsNone(locate_selected_moon(data, False, 150, 185))  # empty sky

    def test_click_below_ring_glow_finds_the_faint_moon(self):
        # A band of glow along a tilted ring edge is brighter than the moon
        # below it and, measured across x and y, about moon-sized.
        data = field()
        yy, xx = np.mgrid[:200, :300]
        tilt = np.radians(20)
        along = np.cos(tilt) * (xx - 215) + np.sin(tilt) * (yy - 110)
        across = -np.sin(tilt) * (xx - 215) + np.cos(tilt) * (yy - 110)
        glow = 0.0016 * np.exp(-(along / 60.0) ** 2 - (across / 3.5) ** 2)
        glow += 0.0008 * np.exp(-((yy - 125) ** 2 + (xx - 220) ** 2) / (2 * 2.5 ** 2))
        data += glow[..., None].astype(np.float32)
        for click in ((220, 125), (217, 122), (224, 128)):
            self.assertEqual(locate_selected_moon(data, False, *click), (220, 125))
        self.assertIsNone(locate_selected_moon(data, False, 200, 104))  # on the band

    def test_selected_moon_comes_first(self):
        data = field()

        def brightened(result, moon):
            x, y = moon
            return result[y, x, 0] > data[y, x, 0] * 1.01
        plain = apply_filter('moon_enhance', data, False, params())
        self.assertTrue(brightened(plain, BRIGHT))
        self.assertFalse(brightened(plain, FAINT))
        chosen = apply_filter('moon_enhance', data, False, params(selected_moons=[list(FAINT)]))
        self.assertTrue(brightened(chosen, FAINT))
        self.assertFalse(brightened(chosen, BRIGHT))
        # Kept even when it is far below Sensitivity.
        forced = apply_filter('moon_enhance', data, False,
                              params(sensitivity=50.0, selected_moons=[list(FAINT)]))
        self.assertTrue(brightened(forced, FAINT))


class SelectionPanelTests(Isolated):
    def panel(self):
        panel = MoonEnhanceDialog()
        self.addCleanup(panel.close)
        panel.set_input_brightness(field(), False)
        return panel

    def test_select_adds_and_removes_moons(self):
        panel = self.panel()
        markers, picks = [], []
        panel.markers_changed.connect(markers.append)
        panel.region_pick_toggled.connect(picks.append)
        self.assertFalse(panel.region_pick_active())
        panel.set_picked_region(FAINT[0] + 5, FAINT[1], 0, 0)
        self.assertEqual(panel.get_params()['selected_moons'], [])  # Select not pressed
        panel.select.click()
        self.assertEqual(picks, [True])
        self.assertTrue(panel.select_hint.isVisibleTo(panel))
        panel.set_picked_region(FAINT[0] + 5, FAINT[1] + 3, 0, 0)
        panel.set_picked_region(BRIGHT[0] - 30, BRIGHT[1] - 25, 60, 50)  # dragged box
        self.assertEqual(panel.get_params()['selected_moons'], [list(FAINT), list(BRIGHT)])
        self.assertEqual(markers[-1], [(*FAINT, 12.0), (*BRIGHT, 12.0)])
        self.assertTrue(panel.clear_selected.isEnabled())
        self.assertIn('Selected: 2', panel.select_hint.text())
        # Selections fit this image only, and are not a preset edit.
        self.assertNotIn('selected_moons', panel.reusable_params())
        self.assertNotEqual(panel._preset_combo.currentText(), '(None)')
        # Clicking a selected moon removes it; nowhere near a moon says so.
        panel.set_picked_region(FAINT[0] + 3, FAINT[1] - 3, 0, 0)
        self.assertEqual(panel.get_params()['selected_moons'], [list(BRIGHT)])
        panel.set_picked_region(150, 185, 0, 0)
        self.assertIn('No moon found', panel.select_hint.text())
        self.assertEqual(len(panel.get_params()['selected_moons']), 1)
        panel.clear_selected.click()
        self.assertEqual((panel.get_params()['selected_moons'], markers[-1]), ([], []))
        self.assertFalse(panel.clear_selected.isEnabled())

    def test_choosing_a_preset_keeps_the_selection(self):
        panel = self.panel()
        panel.select.click()
        panel.set_picked_region(*FAINT, 0, 0)
        panel.set_params({**panel.reusable_params(), 'brightness': 10.0})
        self.assertEqual(panel.get_params()['selected_moons'], [list(FAINT)])

    def test_enhance_and_show_difference_take_turns(self):
        panel = self.panel()
        views = []
        panel.view_changed.connect(lambda: views.append(panel.luminance_view_shown()))
        panel.enhance.click()
        self.assertEqual(views, [True])
        panel.show_difference.setChecked(True)
        self.assertEqual((views, panel.enhance.isChecked()), ([True, False], False))
        panel.enhance.click()
        self.assertFalse(panel.show_difference.isChecked())
        self.assertTrue(panel.difference_shown() is False and panel.luminance_view_shown())


class SelectionWindowTests(Isolated):
    def test_canvas_picks_while_select_is_pressed_and_enhance_shows_l(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window._set_document(ImageDocument(field()))
        shown, errors, seen = [], [], {}
        canvas = window._canvas
        original_show = canvas.show_rgb_uint8

        def record(rgb):
            shown.append(rgb)
            original_show(rgb)

        def drive():
            panel = window._active_filter_dlg
            try:
                seen['pick_before'] = canvas._pick_enabled
                panel.select.click()
                seen['pick_pressed'] = canvas._pick_enabled
                canvas.region_picked.emit(FAINT[0] + 4, FAINT[1] + 4, 0, 0)
                seen['markers'] = sum(item.isVisible() for item in canvas._markers)
                panel.preview.setChecked(False)
                panel.enhance.click()
                seen['enhanced'] = shown[-1] if shown else None
                panel._accept()
            except BaseException as exc:
                errors.append(exc)
                panel._reject()
        with patch.object(canvas, 'show_rgb_uint8', side_effect=record):
            QTimer.singleShot(0, drive)
            window._run_moon_enhance()
        self.assertFalse(errors, errors)
        self.assertEqual((seen['pick_before'], seen['pick_pressed'], seen['markers']),
                         (False, True, 1))
        np.testing.assert_array_equal(seen['enhanced'], luminance_display_rgb(field()))
        # Closed: no picking or markers; the selection stays with this image.
        self.assertFalse(canvas._pick_enabled)
        self.assertFalse(any(item.isVisible() for item in canvas._markers))
        self.assertNotIn('selected_moons', load_presets('moon_enhance', {})['Last'])
        op = window._undo.stack.applied_operations()[-1]
        self.assertNotIn('selected_moons', op.params)
        window._document.modified = False


if __name__ == '__main__':
    import unittest
    unittest.main()
