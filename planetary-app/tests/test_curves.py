"""Curves processing, GIMP source parity, editor and preview lifecycle checks."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog

from planetary_tools.core.colour import linear_to_srgb, srgb_to_linear
from planetary_tools.core.document import ImageDocument
from planetary_tools.filters.curves import (
    apply_curves, change_curve_mode, curve_samples, default_curves_params,
    identity_curve, map_samples, normalize_curves_params,
)
from planetary_tools.filters.registry import apply_filter, batch_filters
from planetary_tools.ui.curves_dialog import CurvesDialog, histogram_display_heights


class CurvesTests(unittest.TestCase):
    def test_identity_preserves_hdr_and_nonfinite_samples(self):
        source = np.array([[[-.3, .4, 2], [np.nan, np.inf, -np.inf]]], np.float32)
        np.testing.assert_array_equal(apply_curves(source, {}), source)
        np.testing.assert_array_equal(apply_filter('curves', source, False, {}), source)

    def test_value_maps_components_not_luminance(self):
        params = default_curves_params()
        params['trc'] = 'linear'
        params['channels']['Value']['points'] = [[0, 1, 'smooth'], [1, 0, 'smooth']]
        source = np.array([[[.1, .4, .8]]], np.float32)
        np.testing.assert_allclose(apply_curves(source, params), 1-source, atol=1e-7)

    def test_rgb_precedes_value_and_alpha_is_independent(self):
        params = default_curves_params()
        params['trc'] = 'linear'
        params['channels']['Red']['points'] = [[0, .2, 'smooth'], [1, 1, 'smooth']]
        params['channels']['Value']['points'] = [[0, 0, 'smooth'], [1, .5, 'smooth']]
        source = np.array([[[.25, .4, .8, .3]]], np.float32)
        np.testing.assert_allclose(apply_curves(source, params), [[[.2, .2, .4, .3]]], atol=1e-7)
        params['channels']['Alpha']['points'] = [[0, 1, 'smooth'], [1, 0, 'smooth']]
        self.assertAlmostEqual(float(apply_curves(source, params)[0, 0, 3]), .7, places=6)

    def test_only_edited_channel_is_changed(self):
        params = default_curves_params()
        params['channels']['Red']['points'][1][1] = .5
        source = np.array([[[.25, -.5, 3]]], np.float32)
        out = apply_curves(source, params)
        np.testing.assert_array_equal(out[..., 1:], source[..., 1:])
        self.assertLess(out[0, 0, 0], source[0, 0, 0])

    def test_perceptual_mapping_and_gray_shapes(self):
        params = default_curves_params()
        params['channels']['Value']['points'][1][1] = .5
        source = np.array([[0, .1, .25, .5, 1]], np.float32)
        expected = srgb_to_linear(linear_to_srgb(source)*.5)
        np.testing.assert_allclose(apply_curves(source, params), expected, atol=1e-7)
        np.testing.assert_allclose(apply_curves(source[..., None], params)[..., 0], expected, atol=1e-7)
        params['trc'] = 'linear'
        np.testing.assert_allclose(apply_curves(source, params), source*.5, atol=1e-7)

    def test_endpoints_clip_and_mapping_interpolates_samples(self):
        curve = {'mode': 'smooth', 'points': [[.2, .3, 'corner'], [.8, .7, 'corner']]}
        samples = curve_samples(curve)
        self.assertEqual(samples[0], .3)
        self.assertEqual(samples[-1], .7)
        values = np.array([-1, np.nan, np.inf, .5, 2])
        np.testing.assert_allclose(map_samples(values, samples), [.3, .3, .7, .5, .7], atol=.003)
        self.assertAlmostEqual(float(map_samples(np.array(100.25/255), samples)),
                               .75*samples[100]+.25*samples[101])

    def test_gimp_nonlinear_smooth_and_corner_reference_values(self):
        curve = {'mode': 'smooth', 'points': [
            [0, 0, 'smooth'], [.25, .1, 'smooth'], [.65, .85, 'corner'], [1, 1, 'smooth'],
        ]}
        indices = [0, 16, 32, 64, 96, 127, 128, 160, 192, 224, 255]
        # Fixed GIMP sample values keep a non-linear regression available even
        # on machines without a local GIMP checkout or C compiler.
        expected = [0, 0, .007593038296559569, .1, .2941010041966688,
                    .5261951709749644, .5341850187565637, .7992312389422098,
                    .8936974789915966, .9474789915966386, 1]
        np.testing.assert_allclose(curve_samples(curve)[indices], expected, atol=2e-15, rtol=0)

    def test_freehand_conversion_and_json_roundtrip(self):
        curve = {'mode': 'smooth', 'points': [[0, 0, 'smooth'], [.5, .8, 'corner'], [1, 1, 'smooth']]}
        free = change_curve_mode(curve, 'freehand')
        np.testing.assert_array_equal(curve_samples(free), curve_samples(curve))
        smooth = change_curve_mode(free, 'smooth')
        self.assertEqual(len(smooth['points']), 9)
        params = default_curves_params()
        params['channels']['Value'] = free
        self.assertEqual(normalize_curves_params(json.loads(json.dumps(params))), params)
        self.assertIn('curves', [f.id for f in batch_filters()])

    def test_invalid_curves_are_rejected(self):
        for curve in (
            {'mode': 'freehand', 'samples': [0, 1]},
            {'mode': 'smooth', 'points': [[0, 0], [0, 1]]},
            {'mode': 'smooth', 'points': [[0, float('nan')], [1, 1]]},
            {'mode': 'smooth', 'points': [[0, 0, 'invalid'], [1, 1]]},
        ):
            with self.subTest(curve=curve), self.assertRaises(ValueError):
                apply_curves(np.zeros((2, 2)), {'channels': {'Value': curve}})


class HistogramScaleTests(unittest.TestCase):
    def test_linear_sky_spikes_do_not_hide_stretched_tones(self):
        counts = np.zeros(256, dtype=np.int64)
        counts[0:3] = [1_000_000, 100_000, 25_000]
        counts[16:255] = np.tile([50, 100, 150], 80)[:239]
        original = counts.copy()
        heights, clipped = histogram_display_heights(counts, False)
        self.assertTrue(clipped)
        self.assertEqual(heights[0], 1)
        self.assertGreaterEqual(heights[16], 1/3)
        self.assertEqual(heights[17] / heights[16], 2)
        self.assertEqual(heights[18] / heights[16], 3)
        self.assertEqual(heights[8], 0)
        np.testing.assert_array_equal(counts, original)

    def test_logarithmic_counts_are_not_changed(self):
        counts = np.arange(256, dtype=float)
        counts[0] = 1_000_000
        heights, clipped = histogram_display_heights(counts, True)
        np.testing.assert_allclose(heights, np.log1p(counts)/np.log1p(counts.max()))
        self.assertFalse(clipped)

    def test_empty_sparse_and_balanced_histograms_keep_their_scale(self):
        for counts in (np.zeros(256), np.ones(256), np.arange(256),
                       np.r_[1000, np.zeros(254), 500]):
            heights, clipped = histogram_display_heights(counts, False)
            self.assertFalse(clipped)
            np.testing.assert_allclose(heights, counts/max(float(counts.max()), 1))


class GimpParityTests(unittest.TestCase):
    def test_smooth_samples_match_local_gimp_c_implementation(self):
        source_path = Path.home()/'Code/gimp/app/core/gimpcurve.c'
        compiler = shutil.which('cc')
        if not compiler or not source_path.exists():
            self.skipTest('Optional parity check needs cc and ~/Code/gimp source')
        source = source_path.read_text()
        # Compile the actual GIMP calculation functions, with only the GObject
        # notification machinery stubbed out. No independent spline substitute.
        source = source[source.rindex('static void\ngimp_curve_calculate ('):]
        header = '''
#include <math.h>
typedef double gdouble;
typedef int gint;
#define EPSILON 1e-6
#define ROUND(x) ((int)((x)+0.5))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define CLAMP(x,a,b) MIN(MAX(x,a),b)
#define GIMP_CURVE_SMOOTH 0
#define GIMP_CURVE_FREE 1
#define GIMP_CURVE_POINT_CORNER 1
#define GIMP_DATA(c) (c)
#define gimp_data_is_frozen(c) 0
#define g_object_notify_by_pspec(...) ((void)0)
typedef struct { double x,y; int type; } GimpCurvePoint;
typedef struct { int curve_type,n_points,n_samples; GimpCurvePoint *points; double *samples; } GimpCurve;
static void gimp_curve_plot(GimpCurve*,int,int,int,int);
'''
        wrapper = '''
void reference_samples(double *xy, int *types, int count, double *output) {
  GimpCurvePoint points[count];
  for(int i=0;i<count;i++) points[i]=(GimpCurvePoint){xy[2*i],xy[2*i+1],types[i]};
  for(int i=0;i<256;i++) output[i]=i/255.0;
  GimpCurve curve={0,count,256,points,output};
  gimp_curve_calculate(&curve);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            c_path = Path(directory)/'gimp_curves.c'
            so_path = Path(directory)/'gimp_curves.so'
            c_path.write_text(header+source+wrapper)
            subprocess.run([compiler, '-shared', '-fPIC', '-O2', str(c_path), '-lm', '-o', str(so_path)],
                           check=True, capture_output=True)
            function = ctypes.CDLL(str(so_path)).reference_samples
            function.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                 np.ctypeslib.ndpointer(dtype=np.int32), ctypes.c_int,
                                 np.ctypeslib.ndpointer(dtype=np.float64)]
            rng = np.random.default_rng(23)
            curves = [identity_curve(), {'mode': 'smooth', 'points':
                      [[.13, .2, 'corner'], [.34, .95, 'smooth'], [.65, .1, 'corner'], [.92, .85, 'smooth']]}]
            for count in (3, 4, 9, 17):
                for _ in range(8):
                    curves.append({'mode': 'smooth', 'points': [
                        [float(x), float(rng.random()), str(rng.choice(['smooth', 'corner']))]
                        for x in sorted(rng.random(count))]})
            for curve in curves:
                xy = np.array([p[:2] for p in curve['points']], np.float64)
                kinds = np.array([p[2] == 'corner' for p in curve['points']], np.int32)
                expected = np.empty(256)
                function(xy, kinds, len(xy), expected)
                np.testing.assert_allclose(curve_samples(curve), expected, atol=2e-15, rtol=0)


class CurvesUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.preset_patch = patch('planetary_tools.core.presets.PRESET_DIR', Path(directory.name))
        self.preset_patch.start()
        self.addCleanup(self.preset_patch.stop)

    def test_graph_edits_numeric_controls_channels_and_freehand(self):
        dialog = CurvesDialog()
        dialog.show()
        self.app.processEvents()
        self.addCleanup(dialog.close)
        editor = dialog.editor
        QTest.mouseClick(editor, Qt.MouseButton.LeftButton, pos=editor._position(.4, .65).toPoint())
        self.assertEqual(len(editor.curve['points']), 3)
        dialog.output.setValue(180)
        self.assertAlmostEqual(editor.curve['points'][editor.selected][1], 180/255)
        dialog.point_type.setCurrentText('Corner')
        value = dialog.get_params()['channels']['Value']
        self.assertEqual(value['points'][1][2], 'corner')
        dialog.channel.setCurrentText('Red')
        editor.add_point(.3, .2)
        dialog.channel.setCurrentText('Value')
        self.assertEqual(editor.curve, value)
        dialog.mode.setCurrentText('Freehand')
        QTest.mousePress(editor, Qt.MouseButton.LeftButton, pos=editor._position(.2,.8).toPoint())
        QTest.mouseMove(editor, editor._position(.8,.2).toPoint())
        QTest.mouseRelease(editor, Qt.MouseButton.LeftButton, pos=editor._position(.8,.2).toPoint())
        self.assertAlmostEqual(editor.curve['samples'][128], .5, delta=.015)
        dialog.mode.setCurrentText('Smooth')
        self.assertEqual(len(editor.curve['points']), 9)
        editor.selected=4
        QTest.keyClick(editor, Qt.Key.Key_Delete)
        self.assertEqual(len(editor.curve['points']), 8)
        dialog._reset_channel()
        self.assertEqual(editor.curve, identity_curve())
        self.assertNotEqual(dialog.get_params()['channels']['Red'], identity_curve())
        dialog._reset_all()
        self.assertEqual(dialog.get_params(), default_curves_params())

    def test_linear_histogram_toggle_renders_tones_on_black_background(self):
        from PyQt6.QtGui import QColor
        source = np.zeros((256, 256), dtype=np.float32)
        source[100:132, :] = np.linspace(0, 1, 256)
        dialog = CurvesDialog()
        self.addCleanup(dialog.close)
        dialog.set_input_brightness(source, True)
        dialog.trc.setCurrentIndex(1)
        before = dialog.get_params()
        counts = dialog.editor.histogram.copy()
        dialog.log_hist.setChecked(False)
        dialog.show()
        self.app.processEvents()
        self.assertFalse(dialog.editor.logarithmic)
        self.assertEqual(dialog.get_params(), before)
        np.testing.assert_array_equal(dialog.editor.histogram, counts)
        # Away from the diagonal and grid, histogram fill must be visible well
        # above the baseline despite the large black-background population.
        image = dialog.editor.grab().toImage()
        pos = dialog.editor._position(.3, .6).toPoint()
        self.assertEqual(image.pixelColor(pos), QColor('#3d424d'))

    def test_params_are_snapshot_and_presets_roundtrip(self):
        dialog = CurvesDialog()
        self.addCleanup(dialog.close)
        source = np.full((4, 4), .25, np.float32)
        dialog.editor.add_point(.5,.8)
        function = dialog.build_filter_func()
        expected = function(source, True)
        dialog._reset_all()
        np.testing.assert_array_equal(function(source, True), expected)
        dialog.editor.add_point(.5,.7)
        dialog.save_last_preset()
        again = CurvesDialog()
        self.addCleanup(again.close)
        self.assertEqual(again.get_params(), dialog.get_params())
        again.set_input_brightness(source, True)
        self.assertFalse(again.channel.model().item(1).isEnabled())
        self.assertGreater(again.editor.histogram.sum(), 0)

    def test_apply_cancel_undo_redo_and_image_picker(self):
        from planetary_tools.ui.main_window import MainWindow
        window = MainWindow()
        window.show()
        def close_window():
            if window._document is not None:
                window._document.modified = False
            window.close()
        self.addCleanup(close_window)
        source = np.full((32, 48, 3), .25, np.float32)
        window._set_document(ImageDocument(source.copy()))
        errors = []
        def edit(accept, pick=False):
            try:
                dialog = window._active_filter_dlg
                if pick:
                    dialog.pick.setChecked(True)
                    pos = window._canvas.mapFromScene(QPointF(20.5, 15.5))
                    QTest.mouseClick(window._canvas.viewport(), Qt.MouseButton.LeftButton, pos=pos)
                    self.assertEqual(len(dialog.editor.curve['points']), 3)
                    self.assertAlmostEqual(dialog.editor.curve['points'][1][0],
                                           float(linear_to_srgb(np.array(.25))), places=5)
                    dialog.output.setValue(200)
                else:
                    dialog.editor.add_point(.5, .8)
                # Apply still evaluates the final settings with preview disabled.
                dialog.preview.setChecked(False)
            except Exception as exc:
                errors.append(exc)
            finally:
                (window._active_filter_dlg._accept if accept else window._active_filter_dlg._reject)()
        QTimer.singleShot(0, lambda: edit(False))
        window._run_curves()
        np.testing.assert_array_equal(window._document.data, source)
        self.assertFalse(window._undo.stack.can_undo())
        QTimer.singleShot(0, lambda: edit(True, True))
        window._run_curves()
        if errors:
            raise errors[0]
        applied = window._document.data.copy()
        self.assertGreater(float(applied[0, 0, 0]), .25)
        window._undo_action()
        np.testing.assert_array_equal(window._document.data, source)
        window._redo_action()
        np.testing.assert_array_equal(window._document.data, applied)

    def test_batch_editor_uses_curves_widget(self):
        from planetary_tools.ui.dialogs import edit_filter_params
        def accept():
            for window in self.app.topLevelWidgets():
                if isinstance(window, QDialog) and window.isVisible():
                    editor = window.findChild(CurvesDialog)
                    if editor:
                        editor.editor.add_point(.5, .7)
                        editor._accept()
        QTimer.singleShot(0, accept)
        result = edit_filter_params('curves', default_curves_params(), False)
        self.assertIsNotNone(result)
        params, name = result
        self.assertEqual(len(params['channels']['Value']['points']), 3)
        self.assertIsNone(name)


if __name__ == '__main__':
    unittest.main()
