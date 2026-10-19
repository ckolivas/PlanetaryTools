"""Graph-derived sensor coefficients, neutral response and stored preset refresh."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.core import presets
from planetary_tools.filters.colour_matrix import SENSOR_MATRICES, apply_colour_matrix
from planetary_tools.filters.registry import FILTERS


class Imx462MatrixTests(unittest.TestCase):
    sensor = 'IMX462'
    old_matrix = [[1.189, -.132, -.020], [-.134, 1.121, -.054], [-.241, -.070, 1.128]]

    def test_preset_matches_integrated_source_graph(self):
        source = Path(__file__).resolve().parents[2] / f'docs/data/{self.sensor.lower()}-sensitivity.csv'
        samples = np.loadtxt(source, delimiter=',', skiprows=1)
        wavelength = samples[:, 0]
        self.assertTrue(np.all(np.diff(wavelength) > 0))
        self.assertEqual((wavelength[0], wavelength[-1]), (400, 680))

        def area(channel, lower, upper):
            self.assertIn(lower, wavelength)
            self.assertIn(upper, wavelength)
            selected = (wavelength >= lower) & (wavelength <= upper)
            x, y = wavelength[selected], samples[selected, channel + 1]
            return np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2)

        expected = np.zeros((3, 3))
        intervals = {(0, 1): (500, 550), (0, 2): (400, 500),
                     (1, 0): (650, 680), (1, 2): (400, 450),
                     (2, 0): (600, 680), (2, 1): (550, 600)}
        for (row, column), (lower, upper) in intervals.items():
            expected[row, column] = round(-area(row, lower, upper) / area(row, 400, 680), 3)
        for row in range(3):
            expected[row, row] = 1 - expected[row].sum()
        np.testing.assert_allclose(SENSOR_MATRICES[self.sensor], expected, atol=1e-12, rtol=0)

    def test_equal_rgb_is_preserved_including_out_of_range_pixels(self):
        levels = np.array([-.1, 0, .01, .4, 1, 1.5], dtype=np.float32)
        image = np.repeat(levels[None, :, None], 3, axis=2)
        result = apply_colour_matrix(image, np.array(SENSOR_MATRICES[self.sensor]))
        np.testing.assert_allclose(result, image, atol=1e-7, rtol=0)

    def test_existing_sensor_preset_refresh_preserves_personal_settings(self):
        old = {'matrix': deepcopy(self.old_matrix),
               'clamp': False}
        with tempfile.TemporaryDirectory() as folder, patch.object(presets, 'PRESET_DIR', Path(folder)):
            presets.save_presets('colour_matrix', {name: deepcopy(old) for name in (self.sensor, 'Last', 'My camera')})
            loaded = presets.ensure_builtin_presets('colour_matrix', FILTERS['colour_matrix'].default_params)
            self.assertEqual(loaded[self.sensor]['matrix'], SENSOR_MATRICES[self.sensor])
            self.assertEqual(loaded['Last'], old)
            self.assertEqual(loaded['My camera'], old)
            self.assertEqual(presets.load_presets('colour_matrix', {}), loaded)


class Imx290MatrixTests(Imx462MatrixTests):
    sensor = 'IMX290'
    old_matrix = [[1.073, -.029, -.044], [-.074, 1.098, -.024], [-.148, -.024, 1.172]]


class Imx183MatrixTests(Imx462MatrixTests):
    sensor = 'IMX183'
    old_matrix = [[1.150, -.102, -.048], [-.029, 1.080, -.051], [-.025, -.021, 1.046]]


class Imx224MatrixTests(Imx462MatrixTests):
    sensor = 'IMX224'
    old_matrix = [[1.192, -.151, -.042], [-.032, 1.110, -.078], [-.060, -.072, 1.132]]


class Imx485MatrixTests(Imx462MatrixTests):
    sensor = 'IMX485'
    old_matrix = [[1.180, -.123, -.018], [-.124, 1.169, -.087], [-.344, -.078, 1.185]]


class Imx571MatrixTests(Imx462MatrixTests):
    sensor = 'IMX571'
    old_matrix = [[1.208, -.147, -.061], [-.005, 1.024, -.018], [-.041, -.092, 1.133]]


class Imx585MatrixTests(Imx462MatrixTests):
    sensor = 'IMX585'
    old_matrix = [[1.025, -.024, .000], [-.130, 1.026, -.021], [-.019, -.076, 1.024]]


if __name__ == '__main__':
    unittest.main()
