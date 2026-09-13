"""Blocked curve lookup retains interpolation bits, ordering and ownership."""
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.filters import curves


def original_map(values, samples):
    return np.interp(np.nan_to_num(values,nan=0.,neginf=0.,posinf=1.),
                     np.linspace(0,1,len(samples)),samples)


class CurvesPerformanceTests(unittest.TestCase):
    def test_mapping_bits_at_knots_neighbors_and_nonfinite_inputs(self):
        knots = np.linspace(0,1,256)
        values = np.r_[knots,np.nextafter(knots,-np.inf),np.nextafter(knots,np.inf),
                       -np.inf,np.inf,np.nan,-0.,-.01,1.01]
        values = np.resize(values,(37,113))
        samples = np.random.default_rng(9).random(256)
        for source in (values, values.astype(np.float32), values[::-1,::2], np.asfortranarray(values)):
            source.setflags(write=False)
            before = source.copy()
            expected = original_map(source,samples)
            with patch.object(curves,'_MAP_DIRECT_LIMIT',97), patch.object(curves,'_MAP_BLOCK_SIZE',97):
                with patch.object(curves.np,'interp',wraps=np.interp) as interpolate:
                    actual = curves.map_samples(source,samples)
                self.assertGreater(interpolate.call_count,1)
                self.assertTrue(all(call.args[0].size <= 97 for call in interpolate.call_args_list))
            np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
            np.testing.assert_array_equal(source,before)

    def test_small_and_nonstandard_inputs_keep_direct_mapping(self):
        samples = np.linspace(0,1,256)**2
        for source in (0.5, [], [0.,.3,1.], np.ones((464,704),dtype=np.float32),
                       np.arange(100,dtype=np.int16), np.arange(100,dtype=np.float16)):
            for table in (samples, samples.astype(np.complex128)*(1+1j)):
                expected = original_map(source,table)
                limit = 97 if (np.iscomplexobj(table) or np.asarray(source).dtype.kind not in 'f') else curves._MAP_DIRECT_LIMIT
                with patch.object(curves,'_MAP_DIRECT_LIMIT',limit), patch.object(curves.np,'interp',wraps=np.interp) as interpolate:
                    actual = curves.map_samples(source,table)
                self.assertEqual(interpolate.call_count,1)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(type(actual),type(expected))

    def test_full_curves_pixels_match_both_trcs_and_channel_orders(self):
        source = np.random.default_rng(7).uniform(-.1,1.1,(71,97,4)).astype(np.float32)
        params = curves.default_curves_params()
        params['channels']['Value']['points'] = [[0.,.01,'smooth'],[.4,.6,'corner'],[1.,.95,'smooth']]
        params['channels']['Red']['points'] = [[0.,0.,'smooth'],[.6,.45,'smooth'],[1.,1.,'smooth']]
        params['channels']['Alpha'] = {'mode':'freehand','samples':(np.linspace(0,1,256)**2).tolist()}
        for data in (source, source[::-1,::2], np.asfortranarray(source), source[...,:3], source[...,:1], source[...,0]):
            data.setflags(write=False)
            before = data.copy()
            for trc in ('linear','perceptual'):
                params['trc'] = trc
                with patch.object(curves,'map_samples',original_map):
                    expected = curves.apply_curves(data,params)
                with patch.object(curves,'_MAP_DIRECT_LIMIT',97), patch.object(curves,'_MAP_BLOCK_SIZE',97):
                    actual = curves.apply_curves(data,params)
                np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
                np.testing.assert_array_equal(data,before)


if __name__ == '__main__':
    unittest.main()
