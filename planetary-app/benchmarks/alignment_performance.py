"""Benchmark default rotation searches with exact match and rendered parity."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scipy.ndimage import rotate, shift

from image_performance import compare, load_baseline
from planetary_tools.core import field_derotate as align
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='48fc394')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-angle', type=float, default=45.)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    old = load_baseline(args.baseline, 'core/field_derotate')
    reference = load_image(args.image).data
    target = shift(rotate(reference, 1.37, reshape=False, order=3),
                   (.38, -.71, 0) if reference.ndim == 3 else (.38, -.71), order=3)
    ref_small = align._structure(align._search_downsample(align.luma(reference)))
    tgt_small = align._structure(align._search_downsample(align.luma(target)))
    report = {'image': str(args.image), 'shape': list(reference.shape), 'search_shape': list(ref_small.shape),
              'baseline': args.baseline, 'repeats': args.repeats, 'max_angle': args.max_angle}
    report['angle_search'] = compare(lambda: old._search_angle(ref_small, tgt_small, args.max_angle),
                                     lambda: align._search_angle(ref_small, tgt_small, args.max_angle), args.repeats)
    def estimate(module):
        return asdict(module.estimate_rigid(reference, target, max_angle=args.max_angle))
    report['complete_match'] = compare(lambda: estimate(old), lambda: estimate(align), args.repeats)
    old_match = old.estimate_rigid(reference, target, max_angle=args.max_angle)
    new_match = align.estimate_rigid(reference, target, max_angle=args.max_angle)
    import numpy as np
    np.testing.assert_array_equal(old.apply_rigid(target, old_match), align.apply_rigid(target, new_match))
    report['match'] = asdict(new_match)
    report['parity'] = 'angle search, complete match and rendered pixels identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
