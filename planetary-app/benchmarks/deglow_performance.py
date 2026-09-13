"""Compare complete Deglow calls against Git, including peak allocations."""
import argparse
import json
from pathlib import Path

import numpy as np

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.filters import deglow
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='ab55be9')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    old = load_baseline(args.baseline, 'filters/deglow')
    data = load_image(args.image).data
    plane = data if data.ndim == 2 else data[...,0]
    gray_rgb = np.stack([plane, plane, plane], axis=-1)
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    for name, source, gray in (('mono_plane',plane,True), ('rgb_mono_context',data,True),
                               ('loaded_grayscale_rgb',gray_rgb,False), ('colour',data,False)):
        before = lambda: old.deglow(source, gray)
        after = lambda: deglow.deglow(source, gray)
        report[name] = compare(before, after, args.repeats)
        report[name].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
    report['parity'] = 'all Deglow output pixels identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
