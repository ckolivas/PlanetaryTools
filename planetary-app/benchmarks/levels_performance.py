"""Compare Levels measurement and auto-balance with exact Git baseline results."""
import argparse
import json
from pathlib import Path

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.filters import levels
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='a8b5ee1')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    old = load_baseline(args.baseline, 'filters/levels')
    doc = load_image(args.image)
    data = doc.data
    samples = levels._channel_values(data, 'R')
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    operations = {
        'red_input_peak': lambda module: module.channel_input_peak(data, 'R'),
        'channel_percentiles': lambda module: module.auto_input_levels_for_channel(samples),
        'rgb_auto_balance': lambda module: module.auto_balance_levels(data, is_grayscale=doc.is_grayscale),
        'auto_balance_and_apply': lambda module: module.apply_levels(data, module.auto_balance_levels(data, is_grayscale=doc.is_grayscale)),
    }
    for name, operation in operations.items():
        before, after = lambda: operation(old), lambda: operation(levels)
        report[name] = compare(before, after, args.repeats)
        report[name].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
    report['parity'] = 'channel peaks, auto-balance parameters and applied pixels identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
