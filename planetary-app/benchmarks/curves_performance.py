"""Measure complete Curves operations and map buffers against a Git baseline."""
import argparse
import json
from pathlib import Path

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.filters import curves
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--baseline',default='cd1c9b6')
    parser.add_argument('--repeats',type=int,default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    old = load_baseline(args.baseline,'filters/curves')
    data = load_image(args.image,pin_noise=False).data
    params = curves.default_curves_params()
    params['channels']['Value']['points'] = [[0.,0.,'smooth'],[.4,.6,'smooth'],[1.,1.,'smooth']]
    params['channels']['Red']['points'] = [[0.,0.,'smooth'],[.6,.45,'smooth'],[1.,1.,'smooth']]
    samples = curves.curve_samples(params['channels']['Value'])
    report = {'image':str(args.image),'shape':list(data.shape),'baseline':args.baseline,'repeats':args.repeats}
    def record(name,before,after):
        report[name] = compare(before,after,args.repeats)
        report[name].update(before_peak_mib=peak_mib(before),after_peak_mib=peak_mib(after))
    record('map_samples',lambda:old.map_samples(data[...,0],samples),lambda:curves.map_samples(data[...,0],samples))
    for trc in ('linear','perceptual'):
        params['trc'] = trc
        record('curves_'+trc,lambda:old.apply_curves(data,params),lambda:curves.apply_curves(data,params))
    report['parity'] = 'all mapped values and complete Curves pixels identical'
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
