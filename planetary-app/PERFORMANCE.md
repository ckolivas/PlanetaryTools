# Performance improvements

## Preview and Apply

A filter now produces preview pixels and pre-clamp brightness/noise readouts
in one background job. Previously, calculating the readouts ran the filter
again on the UI thread. The last completed image remains visible during an
update. Toggling preview does not invalidate the calculation.

Apply reuses a completed result only when its source/settings generation is
still current, including a worker result awaiting Qt signal delivery. Changed
settings invalidate it immediately, before the debounce timer fires. Running
jobs keep immutable input/function snapshots; a pending update runs after the
worker finishes. Cancel and later sessions ignore stale results and failures.

## Auto wavelet sharpening

The Gaussian blur and linear-light conversion of a wavelet scale do not depend
on sharpening amount. Auto search prepares them once per channel/scale and
reuses the resulting detail. Only the latest sharpened layer per channel/scale
is retained, so testing additional amounts no longer grows the image cache
without bound. Unsharp arithmetic and pixel precision are unchanged.

## Measurements

Measured locally on `3moons.png` (704 × 464 RGB), median of three paired runs.
Preview timings cover filter evaluation and output statistics, excluding file
I/O, display conversion and debounce. Noise context was fixed in both paths.
Auto timings cover eight fine-amount trials after decomposition, with medium,
coarse and chunky held at 8, 1 and 0. Actual speedups depend on image and settings.

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| Deglow preview + readouts | 0.324 s | 0.167 s | 1.94× |
| Wavelet Sharpen preview + readouts | 0.609 s | 0.363 s | 1.68× |
| Adaptive Deconvolution preview + readouts | 0.068 s | 0.045 s | 1.51× |
| Eight auto-sharpen trials | 0.783 s | 0.419 s | 1.87× |

All benchmark pixels and statistics compare exactly. Tests also exercise
clamping, RGB/grayscale/luminance sharpening, cache bounds, stale jobs, Apply
during debounce or processing, Cancel/reopen, and the actual main-window
preview/readout/toggle/Apply path.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/preview_performance.py 3moons.png --baseline cc5c12d
QT_QPA_PLATFORM=offscreen PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  -m unittest discover -s planetary-app/tests -p 'test_*performance.py' -v
```

The benchmark loads the baseline auto-trial engine from Git. For preview it
compares the old two-evaluation path with the new combined path using the same
filter implementations, and asserts result parity before reporting timings.
