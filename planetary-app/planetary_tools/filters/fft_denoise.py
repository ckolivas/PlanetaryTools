"""Smooth radial FFT denoising and data-driven initial curve selection.

Frequencies are percentages of axial Nyquist (100 = 0.5 cycles/pixel).
The diagonal corners extend to sqrt(2)*100. No resampling is used to filter.
"""
from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage

MAX_FREQUENCY = 100 * np.sqrt(2)
MIN_START = -141.4
DEFAULT_FFT_DENOISE_PARAMS = dict(start=60.0, feather=40.0, curve=-2.0, auto=False)


def transfer(frequency, start, feather, curve):
    """C2-continuous low pass; negative curve retains more transition detail."""
    t = np.clip((np.asarray(frequency) - start) / max(float(feather), .5), 0, 1)
    # Bias the transition without losing flat derivatives at either endpoint.
    bias = 2.0 ** float(np.clip(curve, -5, 5))
    t = bias * t / (1 + (bias - 1) * t)
    return 1 - t*t*t*(10 + t*(-15 + 6*t))


def _planes(data):
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return arr[..., :3]
    raise ValueError('FFT Denoise requires a grayscale, RGB or RGBA image.')


def fft_denoise(data, is_grayscale=False, *, start=60., feather=40., curve=-2., auto=False):
    arr = np.asarray(data, dtype=np.float32)
    planes = _planes(arr)
    if not planes.size or not np.isfinite(planes).all():
        raise ValueError('FFT Denoise requires a nonempty, finite image.')
    if auto:
        settings = auto_parameters(analyse_spectrum(arr))
        start, feather, curve = (settings[k] for k in ('start', 'feather', 'curve'))
    if not np.isfinite([start, feather, curve]).all():
        raise ValueError('FFT Denoise parameters must be finite.')
    start = float(np.clip(start, MIN_START, MAX_FREQUENCY))
    if start >= 141.4:
        return arr.copy()
    h, w = planes.shape[:2]
    # Reflect rather than wrap opposite limbs. A smooth transition keeps the
    # impulse response short; extra padding protects narrow feather settings.
    pad = int(np.clip(np.ceil(800 / max(feather, .5)), 32, 256))
    shape = (fft.next_fast_len(h + 2*pad), fft.next_fast_len(w + 2*pad))
    radius = np.hypot(fft.fftfreq(shape[0])[:, None], fft.rfftfreq(shape[1])[None, :])*200
    gain = transfer(radius, start, feather, curve).astype(np.float32)
    result = arr.copy()
    dest = result[..., None] if arr.ndim == 2 else result
    for c in range(planes.shape[2]):
        plane = np.pad(planes[..., c], ((pad, shape[0]-h-pad), (pad, shape[1]-w-pad)), mode='reflect')
        spectrum = fft.rfft2(plane)
        spectrum *= gain
        dest[..., c] = fft.irfft2(spectrum, s=shape)[pad:pad+h, pad:pad+w]
    return result


@dataclass(frozen=True)
class Spectrum:
    frequency: np.ndarray
    power: np.ndarray
    noise_power: float
    shape: tuple[int, int]


def analyse_spectrum(data):
    """Native-pixel spectrum of a subject-centred crop, bounded to 1024².

    Averaging channel power avoids hiding colour noise in luminance. The Hann
    window is for measurement only, never multiplied into the filtered image.
    """
    planes = _planes(data)
    if not planes.size or not np.isfinite(planes).all():
        raise ValueError('Cannot analyse an empty or non-finite image.')
    h, w = planes.shape[:2]
    if max(h, w) > 1024:
        sample = np.mean(planes[::max(1, h//128), ::max(1, w//128)], axis=2)
        weight = np.maximum(sample - np.percentile(sample, 75), 0)
        if weight.sum() > 0:
            cy, cx = ndimage.center_of_mass(weight)
            cy *= max(1, h//128)
            cx *= max(1, w//128)
        else:
            cy, cx = h/2, w/2
        y, x = int(np.clip(cy-512, 0, max(0, h-1024))), int(np.clip(cx-512, 0, max(0, w-1024)))
        planes = planes[y:y+1024, x:x+1024]
    h, w = planes.shape[:2]
    frequency = np.linspace(0, MAX_FREQUENCY, 257)
    if min(h, w) < 8:
        return Spectrum(frequency, np.zeros(257), 0., (h, w))
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    radius = np.hypot(fft.fftfreq(h)[:, None], fft.rfftfreq(w)[None, :])*200
    bins = np.minimum(np.rint(radius / MAX_FREQUENCY * 256).astype(int), 256)
    counts = np.bincount(bins.ravel(), minlength=257)
    power = np.zeros(257)
    noise = []
    for c in range(planes.shape[2]):
        plane = planes[..., c]
        transformed = fft.rfft2((plane-plane.mean())*window)
        energy = np.abs(transformed)**2 / np.sum(window**2)
        power += np.bincount(bins.ravel(), weights=energy.ravel(), minlength=257)
        # Robust second differences reject ramps and isolated bright edges.
        # Weight the noise estimate like the spectrum, so a black sky does not
        # drown out noise on the planet. Tile estimates also tolerate shot noise.
        variance, weights = [], []
        for y in range(0, h-7, 32):
            for x in range(0, w-7, 32):
                tile = plane[y:y+32, x:x+32]
                d = np.concatenate((np.diff(tile, n=2, axis=0).ravel(), np.diff(tile, n=2, axis=1).ravel()))
                sigma = np.median(np.abs(d-np.median(d))) / (.67448975*np.sqrt(6))
                variance.append(sigma*sigma)
                weights.append(np.sum(window[y:y+32, x:x+32]**2))
        noise.append(np.average(variance, weights=weights))
    valid = counts > 0
    power[valid] /= counts[valid]*planes.shape[2]
    power[~valid] = np.interp(frequency[~valid], frequency[valid], power[valid])
    return Spectrum(frequency, power, float(np.mean(noise)), (h, w))


def auto_parameters(spectrum):
    """Fit a smooth low-pass to an estimated Wiener response.

    A sustained spectral upturn after a valley is treated as a possible
    sharpening/drizzle noise band. The target must decrease with frequency,
    so such peaks cannot reopen the passband. This is a starting estimate,
    not a claim of recovering the unknown noise-free image.
    """
    f, p = spectrum.frequency, spectrum.power
    if spectrum.noise_power <= max(float(p.max())*1e-12, 1e-20):
        return dict(start=float(MAX_FREQUENCY), feather=1., curve=0.)
    smooth = np.exp(ndimage.gaussian_filter1d(np.log(np.maximum(p, 1e-30)), 2))
    floor = spectrum.noise_power
    target = np.clip(1 - floor/np.maximum(smooth, 1e-30), 0, 1)
    # A white-noise estimate undercounts correlated (sharpened/drizzled) noise.
    # Find a broad valley followed by a real rise, not single-bin fluctuations.
    for i in np.flatnonzero((f > 15) & (f < 85)):
        if i < 6 or i+18 >= len(f):
            continue
        if smooth[i] == np.min(smooth[i-5:i+6]) and np.max(smooth[i+6:i+19]) > 1.35*smooth[i]:
            peak = i + 6 + int(np.argmax(smooth[i+6:i+41]))
            start = f[i]*.85
            target *= transfer(f, start, max(12., f[peak]-start), 0)
            break
    target[f < 3] = 1
    target = np.minimum.accumulate(target)
    # Fit all three controls jointly. Penalise removal of high-SNR structure
    # more strongly than disagreement in the noisy tail.
    weights = np.where(target > .8, 4., 1.)
    best = (float('inf'), 60., 40., -2.)
    for start in np.arange(4., 101., 2.):
        for feather in (12., 20., 30., 40., 55., 75., 100.):
            for curve in (-3., -2., -1., 0., 1., 2., 3.):
                gain = transfer(f, start, feather, curve)
                error = float(np.mean(weights*(gain-target)**2))
                if error < best[0]:
                    best = error, start, feather, curve
    return dict(zip(('start', 'feather', 'curve'), map(float, best[1:])))
