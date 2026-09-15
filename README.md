This is a set of tools designed for processing stacked planetary image captures.

Contains Planetary Tools standalone application and plug-ins for GIMP v3

Standalone application requires no installation, simply run the executable.
Download the latest for Ubuntu & Windows here:
https://github.com/ckolivas/PlanetaryTools/releases

Animate can export GIF, animated PNG, WebP, or MP4 video. MP4 export requires
FFmpeg on PATH, built with the `libx264rgb` encoder. Its constant quality (CRF)
control ranges from 0 (lossless, the default) to 51 (lowest quality/smallest files).
Lossless preserves the 8-bit sRGB animation frames exactly; higher precision
source images are converted to 8-bit sRGB, as with the other animation formats.
MP4 retains the chosen frame rate and back-and-forth sequence. Loop playback is
controlled by the video player. Playback requires support for H.264 RGB (4:4:4).


GIMP plugins:

The whole set of directories and their files should go into the user's GIMP v3 plug-ins directory.

On Linux this is usually in ~/.config/GIMP/3.0/plug-ins/

On Windows this is usually in C:\Users\username\AppData\Roaming\GIMP\3.0\plug-ins

This can be configured within GIMP under Settings->Folders->Plug-Ins


A suggested processing workflow for stacked planetary images would be:

Wavelet Sharpen->

Deconvolution (non-adaptive)->

+/-WinJUPOS derotation->

Wavelet Denoise->

Adaptive Deconvolution


The OKLab tools may be helpful to those interested, though not strictly planetary image processing tools.


Wavelet Sharpen

  Filters->Enhance->Wavelet Sharpen

Provides 3 sliders for fine, medium, and coarse detail sharpening.

Adaptive Deconvolution

  Filters->Enhance->Adaptive Deconvolution

Provides a single slider for performing fine deconvolution.
Adaptive option makes sharpening contrast dependent, allowing more sharpening in areas of the image that tolerate it more before becoming noisy.
OKLab option performs sharpening on OKLab luminance to avoid sharpening colour noise, but has gradual saturation loss the more is applied.

Wavelet Denoise

  Filters->Enhance->Wavelet Denoise

Provides 3 sliders equivalent to wavelet sharpen's settings for denoising.

Stretch Contrast OKLab

  Colours->Auto->Stretch Contrast OKLab

Performs a contrast stretch on OKLab Luminance which most accurately preserves perceptual colour balance.

OKLab Luminance

  Colours->Desaturate->OKLab Luminance

Does a simple desaturation to OKLab luminance.

OKLab Decomposose

  Colours->Components->OKLab Decompose

Creates a new image with layers consistuted from OKLab L, a, and b channels. RGB input only.

OKLab Compose

  Colours->Components->OKLab Compose

Creates a new image from 3 selected layers corresponding to OKLab L, a, and b channels.

Con Kolivas 2026
<kernel@kolivas.org>
