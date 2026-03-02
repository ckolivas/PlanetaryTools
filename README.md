Planetary Tools plug-ins for GIMP v3

This is a set of tools designed for processing stacked planetary image captures.


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
