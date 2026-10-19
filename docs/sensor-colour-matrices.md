# Sensor colour-subtraction calculations

These notes record recalculations of built-in sensor presets from graphs supplied
by the author. Only explicitly recalculated sensors are covered; the other stored
presets have not yet been revalidated against their source graphs.

## Method

Use an ideal rectangular 400–680 nm bandpass. Integrate each channel over that
range to obtain its total. The coloured filter-transmission curves on a graph are
not multiplied into the sensor response for this calculation.

Define the subtraction areas as follows, truncating every interval to the bandpass:

| Area | Channel | Wavelength interval (nm) |
| --- | --- | --- |
| R_G | Red | 500–550 |
| R_B | Red | 400–500 |
| G_R | Green | 650–680 |
| G_B | Green | 400–450 |
| B_G | Blue | 550–600 |
| B_R | Blue | 600–680 |

Each off-diagonal coefficient is minus that area divided by the total area of its
**output** channel. Each diagonal is one minus the sum of its two negative
coefficients. Rows are output R, G, B; columns are input R, G, B. PT applies the
matrix to linear RGB. Equal-RGB input remains unchanged because each row sums to
one. This is the author's prescribed subtraction heuristic, not a fitted
colourimetric calibration or an exact inversion of sensor spectral mixing.

## IMX462 — recalculated 19 October 2026

Source: the author-supplied `462colour.png`, titled **Estimated QHY5III462C Extended
IR Response**, 1062 × 593 pixels. No internet references were used. Source SHA-256:
`04f38f13cc939392ce9305c8c5b782c7644709bf095fcc894225267344079642`.

The solid red, green and blue traces were digitised using their colours. Orange,
purple and black filter curves, and the blue filter annotation arrow, were
excluded. Horizontal calibration uses x = 88 at 400 nm and x = 1017 at 1000 nm;
vertical calibration uses y = 495 at zero and approximately y = 65 at 100.
Coordinates are zero-based image pixels.

For each horizontal pixel column, take the median vertical position of matching
trace pixels. Linearly interpolate short gaps where another curve covers the
trace (at most four consecutive columns, about 2.6 nm). Add exact wavelength
boundaries by linear interpolation and integrate with the trapezoidal rule.
The retained [digitised samples](data/imx462-sensitivity.csv) contain the native
horizontal samples plus all integration boundaries within 400–680 nm.

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 9090.43 | R_G = 265.79 | R_B = 426.55 |
| Green | 10680.76 | G_R = 250.85 | G_B = 182.44 |
| Blue | 7327.48 | B_R = 151.57 | B_G = 161.84 |

Areas are in graph relative-sensitivity units × nm. Numerical values are
approximate readings from a raster graph, not original sensor measurements.
Changing the colour-selection thresholds by ±15 RGB levels changed the calculated
matrix coefficients by at most 0.00034. Axis placement and the original plotted
curves add uncertainty; this threshold check is not a complete error bound.

Before rounding, the matrix is approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.076162 | -0.029238 | -0.046923 |
| G | -0.023487 | 1.040567 | -0.017081 |
| B | -0.020686 | -0.022087 | 1.042773 |

PT stores three decimal places, consistent with the existing preset controls.
Round the off-diagonal terms first, then calculate each diagonal from those
rounded terms so every stored row sums to exactly one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.076** | **-0.029** | **-0.047** |
| G | **-0.023** | **1.040** | **-0.017** |
| B | **-0.021** | **-0.022** | **1.043** |

Select **IMX462** again in Colour Correction Matrix to use the updated built-in
preset. PT refreshes the reserved sensor preset when loading presets. Saved
**Last**, personal named presets and existing batch workflows retain their stored
settings; selecting the sensor preset does not rewrite earlier edits.

## IMX290 — recalculated 19 October 2026

Source: the author-supplied `ASI290MC-QE.jpg`, 1024 × 699 pixels, with relative
response plotted on a 0–1 scale. No internet references were used. Source SHA-256:
`8fe867cb103b1a1ed7f386f04ce973b049a5af79a73c929503939d71acd10b9e`.

Axis positions were fitted to the grid lines: the seven wavelength ticks from
400 to 1000 nm and eleven response ticks from 0 to 1. Grid-line centres were
estimated from their grayscale darkness across the plot. The resulting mapping
uses zero-based pixel coordinates:

- `x = 1.3325869215 × wavelength_nm - 358.3242356300`
- `y = 586.4629826915 - 545.7773615858 × relative_response`

For each column within the bandpass, the trace position is the median row whose
matching RGB component exceeds 160 and exceeds each other component by more than
90 (8-bit colour values). The legend lies outside the selected wavelength range.
Short covered sections were interpolated. The longest gap is 17 columns, about
13 nm, where the green trace is obscured by red near 415–427 nm. This contributes
uncertainty to the green 400–450 nm subtraction area. The first available trace
column is about 400.22 nm; its value is extended to the 400 nm boundary.

The retained [digitised samples](data/imx290-sensitivity.csv) include native pixel
columns and exact integration boundaries. Integrating them by the trapezoidal
rule gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 104.12158 | R_G = 3.48795 | R_B = 4.10471 |
| Green | 130.70548 | G_R = 5.69204 | G_B = 2.60917 |
| Blue | 77.43081 | B_R = 3.99822 | B_G = 2.62385 |

Areas are in graph relative-response units × nm; this graph uses 0–1 rather than
the IMX462 graph's 0–100 scale. Per-channel area ratios are independent of that
scale. These are approximate digitised readings, not original sensor measurements.

Before rounding, the matrix is approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.072921 | -0.033499 | -0.039422 |
| G | -0.043549 | 1.063511 | -0.019962 |
| B | -0.051636 | -0.033886 | 1.085522 |

Changing the colour-dominance threshold by ±15 RGB levels changes calculated
coefficients by at most 0.00021. Axis placement, JPEG compression, line thickness
and the obscured trace add uncertainty. In particular, R_G/R_total is close to a
three-decimal rounding boundary, so the last stored decimal is not uniquely
established by the graph.

As for IMX462, round the off-diagonals first and derive the diagonals from those
rounded values. The preset is:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.072** | **-0.033** | **-0.039** |
| G | **-0.044** | **1.064** | **-0.020** |
| B | **-0.052** | **-0.034** | **1.086** |

Select **IMX290** again to load the updated built-in preset. Previously saved
Last settings, personal presets and batch workflows retain their own matrices.

## IMX183 — recalculated 19 October 2026

Source: the author-supplied `imx183.png`, 1200 × 700 pixels, with relative response
plotted on a 0–100% scale. No internet references were used. Source SHA-256:
`c4fd1cd7c2e28a437206e773ab4814748694a6849deea63450aa480e221be2ec`.

The seven wavelength grid lines from 400 to 700 nm and eleven response grid lines
from 0 to 100 were fitted using their grayscale darkness to estimate their centres.
The mapping uses zero-based image pixel coordinates:

- `x = 2.6536789335 × wavelength_nm - 836.8448362005`
- `y = 583.5821651249 - 5.2111164272 × relative_response_percent`

Each trace was extracted using its median vertical position in each image column.
Matching pixels have their channel's RGB component above 100 and more than 60
above each other component. Short gaps at curve crossings were interpolated;
the largest internal gap is four columns, about 1.5 nm. The first available
trace values extend to the exact 400 nm boundary (less than 0.6 nm of extension).
Only 400–680 nm is integrated, despite the graph extending to 700 nm.

The [digitised samples](data/imx183-sensitivity.csv) retain native horizontal
sampling plus the exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 9393.78 | R_G = 344.45 | R_B = 404.33 |
| Green | 13086.67 | G_R = 486.28 | G_B = 293.27 |
| Blue | 8476.18 | B_R = 353.59 | B_G = 249.18 |

Areas are in relative-response percentage units × nm. Before rounding, the
matrix is approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.079709 | -0.036667 | -0.043042 |
| G | -0.037158 | 1.059568 | -0.022410 |
| B | -0.041716 | -0.029398 | 1.071114 |

Changing the colour-dominance threshold by ±15 RGB levels changed calculated
coefficients by at most 0.000021 and did not change the rounded preset. This is
a trace-selection check, not a complete uncertainty estimate: axis calibration,
line thickness and the original plotted curves also limit accuracy.

Rounding off-diagonals to three decimals and recalculating the diagonals to
preserve unit row sums gives:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.080** | **-0.037** | **-0.043** |
| G | **-0.037** | **1.059** | **-0.022** |
| B | **-0.042** | **-0.029** | **1.071** |

Select **IMX183** again to load the updated built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX224 — recalculated 19 October 2026

Source: the author-supplied `IMX224.jpg`, 1269 × 784 pixels, with relative response
plotted on a 0–1 scale. No internet references were used. Source SHA-256:
`099c7dfbafa41a3404dc21f7d6ede47e817c50016a3066c9e61c3f6f9246411e`.

The grid-line centres were measured using median grayscale darkness across the
plot. The grid spacing is slightly uneven in this raster, so pixel coordinates
were mapped by linear interpolation between neighbouring grid lines rather than
using one global scale. Zero-based pixel coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 151.876, 240.469, 329.088, 420.522,
  509.161, 597.889, 686.513, 775.119, 863.829, 955.171, 1043.902, 1132.496, 1221.100.
- Responses 0, 0.1, …, 1.0: y = 652.086, 596.715, 541.246, 485.897, 430.443,
  375.011, 319.604, 264.103, 205.985, 150.570, 95.101.

Trace pixels have their matching RGB component above 100 and more than 60 above
each other component. Small disconnected colour fragments with fewer than 30
pixels were excluded using eight-neighbour connectivity; these are JPEG fringes
around the black labels, not parts of the curves. The median vertical trace
position was measured in each column. A two-column gap in red was interpolated
(about 1.1 nm); the first sampled values extend about 0.07 nm to reach 400 nm.
The legend is outside the sampled plot area.

The [digitised samples](data/imx224-sensitivity.csv) include the native horizontal
samples and exact integration boundaries. Trapezoidal integration over the
specified 400–680 nm bandpass gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 108.88953 | R_G = 3.88384 | R_B = 4.86541 |
| Green | 129.12520 | G_R = 6.36448 | G_B = 2.84595 |
| Blue | 78.12564 | B_R = 5.14754 | B_G = 3.25550 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.080350 | -0.035668 | -0.044682 |
| G | -0.049289 | 1.071329 | -0.022040 |
| B | -0.065888 | -0.041670 | 1.107558 |

Changing the colour-dominance threshold by ±15 RGB levels changed coefficients
by at most 0.000075 without changing the rounded preset. Axis placement,
compression and line thickness add uncertainty, so these remain approximate
graph readings rather than original sensor measurements.

Round the off-diagonals first and calculate the diagonals from those rounded
values to preserve unit row sums:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.081** | **-0.036** | **-0.045** |
| G | **-0.049** | **1.071** | **-0.022** |
| B | **-0.066** | **-0.042** | **1.108** |

Select **IMX224** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.
