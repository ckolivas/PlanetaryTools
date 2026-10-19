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

## IMX485 — recalculated 19 October 2026

Source: the author-supplied `IMX485.png`, 608 × 433 pixels, labelled
**FSM-IMX485C (Color)**, with relative response plotted on a 0–1 scale.
No internet references were used. Source SHA-256:
`6b97602b5ae2fbcf505143ade35ad6c10bf59628fa2e2b3f4bce1d56492fbdfd`.

Median grayscale darkness was used to locate the grid-line centres. Coordinates
were mapped by linear interpolation between neighbouring grid lines. Zero-based
pixel coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 500, …, 1000 nm: x = 103.173, 182.754, 262.362, 341.829,
  421.580, 501.171, 580.746.
- Responses 0, 0.1, …, 1.0: y = 365.398, 334.567, 304.333, 273.500, 242.848,
  212.333, 181.667, 151.000, 120.500, 89.667, 59.191.

Trace pixels have their matching RGB component above 100 and more than 60 above
each other component. Disconnected fragments smaller than 30 pixels were removed
with eight-neighbour connectivity. The median vertical position was measured in
each column. Internal gaps were interpolated, with the largest being three blue
columns at a curve crossing (about 3.8 nm). Red is first resolved at about
403.6 nm; its first measured value was held constant back to 400 nm. Green and
blue similarly extend about 1 nm to the bandpass boundary. These small extensions
are estimates, not measured detail from the graph.

The [digitised samples](data/imx485-sensitivity.csv) retain the native horizontal
samples plus exact integration boundaries. Trapezoidal integration within the
specified 400–680 nm bandpass gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 104.43568 | R_G = 3.80635 | R_B = 5.22101 |
| Green | 131.48293 | G_R = 6.18565 | G_B = 2.84537 |
| Blue | 84.31686 | B_R = 4.40749 | B_G = 3.47806 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.086439 | -0.036447 | -0.049993 |
| G | -0.047045 | 1.068686 | -0.021641 |
| B | -0.052273 | -0.041250 | 1.093523 |

Changing the colour-dominance threshold by ±15 RGB levels changed coefficients
by at most 0.000346 without changing the rounded preset. The graph's small pixel
dimensions, line thickness, axis placement and boundary extensions add uncertainty;
these values are approximate graph readings rather than original sensor data.

Round off-diagonals to three decimals, then calculate diagonals from those
rounded values so every row sums to one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.086** | **-0.036** | **-0.050** |
| G | **-0.047** | **1.069** | **-0.022** |
| B | **-0.052** | **-0.041** | **1.093** |

Select **IMX485** again to load the updated built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX571 — recalculated 19 October 2026

Source: the author-supplied `imx571.png`, 1200 × 900 pixels, with relative response
plotted on a 0–100% scale. No internet references were used. Source SHA-256:
`baba30c79731f44fcfb7a0fc3cd2633a03001c3b0fcb223d3365604358f49317`.

Median grayscale darkness located the grid-line centres. Pixel positions were
mapped by linear interpolation between neighbouring ticks. Zero-based pixel
coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 182.000, 256.552, 331.000, 405.115,
  479.885, 554.000, 628.500, 703.000, 777.110, 851.885, 926.000, 1000.369, 1075.000.
- Responses 0, 10, …, 100%: y = 704.885, 646.000, 588.000, 529.369, 471.000,
  412.631, 354.000, 296.000, 237.115, 179.000, 120.500.

Trace pixels have their matching RGB component above 100 and more than 60 above
each other component. Their median vertical position was measured in each
column. Short gaps at crossings were interpolated; the largest is eleven green
columns (about 7.4 nm), obscured by red near 407–414 nm. The short visible green
segment at the 400 nm edge was retained. The bandpass ends at 680 nm, excluding
the near-infrared response shown on the rest of the graph.

The [digitised samples](data/imx571-sensitivity.csv) contain native horizontal
samples plus exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 9095.77 | R_G = 250.97 | R_B = 339.30 |
| Green | 13219.49 | G_R = 496.25 | G_B = 285.16 |
| Blue | 8588.67 | B_R = 308.94 | B_G = 226.10 |

Areas are in relative-response percentage units × nm. Before rounding, the
matrix is approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.064895 | -0.027592 | -0.037303 |
| G | -0.037539 | 1.059110 | -0.021571 |
| B | -0.035971 | -0.026326 | 1.062297 |

Changing the colour-dominance threshold by ±15 RGB levels changes coefficients
by at most 0.000098. G_B/G_total is close to a rounding boundary: the less strict
threshold rounds it to 0.021 instead of 0.022, with the green diagonal changing
accordingly. The stored preset follows the central extraction threshold used
above. Axis placement, line thickness and interpolation also limit accuracy;
these are approximate readings from a graph, not original sensor measurements.

Round off-diagonals to three decimals and derive the diagonals from those
rounded terms to preserve unit row sums:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.065** | **-0.028** | **-0.037** |
| G | **-0.038** | **1.060** | **-0.022** |
| B | **-0.036** | **-0.026** | **1.062** |

Select **IMX571** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX585 — recalculated 19 October 2026

Source: the author-supplied `IMX585-QE.jpg`, 1156 × 1000 pixels, titled **IMX585 QE
Curve**, with relative response plotted on a 0–1 scale. No internet references
were used. Source SHA-256:
`338f274a0e4c2ebaa9904904b3151f77befac9c22e65b2e49481ab07ee20ae87`.

Grid-line centres were located using median grayscale darkness across the plot.
Pixel positions were mapped by linear interpolation between neighbouring ticks.
Zero-based pixel coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 140.500, 220.624, 300.624, 381.624,
  461.624, 541.624, 621.624, 701.624, 782.624, 862.624, 942.624, 1022.624, 1103.500.
- Responses 0, 0.1, …, 1.0: y = 865.500, 805.624, 745.624, 684.624, 624.500,
  564.500, 504.500, 444.500, 384.500, 324.500, 264.500.

Trace pixels have their matching RGB component above 100 and more than 60 above
each other component. Their median vertical position was measured in each column.
A single missing red column at a crossing was interpolated (about 0.62 nm).
The first sampled trace values extend about 0.31 nm to reach the 400 nm boundary.
The title and legend are outside the sampled plot area. Only the 400–680 nm
bandpass contributes to the calculation; the high near-infrared responses do not.

The [digitised samples](data/imx585-sensitivity.csv) retain native horizontal
samples and exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 102.19372 | R_G = 2.29376 | R_B = 2.89997 |
| Green | 121.70174 | G_R = 6.17197 | G_B = 1.89745 |
| Blue | 72.08331 | B_R = 3.66472 | B_G = 1.82947 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.050822 | -0.022445 | -0.028377 |
| G | -0.050714 | 1.066305 | -0.015591 |
| B | -0.050840 | -0.025380 | 1.076220 |

Changing the colour-dominance threshold by ±15 RGB levels changed coefficients
by at most 0.000034 without changing the rounded preset. Axis placement,
line thickness and JPEG compression also limit accuracy; these remain approximate
graph readings rather than original sensor measurements.

Round the off-diagonals to three decimals, then calculate the diagonals from
those rounded values to preserve unit row sums:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.050** | **-0.022** | **-0.028** |
| G | **-0.051** | **1.067** | **-0.016** |
| B | **-0.051** | **-0.025** | **1.076** |

Select **IMX585** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX662 — recalculated 19 October 2026

Source: the author-supplied `IMX662.png`, 591 × 445 pixels, labelled
**FSM-IMX662C (Color)**, with relative response plotted on a 0–1 scale.
No internet references were used. Source SHA-256:
`1a4de495ab88322ef55f7169d4c45860edb151bad36a5ee4b12c716b2b64733b`.

This source is noticeably blurred and contains colour speckles. Grid-line centres
were estimated from median grayscale darkness, and positions were mapped by linear
interpolation between neighbouring ticks. Zero-based pixel coordinates, rounded
here to 0.001 pixel:

- Wavelengths 400, 500, …, 1000 nm: x = 94.228, 173.413, 251.228, 330.685,
  410.080, 488.070, 567.026.
- Responses 0, 0.1, …, 1.0: y = 373.038, 344.168, 314.278, 284.494, 255.583,
  224.998, 196.148, 165.349, 135.815, 105.657, 76.955.

The dark central traces were selected using the following 8-bit RGB conditions:

- Red: R > G + 40, R > B + 30, G < 180.
- Green: G > R + 20, G > B + 10, R < 180.
- Blue: B > R + 20, B > G + 10, R < 180.

The lower colour-difference thresholds accommodate the muted green and blue
traces. Isolated green components smaller than four pixels were removed using
eight-neighbour connectivity; the continuous green trace was retained. The median
trace row in each column was used, with linear interpolation across missing blue
sections. The longest missing run is six columns (about 7.6 nm) near the green/blue
crossing around 480 nm. Other blue gaps near 543–568 nm occur where it is faint or
crosses red. These interpolations and the broad, blurred lines limit accuracy.

The [digitised samples](data/imx662-sensitivity.csv) retain native horizontal
samples plus exact integration boundaries. Trapezoidal integration within
400–680 nm gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 104.74397 | R_G = 4.80427 | R_B = 3.96809 |
| Green | 151.96475 | G_R = 11.96102 | G_B = 6.39191 |
| Blue | 87.94749 | B_R = 9.84714 | B_G = 4.86220 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.083751 | -0.045867 | -0.037884 |
| G | -0.078709 | 1.120771 | -0.042062 |
| B | -0.111966 | -0.055285 | 1.167251 |

Varying the colour-difference thresholds by ±5 RGB levels changes coefficients
by at most 0.000497. The three-decimal preset remains unchanged in these checks,
but this does not establish three-decimal physical accuracy: blur, interpolation
and axis placement add uncertainty beyond threshold selection.

Round off-diagonals to three decimals, then calculate the diagonals from those
rounded values so every row sums to one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.084** | **-0.046** | **-0.038** |
| G | **-0.079** | **1.121** | **-0.042** |
| B | **-0.112** | **-0.055** | **1.167** |

Select **IMX662** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX664 — recalculated 19 October 2026

Source: the author-supplied `IMX664-Curve.png`, 1597 × 898 pixels, titled
**IMX664 QE Curve**, with relative response plotted on a 0–1 scale.
No internet references were used. Source SHA-256:
`65d95327356aa949db4d09551a75e56219fc40c32e2068103c51a4de4960e7eb`.

The light grid-line centres were located from median grayscale brightness above
the dark background. Pixel coordinates were mapped by linear interpolation
between neighbouring ticks. Zero-based pixel coordinates, rounded here to
0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 104.500, 225.500, 346.500, 468.618,
  589.618, 711.618, 832.618, 954.618, 1075.618, 1197.618, 1318.618, 1439.618, 1561.500.
- Responses 0, 0.1, …, 1.0: y = 807.500, 743.618, 680.500, 616.500, 552.500,
  488.500, 424.500, 361.500, 297.500, 233.500, 169.500.

The bright coloured trace centres were selected using these 8-bit RGB conditions:

- Red: R > 140, R > G + 90, R > B + 90.
- Green: G > 100, G > R + 60, G > B + 35.
- Blue/cyan: B > 130, B > R + 90, B > G + 25.

These exclude the dark shaded outlines, grayscale watermark and grid. The title
and legend are outside the sampled plot area. Median trace positions were measured
per column. Short red gaps at its green crossing were interpolated, with the
largest missing run being three columns (about 1.2 nm). Only the 400–680 nm
bandpass was integrated.

The [digitised samples](data/imx664-sensitivity.csv) retain native horizontal
sampling plus exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 101.14211 | R_G = 2.21768 | R_B = 2.91933 |
| Green | 125.32349 | G_R = 6.17068 | G_B = 2.07846 |
| Blue | 75.64040 | B_R = 3.40888 | B_G = 1.96297 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.050790 | -0.021926 | -0.028864 |
| G | -0.049238 | 1.065823 | -0.016585 |
| B | -0.045067 | -0.025951 | 1.071018 |

Varying the colour-difference thresholds by ±15 RGB levels changes coefficients
by at most 0.000028 without changing the rounded preset. The plotted curves, line
thickness and axis placement also limit accuracy; these are approximate graph
readings rather than original sensor measurements.

Round the off-diagonals to three decimals, then derive the diagonals from those
rounded terms so every row sums to one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.051** | **-0.022** | **-0.029** |
| G | **-0.049** | **1.066** | **-0.017** |
| B | **-0.045** | **-0.026** | **1.071** |

Select **IMX664** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX676 — recalculated 19 October 2026

Source: the author-supplied `IMX676.jpg`, 1200 × 1570 pixels, headed
**Quantum Efficiency**, identifying the ASI676MC and plotting relative response
on a 0–1 scale. No internet references were used. Source SHA-256:
`6ebadafd8d08f710e3c0f8d081421ff24516b5291d91871144acdeac3d35ef8f`.

The stated estimated 83% peak QE does not affect the calculation: a common
amplitude factor cancels when dividing each subtraction area by its channel's
total. Only the plotted RGB traces within the ideal 400–680 nm bandpass are used.

Grid-line centres were located from median grayscale darkness relative to the
light background. Pixel coordinates were mapped by linear interpolation between
neighbouring ticks. Zero-based pixel coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 181.500, 256.221, 331.157, 405.541,
  480.500, 555.157, 629.763, 704.669, 779.224, 854.155, 928.669, 1003.431, 1078.331.
- Responses 0, 0.1, …, 1.0: y = 1321.608, 1262.881, 1204.217, 1145.662,
  1087.275, 1028.426, 969.664, 910.500, 852.698, 793.974, 735.331.

For each colour, select pixels whose corresponding 8-bit RGB channel is greater
than 100 and exceeds both other channels by more than 60. Use the median trace
row in each column. The one-column red gap near its blue crossing and two-column
blue gap near its green crossing were interpolated linearly; the latter spans
about 1.3 nm of missing samples. The first sampled column, x = 182 (about
400.33 nm), was extended constantly back to 400 nm. Text and labels outside the
plot were excluded.

The [digitised samples](data/imx676-sensitivity.csv) retain native horizontal
sampling plus exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 105.21217 | R_G = 4.77048 | R_B = 3.70614 |
| Green | 152.75759 | G_R = 12.10159 | G_B = 6.51336 |
| Blue | 90.37506 | B_R = 10.19381 | B_G = 6.82892 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.080567 | -0.045341 | -0.035225 |
| G | -0.079221 | 1.121859 | -0.042639 |
| B | -0.112794 | -0.075562 | 1.188357 |

Varying the colour-difference thresholds by ±15 RGB levels changes coefficients
by at most 0.000051 without changing the rounded preset. Line thickness, axis
placement and the original plotted curves also limit accuracy; these remain
approximate graph readings rather than original sensor measurements.

Round the off-diagonals to three decimals, then derive the diagonals from those
rounded terms so every row sums to one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.080** | **-0.045** | **-0.035** |
| G | **-0.079** | **1.122** | **-0.043** |
| B | **-0.113** | **-0.076** | **1.189** |

Select **IMX676** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX678 — recalculated 19 October 2026

Source: the author-supplied `IMX678.png`, 2242 × 1272 pixels, showing RGB relative
response on a 0–1 scale. No internet references were used. Source SHA-256:
`2512eebd22dbac1148fedeb380463368e3131380acbc9d79725d4e3128bb6c4a`.

Grid-line centres were located from median grayscale darkness relative to the
white background. Pixel coordinates were mapped by linear interpolation between
neighbouring ticks. Zero-based pixel coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 281.518, 427.368, 571.186, 715.000,
  858.648, 1002.297, 1146.104, 1291.933, 1435.753, 1579.565, 1723.398, 1867.223,
  2012.698.
- Responses 0, 0.1, …, 1.0: y = 1099.398, 1008.276, 919.347, 830.228,
  739.094, 649.963, 560.843, 469.712, 380.574, 289.463, 200.326.

For each colour, select pixels whose corresponding 8-bit RGB channel is greater
than 100 and exceeds both other channels by more than 60. Use the median trace
row in each column. The three-column red gap near its blue crossing (about
1.0 nm of missing samples) was interpolated linearly. The first visible trace
column, x = 282 (about 400.17 nm), was extended constantly back to 400 nm.
The legend and labels are outside the sampled plot area. Only the ideal
400–680 nm bandpass was integrated.

The [digitised samples](data/imx678-sensitivity.csv) retain native horizontal
sampling plus exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 103.94550 | R_G = 4.63048 | R_B = 3.70490 |
| Green | 150.58724 | G_R = 11.77524 | G_B = 6.27617 |
| Blue | 88.99020 | B_R = 9.55856 | B_G = 6.43772 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.080190 | -0.044547 | -0.035643 |
| G | -0.078195 | 1.119873 | -0.041678 |
| B | -0.107411 | -0.072342 | 1.179753 |

Varying the colour-difference thresholds by ±15 RGB levels changes coefficients
by at most 0.000136. With the looser threshold, B_R rounds to -0.108 instead of
-0.107, making the derived blue diagonal 1.180 instead of 1.179. The central
threshold values below are retained. This final-digit sensitivity, line thickness
and axis placement limit accuracy; these are approximate graph readings rather
than original sensor measurements.

Round the off-diagonals to three decimals, then derive the diagonals from those
rounded terms so every row sums to one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.081** | **-0.045** | **-0.036** |
| G | **-0.078** | **1.120** | **-0.042** |
| B | **-0.107** | **-0.072** | **1.179** |

Select **IMX678** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.

## IMX715 — recalculated 19 October 2026

Source: the author-supplied `IMX715.jpg`, 1200 × 1379 pixels, headed
**Quantum Efficiency**, identifying the ASI715MC and plotting relative response
on a 0–1 scale. No internet references were used. Source SHA-256:
`9454b056d935218e0f2502d6f4807cc6d31cc2adc187e6792578d00ac0032240`.

The stated estimated 80% peak QE does not affect the calculation: a common
amplitude factor cancels when dividing each subtraction area by its channel's
total. Only the plotted RGB traces within the ideal 400–680 nm bandpass are used.

Grid-line centres were located from median grayscale darkness relative to the
light background. Pixel coordinates were mapped by linear interpolation between
neighbouring ticks. Zero-based pixel coordinates, rounded here to 0.001 pixel:

- Wavelengths 400, 450, …, 1000 nm: x = 185.000, 259.000, 333.885, 408.000,
  482.500, 557.000, 631.243, 706.000, 780.000, 854.757, 929.000, 1003.375, 1078.000.
- Responses 0, 0.1, …, 1.0: y = 1159.897, 1101.115, 1043.000, 984.375,
  926.000, 867.757, 809.000, 751.000, 692.214, 634.000, 575.500.

For each colour, select pixels whose corresponding 8-bit RGB channel is greater
than 100 and exceeds both other channels by more than 60. Use the median trace
row in each column. Short gaps at curve crossings were interpolated linearly:
three green columns near 590 nm, two blue columns near 482 nm and one blue column
near 561 nm. The longest run spans about 2.0 nm of missing samples. All three
traces have a sample at 400 nm. Text and labels outside the plot were excluded.

The [digitised samples](data/imx715-sensitivity.csv) retain native horizontal
sampling plus exact integration boundaries. Trapezoidal integration gives:

| Channel | Total area | First subtraction area | Second subtraction area |
| --- | ---: | ---: | ---: |
| Red | 104.65044 | R_G = 4.80264 | R_B = 4.28063 |
| Green | 150.41553 | G_R = 11.74112 | G_B = 6.33534 |
| Blue | 87.33233 | B_R = 9.72053 | B_G = 4.70436 |

Areas are in relative-response units × nm. Before rounding, the matrix is
approximately:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | 1.086796 | -0.045892 | -0.040904 |
| G | -0.078058 | 1.120177 | -0.042119 |
| B | -0.111305 | -0.053867 | 1.165172 |

Varying the colour-difference thresholds by ±15 RGB levels changes coefficients
by at most 0.000062 without changing the rounded preset. Line thickness, axis
placement and the original plotted curves also limit accuracy; these remain
approximate graph readings rather than original sensor measurements.

Round the off-diagonals to three decimals, then derive the diagonals from those
rounded terms so every row sums to one:

| Output / input | R | G | B |
| --- | ---: | ---: | ---: |
| R | **1.087** | **-0.046** | **-0.041** |
| G | **-0.078** | **1.120** | **-0.042** |
| B | **-0.111** | **-0.054** | **1.165** |

Select **IMX715** again to load the revised built-in preset. Saved Last settings,
personal presets and existing batch workflows retain their own matrices.
