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
