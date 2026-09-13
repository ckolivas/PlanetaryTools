# Curves

Open **Colours → Curves…** with an image loaded. The same editor is available
for a Curves step in Batch Processing, and settings can be saved as presets.

- **Value** maps each RGB component after its individual Red, Green or Blue
  curve, as GIMP does. Greyscale images use Value. Alpha is available only
  when the input has an alpha channel and is adjusted independently.
- **Perceptual (sRGB)** is the default. **Linear light** edits linear pixel
  values. The graph, histogram and input/output coordinates use the selected
  space; the document remains float32 linear light.
- In **Smooth** mode, click to add a point and drag it to reshape the curve.
  Select a point to edit its Input/Output coordinates (0–255) or change its
  type to **Corner**. Delete, Backspace or right-click removes a selected
  point while retaining at least two. Arrow keys move it by one level;
  Shift+arrow moves it by ten.
- **Freehand** draws directly into the curve, filling gaps between mouse
  positions. Switching back to Smooth creates nine control points sampled
  from the freehand curve, following GIMP's conversion.
- Enable **Pick a curve point from the image**, then click a tone in the
  canvas to insert a point at that input level. Drag the point or change its
  Output value to adjust that tone.
- The graph shows the original input histogram for the selected channel.
  **Log histogram** changes only the histogram's vertical display scale.
  Linear mode automatically zooms the count axis when a few sky bins would
  hide a broad tonal distribution. Heights remain proportional to counts,
  with oversized peaks capped and marked **Tall peaks clipped**. Pixel data
  and histogram counts are unchanged.
- **Reset channel** clears the selected curve; **Reset all** restores all
  curves and the default perceptual mode. OK applies the result as one undo
  step. Cancel restores the original. OK also works with preview disabled.

The implementation follows the local GIMP 3 sources:
`app/core/gimpcurve.c`, `app/core/gimpcurve-map.c`,
`app/operations/gimpcurvesconfig.c` and `gimpoperationpointfilter.c`.
Smooth curves use GIMP's cubic Bézier tangents, corner handling, endpoint
extension and 256-sample lookup with linear interpolation between samples.
Edited curves map out-of-range values to their endpoints; unchanged channels
and an entirely unchanged image retain their original samples, including HDR.
Presets use Planetary Tools' own preset format, not GIMP settings files.

From the project root, run:

```sh
QT_QPA_PLATFORM=offscreen PYTHONPATH=planetary-app \
  planetary-app/.venv/bin/python -m unittest discover \
  -s planetary-app/tests -p test_curves.py -v
```

Tests cover channel order, colour space, greyscale/alpha, endpoint mapping,
freehand and corner curves, presets, batch editing, image picking, Apply,
Cancel and undo/redo. When `~/Code/gimp` and a C compiler are present, the
parity test compiles GIMP's actual curve calculation functions and compares
all 256 samples for deterministic smooth/corner configurations.
