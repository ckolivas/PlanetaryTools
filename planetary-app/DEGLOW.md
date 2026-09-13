# Deglow (trial implementation)

Open **Enhance → Deglow…**. Toggle **Preview on canvas** to compare with the
original; **OK** applies the result and **Cancel** discards it. Undo/redo,
presets and batch workflows are supported.

- **Amount** controls how much estimated glow is subtracted (default 100%).
- **Glow scale** controls the background sampling radius in original-image
  pixels. Set it to the width of the diffuse glow. Smaller values follow
  narrower glow; larger values estimate broader variations.
- **Protect above** is a percentage of the subject peak above the sky level.
  Lower it to protect more faint disk/ring material; raise it if too much glow
  is included in the protected area.
- **Protection margin** adds unchanged pixels around that area.
- **Feather** controls the gradual transition to full removal outside it.

The filter estimates a smooth background separately in each linear RGB
channel. Extended bright regions are excluded from that estimate. Single-pixel
moons and stars are suppressed in the background model before downsampling;
the original image pixels are never median-filtered in the output. Isolated
bright points do not create protected patches of leftover glow. The filter
subtracts the estimated excess above the
dark sky level, preserving the sky pedestal and the protected subject. It
preserves alpha and does not rescale highlights.

This is an adjustable background estimate: broad, faint real features can be
included in it. Check the preview around faint rings and moons before applying.
It targets diffuse surrounding glow, rather than sharpening/deconvolution
ringing at a bright edge.
