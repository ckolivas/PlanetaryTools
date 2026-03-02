#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OKLab Luminance desaturate plugin for GIMP v3 by Con Kolivas <kernel@kolivas.org>
# Desaturates by extracting the OKLab L channel and setting R=G=B=L.

import sys
import gi
from array import array
gi.require_version('Gimp', '3.0')
gi.require_version('Babl', '0.1')
gi.require_version('Gegl', '0.4')
from gi.repository import Gimp
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Babl
from gi.repository import Gegl
import traceback

Gegl.init([])

# ---------------------------------------------------------------------------
# OKLab conversion via GEGL (adapted from adaptive-deconvolution plugin)
# ---------------------------------------------------------------------------

def _gegl_color_graph(src_buf, out_buf, width, height, *ops):
    graph = Gegl.Node()
    src = graph.create_child("gegl:buffer-source")
    src.set_property("buffer", src_buf)
    prev = src
    for op_name, props in ops:
        node = graph.create_child(op_name)
        for k, v in props.items():
            node.set_property(k, v)
        prev.connect_to("output", node, "input")
        prev = node
    write = graph.create_child("gegl:write-buffer")
    write.set_property("buffer", out_buf)
    prev.connect_to("output", write, "input")
    write.process()

def linearrgb_to_oklab(flat_linear, width, height):
    rect    = Gegl.Rectangle.new(0, 0, width, height)
    in_buf  = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    out_buf = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    in_buf.set(rect, "R~G~B~ float", flat_linear.tobytes())
    _gegl_color_graph(in_buf, out_buf, width, height,
        ("gegl:svg-matrix", {"values":
            "0.4122214708 0.5363325363 0.0514459929 0 0 "
            "0.2119034982 0.6806995451 0.1073969566 0 0 "
            "0.0883024619 0.2817188376 0.6299787005 0 0 "
            "0 0 0 1 0"}),
        ("gegl:gamma",      {"value": 1.0/3.0}),
        ("gegl:svg-matrix", {"values":
            "0.2104542553  0.7936177850 -0.0040720468 0 0 "
            "1.9779984951 -2.4285922050  0.4505937099 0 0 "
            "0.0259040371  0.7827717662 -0.8086757660 0 0 "
            "0 0 0 1 0"}),
    )
    raw = out_buf.get(rect, 1.0, "R~G~B~ float", Gegl.AbyssPolicy.CLAMP)
    result = array('f')
    result.frombytes(raw)
    return result

# ---------------------------------------------------------------------------

class OklabLuminance(Gimp.PlugIn):

    def do_set_i18n(self, name):
        return False

    def do_query_procedures(self):
        return ['plug-in-oklab-luminance']

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.oklab_luminance, None)
        procedure.set_menu_label('OKLab _Luminance')
        procedure.set_documentation(
            'Desaturate using OKLab luminance channel.',
            'Converts the image to greyscale by extracting the perceptual '
            'luminance (L) from the OKLab colour space.',
            name)
        procedure.add_menu_path('<Image>/Colors/Desaturate')
        procedure.set_attribution('Con Kolivas', 'Con Kolivas', '2026')
        procedure.set_image_types('RGB*')
        return procedure

    def oklab_luminance(self, procedure, run_mode, image, drawables, config, run_data):
        if len(drawables) != 1:
            return procedure.new_return_values(
                Gimp.PDBStatusType.CALL_ERROR,
                GLib.Error.new_literal(
                    GLib.quark_from_string('gimp-plug-in-error'), 0,
                    'This plug-in works on one drawable.'))

        drawable = drawables[0]
        if not drawable.is_valid():
            return procedure.new_return_values(
                Gimp.PDBStatusType.CALL_ERROR,
                GLib.Error.new_literal(
                    GLib.quark_from_string('gimp-plug-in-error'), 0,
                    'Invalid drawable.'))

        try:
            image.undo_group_start()

            width  = drawable.get_width()
            height = drawable.get_height()
            rect   = Gegl.Rectangle.new(0, 0, width, height)

            # Read drawable as linear RGB float
            buf = drawable.get_buffer()
            raw = buf.get(rect, 1.0, "R~G~B~ float", Gegl.AbyssPolicy.CLAMP)
            flat_linear = array('f')
            flat_linear.frombytes(raw)

            # Convert to OKLab and extract L channel
            flat_oklab = linearrgb_to_oklab(flat_linear, width, height)
            L_vals = flat_oklab[0::3]

            # Build R=G=B=L so the result is a greyscale representation of
            # perceptual luminance, stored in the same linear-RGB buffer format.
            out_flat = array('f', [0.0] * (width * height * 3))
            out_flat[0::3] = L_vals
            out_flat[1::3] = L_vals
            out_flat[2::3] = L_vals

            # Write back via shadow buffer
            shadow = drawable.get_shadow_buffer()
            shadow.set(rect, "R~G~B~ float", out_flat.tobytes())
            shadow.flush()
            drawable.merge_shadow(True)
            drawable.update(0, 0, width, height)

            image.undo_group_end()
            Gimp.displays_flush()
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

        except Exception:
            image.undo_group_end()
            return procedure.new_return_values(
                Gimp.PDBStatusType.EXECUTION_ERROR,
                GLib.Error.new_literal(
                    GLib.quark_from_string('gimp-plug-in-error'), 0,
                    traceback.format_exc()))

Gimp.main(OklabLuminance.__gtype__, sys.argv)
