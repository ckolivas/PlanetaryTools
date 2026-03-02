#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# OKLab Decompose - GIMP 3.0 Python plug-in by Con Kolivas <kernel@kolivas.org>

import sys
import struct

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl
from gi.repository import GLib

CHUNK_HEIGHT = 128

class OkLabDecompose(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["plug-in-oklab-decompose"]

    def do_set_i18n(self, name):
        return False

    def do_create_procedure(self, name):
        if name != "plug-in-oklab-decompose":
            return None

        procedure = Gimp.ImageProcedure.new(
            self, name,
            Gimp.PDBProcType.PLUGIN,
            self.run, None
        )

        procedure.set_image_types("RGB*")
        procedure.set_menu_label("OKLab Decompose")
        procedure.add_menu_path("<Image>/Colors/Components/")
        procedure.set_documentation(
            "Decompose image to 32-bit OKLab L, a, b grayscale layers",
            "Creates a new 32-bit floating-point grayscale image with "
            "three layers: OKLab L (0–1), a (+0.5), b (+0.5).",
            name
        )
        procedure.set_attribution("Grok (xAI)", "Grok (xAI)", "2026")

        return procedure

    @staticmethod
    def rgb_to_oklab(r, g, b):
        """r, g, b are already linear [0,1] (from 'RGB float')"""
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l ** (1.0 / 3.0) if l > 0 else 0.0
        m_ = m ** (1.0 / 3.0) if m > 0 else 0.0
        s_ = s ** (1.0 / 3.0) if s > 0 else 0.0

        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285921218 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return L, a, b

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        if len(drawables) != 1:
            return procedure.new_return_values(
                Gimp.PDBStatusType.CALLING_ERROR,
                GLib.Error("OKLab decompose requires exactly one drawable.")
            )

        drawable = drawables[0]
        width = drawable.get_width()
        height = drawable.get_height()

        Gegl.init(None)

        # Create 32-bit linear grayscale image
        new_image = Gimp.Image.new(width, height, Gimp.ImageBaseType.GRAY)
        new_image.convert_precision(Gimp.Precision.FLOAT_LINEAR)

        # Create the three float layers
        def create_layer(name):
            layer = Gimp.Layer.new(
                new_image, name, width, height,
                Gimp.ImageType.GRAY_IMAGE, 100.0, Gimp.LayerMode.NORMAL
            )
            new_image.insert_layer(layer, None, -1)
            return layer

        layer_l = create_layer("OKLab L")
        layer_a = create_layer("OKLab a")
        layer_b = create_layer("OKLab b")

        buffer_l = layer_l.get_buffer()
        buffer_a = layer_a.get_buffer()
        buffer_b = layer_b.get_buffer()

        # Chunked processing (safe for huge 32-bit images)
        for y in range(0, height, CHUNK_HEIGHT):
            chunk_h = min(CHUNK_HEIGHT, height - y)
            chunk_rect = Gegl.Rectangle.new(0, y, width, chunk_h)

            # Read input as linear float (GIMP auto-converts gamma if needed)
            input_chunk = drawable.get_buffer().get(chunk_rect, 1.0, "RGB float", Gegl.AbyssPolicy.NONE)

            data_l = bytearray()
            data_a = bytearray()
            data_b = bytearray()

            for i in range(width * chunk_h):
                offset = i * 12
                r, g, b = struct.unpack_from('fff', input_chunk, offset)

                L, aa, bb = self.rgb_to_oklab(r, g, b)

                data_l.extend(struct.pack('f', L))
                data_a.extend(struct.pack('f', aa + 0.5))
                data_b.extend(struct.pack('f', bb + 0.5))

            buffer_l.set(chunk_rect, "Y float", data_l)
            buffer_a.set(chunk_rect, "Y float", data_a)
            buffer_b.set(chunk_rect, "Y float", data_b)

        # Force instant correct display
        buffer_l.flush()
        buffer_a.flush()
        buffer_b.flush()
        layer_l.update(0, 0, width, height)
        layer_a.update(0, 0, width, height)
        layer_b.update(0, 0, width, height)
        Gimp.displays_flush()

        Gimp.Display.new(new_image)
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)


Gimp.main(OkLabDecompose.__gtype__, sys.argv)
