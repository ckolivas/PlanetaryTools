#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Stretch Contrast OKLab - GIMP 3.0 plugin by Con Kolivas <kernel@kolivas.org>
#
# Stretches image contrast using OKLab L as the brightness measure,
# but applies the stretch as a proportional RGB scale (like HSV stretch)
# to guarantee no channel clipping.
#
# For each pixel:
#   scale_desired = L_new / L_old   (L_new = stretched target luminance)
#   scale_max     = 1.0 / max(r,g,b) (maximum scale before any channel clips)
#   scale_applied = min(scale_desired, scale_max)
#   output        = (r, g, b) * scale_applied
#
# This is identical in structure to Stretch Contrast HSV, but uses perceptual
# OKLab luminance to decide how bright each pixel should become, rather than
# HSV value (max channel). For neutral greys the two are equivalent.

import sys
import struct
import gi
gi.require_version('Gimp', '3.0')
gi.require_version('Gegl', '0.4')
from gi.repository import Gimp, Gegl, GLib
import traceback

CHUNK_HEIGHT = 128

class StretchContrastOklab(Gimp.PlugIn):

    def do_set_i18n(self, name):
        return False

    def do_query_procedures(self):
        return ['plug-in-stretch-contrast-oklab']

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.run, None)
        procedure.set_menu_label('Stretch Contrast _OKLab')
        procedure.set_documentation(
            'Stretch contrast using OKLab luminance.',
            'Finds the minimum and maximum OKLab L values and stretches '
            'them to [0, 1] by scaling RGB channels proportionally per pixel. '
            'Hue and saturation are preserved and no channel clipping occurs.',
            name)
        procedure.add_menu_path('<Image>/Colors/Auto')
        procedure.set_attribution('Con Kolivas', 'Con Kolivas', '2026')
        procedure.set_image_types('RGB*')
        return procedure

    @staticmethod
    def rgb_to_L(r, g, b):
        """Compute OKLab L only — no need for a and b during pass 1."""
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = l ** (1.0/3.0) if l > 0.0 else 0.0
        m_ = m ** (1.0/3.0) if m > 0.0 else 0.0
        s_ = s ** (1.0/3.0) if s > 0.0 else 0.0
        return 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        # Prefer the explicitly passed drawable, but fall back to the active
        # drawable when drawables is empty or contains an invalid item — GIMP 3
        # does not always populate the drawables argument reliably.
        if len(drawables) == 1 and drawables[0].is_valid():
            drawable = drawables[0]
        else:
            drawable = image.get_active_drawable()
            if drawable is None or not drawable.is_valid():
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CALL_ERROR,
                    GLib.Error.new_literal(
                        GLib.quark_from_string('gimp-plug-in-error'), 0,
                        'No valid drawable found. Please select a layer.'))

        try:
            Gegl.init(None)
            image.undo_group_start()

            width  = drawable.get_width()
            height = drawable.get_height()

            # --- Pass 1: find global OKLab L min and max ---
            L_min =  1.0e38
            L_max = -1.0e38

            for y in range(0, height, CHUNK_HEIGHT):
                chunk_h    = min(CHUNK_HEIGHT, height - y)
                chunk_rect = Gegl.Rectangle.new(0, y, width, chunk_h)
                raw = drawable.get_buffer().get(
                    chunk_rect, 1.0, 'RGB float', Gegl.AbyssPolicy.NONE)
                n = width * chunk_h
                for i in range(n):
                    r, g, b = struct.unpack_from('fff', raw, i * 12)
                    # Clamp to [0,1]: 32-bit float images from RAW processing
                    # can contain small negative values. Negative r/g/b causes
                    # V = max(r,g,b) < 0, making f_max = 1/V negative and
                    # enormously large in magnitude, producing blown highlights.
                    r = max(0.0, min(1.0, r))
                    g = max(0.0, min(1.0, g))
                    b = max(0.0, min(1.0, b))
                    L = self.rgb_to_L(r, g, b)
                    if L < L_min:
                        L_min = L
                    if L > L_max:
                        L_max = L

            L_range = L_max - L_min
            if L_range < 1e-6:
                image.undo_group_end()
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

            scale = 1.0 / L_range

            # --- Pass 2: find the actual maximum output brightness ---
            # The pixel with the highest OKLab L may not be the brightest by
            # max(r,g,b), so the output peak can fall short of 1.0 without
            # this renormalisation step.
            out_max = 0.0
            for y in range(0, height, CHUNK_HEIGHT):
                chunk_h    = min(CHUNK_HEIGHT, height - y)
                chunk_rect = Gegl.Rectangle.new(0, y, width, chunk_h)
                raw = drawable.get_buffer().get(
                    chunk_rect, 1.0, 'RGB float', Gegl.AbyssPolicy.NONE)
                n = width * chunk_h
                for i in range(n):
                    r, g, b = struct.unpack_from('fff', raw, i * 12)
                    r = max(0.0, min(1.0, r))
                    g = max(0.0, min(1.0, g))
                    b = max(0.0, min(1.0, b))
                    L = self.rgb_to_L(r, g, b)
                    if L < 1e-7:
                        continue
                    L_new = (L - L_min) * scale
                    f_desired = L_new / L
                    V = max(r, g, b)
                    f_max = (1.0 / V) if V > 1e-7 else f_desired
                    f = min(f_desired, f_max)
                    peak = V * f
                    if peak > out_max:
                        out_max = peak

            if out_max < 1e-7:
                image.undo_group_end()
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

            # Renormalise so the brightest output pixel reaches exactly 1.0
            renorm = 1.0 / out_max

            # --- Pass 3: apply proportional RGB scale per pixel ---
            shadow = drawable.get_shadow_buffer()

            for y in range(0, height, CHUNK_HEIGHT):
                chunk_h    = min(CHUNK_HEIGHT, height - y)
                chunk_rect = Gegl.Rectangle.new(0, y, width, chunk_h)
                raw = drawable.get_buffer().get(
                    chunk_rect, 1.0, 'RGB float', Gegl.AbyssPolicy.NONE)
                n = width * chunk_h
                out = bytearray()
                for i in range(n):
                    r, g, b = struct.unpack_from('fff', raw, i * 12)
                    r = max(0.0, min(1.0, r))
                    g = max(0.0, min(1.0, g))
                    b = max(0.0, min(1.0, b))
                    L = self.rgb_to_L(r, g, b)

                    if L < 1e-7:
                        out += struct.pack('fff', 0.0, 0.0, 0.0)
                        continue

                    L_new = (L - L_min) * scale
                    f_desired = L_new / L
                    V = max(r, g, b)
                    f_max = (1.0 / V) if V > 1e-7 else f_desired
                    f = min(f_desired, f_max) * renorm

                    out += struct.pack('fff',
                                       max(0.0, min(1.0, r * f)),
                                       max(0.0, min(1.0, g * f)),
                                       max(0.0, min(1.0, b * f)))

                shadow.set(chunk_rect, 'RGB float', bytes(out))

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

Gimp.main(StretchContrastOklab.__gtype__, sys.argv)
