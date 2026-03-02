#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OKLab Compose - GIMP 3.0 Python plug-in by Con Kolivas <kernel@kolivas.org>

import sys
import struct

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
gi.require_version('GObject', '2.0')
from gi.repository import GObject
from gi.repository import GLib
import traceback

CHUNK_HEIGHT = 128

class OkLabCompose(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["plug-in-oklab-compose"]

    def do_set_i18n(self, name):
        return False

    def do_create_procedure(self, name):
        if name != "plug-in-oklab-compose":
            return None

        procedure = Gimp.ImageProcedure.new(
            self, name,
            Gimp.PDBProcType.PLUGIN,
            self.run, None
        )

        procedure.set_image_types("*")
        procedure.set_menu_label("OKLab _Compose")
        procedure.add_menu_path("<Image>/Colors/Components/")
        procedure.set_documentation(
            "Compose RGB image from OKLab L, a, b grayscale layers",
            "Creates a new 32-bit floating-point linear RGB image "
            "from three OKLab layers (works with 8-bit or 32-bit).",
            name
        )
        procedure.set_attribution("Con Kolivas", "Con Kolivas", "2026")

        return procedure

    @staticmethod
    def oklab_to_linear_rgb(L, a, b):
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b

        l = l_ ** 3 if l_ > 0 else 0.0
        m = m_ ** 3 if m_ > 0 else 0.0
        s = s_ ** 3 if s_ > 0 else 0.0

        r  =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g  = -1.2684380046 * l + 2.6097574007 * m - 0.3413193963 * s
        bb = -0.0041960863 * l - 0.7034186145 * m + 1.7076147010 * s

        return max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, bb))

    def _make_layer_combo(self, all_layers, label_text, default_name):
        """Return (Gtk.Box, get_selected_layer callable) for a labelled layer picker."""
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        label = Gtk.Label(label=label_text)
        label.set_width_chars(10)
        label.set_xalign(0.0)
        hbox.pack_start(label, False, False, 0)

        combo = Gtk.ComboBoxText()
        for layer in all_layers:
            combo.append(str(layer.get_id()), layer.get_name())

        selected = False
        for i, layer in enumerate(all_layers):
            if layer.get_name() == default_name:
                combo.set_active(i)
                selected = True
                break
        if not selected:
            combo.set_active(0)

        hbox.pack_start(combo, True, True, 0)

        def get_layer():
            layer_id_str = combo.get_active_id()
            if layer_id_str is None:
                return None
            layer_id = int(layer_id_str)
            for layer in all_layers:
                if layer.get_id() == layer_id:
                    return layer
            return None

        return hbox, get_layer

    def _collect_all_layers(self, image):
        """Collect all layers from all open images."""
        all_layers = []
        for img in Gimp.get_images():
            for layer in img.get_layers():
                all_layers.append(layer)
        return all_layers

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        try:
            GimpUi.init("plug-in-oklab-compose")
            Gegl.init(None)

            # Gather layers from all open images so the user can pick from
            # decomposed channel images as well as layers in the current image
            all_layers = self._collect_all_layers(image)

            if len(all_layers) < 3:
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CALL_ERROR,
                    GLib.Error.new_literal(
                        GLib.quark_from_string('gimp-plug-in-error'), 0,
                        'At least 3 layers must be open to compose.'))

            dialog = Gtk.Dialog(title='OKLab Compose',
                                modal=True,
                                destroy_with_parent=True)
            dialog.add_buttons('_Cancel', Gtk.ResponseType.CANCEL,
                               '_Compose', Gtk.ResponseType.OK)
            dialog.set_default_response(Gtk.ResponseType.OK)
            dialog.set_border_width(12)

            content = dialog.get_content_area()
            content.set_spacing(6)

            info = Gtk.Label()
            info.set_markup(
                '<b>Select the three OKLab channel layers.</b>\n'
                '<small>L range: 0–1    '
                'a and b range: 0–1 display-shifted (0.5 = neutral)\n'
                'as produced by OKLab Decompose.</small>')
            info.set_xalign(0.0)
            info.set_line_wrap(True)
            content.pack_start(info, False, False, 6)
            content.pack_start(Gtk.Separator(), False, False, 4)

            row_L, get_L = self._make_layer_combo(all_layers, 'OKLab L:', 'OKLab L')
            row_a, get_a = self._make_layer_combo(all_layers, 'OKLab a:', 'OKLab a')
            row_b, get_b = self._make_layer_combo(all_layers, 'OKLab b:', 'OKLab b')

            content.pack_start(row_L, False, False, 2)
            content.pack_start(row_a, False, False, 2)
            content.pack_start(row_b, False, False, 2)

            dialog.show_all()
            response = dialog.run()
            layer_l = get_L()
            layer_a = get_a()
            layer_b = get_b()
            dialog.destroy()

            if response != Gtk.ResponseType.OK:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)

            if not layer_l or not layer_a or not layer_b:
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CALL_ERROR,
                    GLib.Error.new_literal(
                        GLib.quark_from_string('gimp-plug-in-error'), 0,
                        'You must select three layers (L, a, b).'))

            width  = layer_l.get_width()
            height = layer_l.get_height()

            if (layer_a.get_width()  != width or layer_a.get_height() != height or
                layer_b.get_width()  != width or layer_b.get_height() != height):
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CALL_ERROR,
                    GLib.Error.new_literal(
                        GLib.quark_from_string('gimp-plug-in-error'), 0,
                        'All three layers must have identical dimensions.'))

            new_image = Gimp.Image.new(width, height, Gimp.ImageBaseType.RGB)
            new_image.convert_precision(Gimp.Precision.FLOAT_LINEAR)

            layer = Gimp.Layer.new(
                new_image, 'OKLab Composed', width, height,
                Gimp.ImageType.RGB_IMAGE, 100.0, Gimp.LayerMode.NORMAL)
            new_image.insert_layer(layer, None, -1)

            buffer = layer.get_buffer()

            for y in range(0, height, CHUNK_HEIGHT):
                chunk_h    = min(CHUNK_HEIGHT, height - y)
                chunk_rect = Gegl.Rectangle.new(0, y, width, chunk_h)

                l_chunk = layer_l.get_buffer().get(chunk_rect, 1.0, "Y float", Gegl.AbyssPolicy.NONE)
                a_chunk = layer_a.get_buffer().get(chunk_rect, 1.0, "Y float", Gegl.AbyssPolicy.NONE)
                b_chunk = layer_b.get_buffer().get(chunk_rect, 1.0, "Y float", Gegl.AbyssPolicy.NONE)

                chunk_data = bytearray()
                for i in range(width * chunk_h):
                    offset = i * 4
                    L  = struct.unpack_from('f', l_chunk, offset)[0]
                    aa = struct.unpack_from('f', a_chunk, offset)[0] - 0.5
                    bb = struct.unpack_from('f', b_chunk, offset)[0] - 0.5

                    r, g, b = self.oklab_to_linear_rgb(L, aa, bb)
                    chunk_data.extend(struct.pack('fff', r, g, b))

                buffer.set(chunk_rect, "RGB float", chunk_data)

            buffer.flush()
            layer.update(0, 0, width, height)

            Gimp.Display.new(new_image)
            Gimp.displays_flush()

            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

        except Exception:
            return procedure.new_return_values(
                Gimp.PDBStatusType.EXECUTION_ERROR,
                GLib.Error.new_literal(
                    GLib.quark_from_string('gimp-plug-in-error'), 0,
                    traceback.format_exc()))


Gimp.main(OkLabCompose.__gtype__, sys.argv)
