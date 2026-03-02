#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Adaptive Deconvolution plugin for GIMP v3 by Con Kolivas <kernel@kolivas.org>

import sys
import json
import os
import gi
import math
import operator
from array import array
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')
gi.require_version('Gtk', '3.0')
gi.require_version('Babl', '0.1')
gi.require_version('Gegl', '0.4')
from gi.repository import Gimp
from gi.repository import GimpUi
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gtk
from gi.repository import Babl
from gi.repository import Gegl
import traceback
import cProfile
import pstats
import io

Gegl.init([])

# Helper functions adapted without numpy/scipy

def generate_moffat_kernel(gamma=1.0, beta=2.0, size=5):
    half = size // 2
    psf = [[0.0 for _ in range(size)] for _ in range(size)]
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            r = math.sqrt(dx**2 + dy**2)
            psf[dy + half][dx + half] = (1 + (r / gamma)**2) ** (-beta)
    total = sum(sum(row) for row in psf)
    for y in range(size):
        for x in range(size):
            psf[y][x] /= total
    return psf

# Precompute Moffat kernel and mirror
psf = generate_moffat_kernel(gamma=1.0, beta=2.0, size=5)
psf_mirror = [row[::-1] for row in psf[::-1]]

psf_flat = [psf[y][x] for y in range(5) for x in range(5)]
psf_mirror_flat = [psf_mirror[y][x] for y in range(5) for x in range(5)]

one_third = 1.0 / 3.0

def _gegl_color_graph(src_buf, out_buf, width, height, *ops):
    """Build and run a linear chain of GEGL ops: (op_name, {prop: val}, ...) """
    rect = Gegl.Rectangle.new(0, 0, width, height)
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
    # Entirely in GEGL native C:
    #   [r,g,b] --(svg-matrix M1)--> [l,m,s]
    #           --(gamma 1/3)-------> [l^1/3, m^1/3, s^1/3]
    #           --(svg-matrix M2)--> [L,a,b]
    rect = Gegl.Rectangle.new(0, 0, width, height)
    n = width * height

    in_buf  = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    out_buf = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    in_buf.set(rect, "R~G~B~ float", flat_linear.tobytes())

    _gegl_color_graph(in_buf, out_buf, width, height,
        ("gegl:svg-matrix", {"values": "0.4122214708 0.5363325363 0.0514459929 0 0 0.2119034982 0.6806995451 0.1073969566 0 0 0.0883024619 0.2817188376 0.6299787005 0 0 0 0 0 1 0"}),
        ("gegl:gamma",      {"value":  1.0/3.0}),
        ("gegl:svg-matrix", {"values": "0.2104542553 0.7936177850 -0.0040720468 0 0 1.9779984951 -2.4285922050 0.4505937099 0 0 0.0259040371 0.7827717662 -0.8086757660 0 0 0 0 0 1 0"}),
    )

    raw = out_buf.get(rect, 1.0, "R~G~B~ float", Gegl.AbyssPolicy.CLAMP)
    result = array('f')
    result.frombytes(raw)
    return result

def oklab_to_linearrgb(flat_oklab, ratio, width, height, max_val=1.0):
    # Scale L channel by per-pixel ratio in Python (unavoidable per-pixel variation),
    # then do the matrix math + cube + matrix + clamp entirely in GEGL.
    rect = Gegl.Rectangle.new(0, 0, width, height)
    n = width * height

    # Scale L (channel 0) by ratio using GEGL multiply:
    # Write ratio as single-channel buffer, multiply against L channel of oklab.
    # Build scaled array: interleave ratio*L, a, b using slices to avoid per-pixel loop.
    # flat_oklab layout: [L0,a0,b0, L1,a1,b1, ...]
    # Extract channels via slices (fast array copy), scale L, re-interleave.
    L_ch = flat_oklab[0::3]   # array slice — C-level copy, no Python loop
    a_ch = flat_oklab[1::3]
    b_ch = flat_oklab[2::3]
    L_scaled = array('f', map(operator.mul, L_ch, ratio))
    # Re-interleave: build scaled flat array
    scaled = array('f', [0.0] * (n * 3))
    scaled[0::3] = L_scaled
    scaled[1::3] = a_ch
    scaled[2::3] = b_ch

    in_buf  = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    out_buf = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    in_buf.set(rect, "R~G~B~ float", scaled.tobytes())

    _gegl_color_graph(in_buf, out_buf, width, height,
        ("gegl:svg-matrix", {"values": "1.0 0.3963377774 0.2158037573 0 0 1.0 -0.1055613458 -0.0638541728 0 0 1.0 -0.0894841775 -1.2914855480 0 0 0 0 0 1 0"}),
        ("gegl:gamma",      {"value":  3.0}),
        ("gegl:svg-matrix", {"values": "4.0767416621 -3.3077115913 0.2309699292 0 0 -1.2684380046 2.6097574011 -0.3413193965 0 0 -0.0041960863 -0.7034186147 1.7076147010 0 0 0 0 0 1 0"}),
    )

    raw = out_buf.get(rect, 1.0, "R~G~B~ float", Gegl.AbyssPolicy.CLAMP)
    result = array('f')
    result.frombytes(raw)
    return result

def linear_rgb2lum(flat_rgb, width, height):
    # Use gegl:svg-matrix to compute weighted luminance sum in native C.
    # All three output channels = 0.212671*R + 0.715160*G + 0.072169*B,
    # then read back as single-channel "Y float" which GEGL converts automatically.
    rect = Gegl.Rectangle.new(0, 0, width, height)

    in_buf  = Gegl.Buffer.new("R~G~B~ float", 0, 0, width, height)
    out_buf = Gegl.Buffer.new("Y float",      0, 0, width, height)
    in_buf.set(rect, "R~G~B~ float", flat_rgb.tobytes())

    _gegl_color_graph(in_buf, out_buf, width, height,
        ("gegl:svg-matrix", {"values":
            "0.212671 0.715160 0.072169 0 0 "
            "0.212671 0.715160 0.072169 0 0 "
            "0.212671 0.715160 0.072169 0 0 "
            "0 0 0 1 0"}),
    )

    raw = out_buf.get(rect, 1.0, "Y float", Gegl.AbyssPolicy.CLAMP)
    lum = array('f')
    lum.frombytes(raw)
    max_lum = max(lum)
    return lum, max_lum

def std_windowed(lum, width, height, win_size):
    # Compute local std dev entirely in GEGL native C:
    #   std = sqrt(E[x^2] - E[x]^2)
    #
    # Pipeline built as a single graph:
    #   lum ──► box-blur ──────────────────────────► mean
    #   lum ──► gamma(2.0) ──► box-blur ──────────► mean_sq
    #                                                  │
    #   mean_sq - mean^2 ──► gamma(0.5) ──► std
    #
    # gegl:gamma computes x^gamma, so:
    #   gamma(2.0) = x^2  (squaring)
    #   gamma(0.5) = sqrt (square root)
    # gegl:subtract computes aux - input (we use it for mean_sq - mean^2)
    # gegl:multiply computes input * aux  (we use it for mean * mean)

    wh, ww = win_size
    pad = wh // 2
    radius = pad
    rect = Gegl.Rectangle.new(0, 0, width, height)

    in_buf = Gegl.Buffer.new("Y float", 0, 0, width, height)
    in_buf.set(rect, "Y float", lum.tobytes())
    out_buf = Gegl.Buffer.new("Y float", 0, 0, width, height)

    graph = Gegl.Node()

    # Source
    src = graph.create_child("gegl:buffer-source")
    src.set_property("buffer", in_buf)

    # Branch 1: box-blur(lum) = mean
    blur_mean = graph.create_child("gegl:box-blur")
    blur_mean.set_property("radius", radius)
    src.connect_to("output", blur_mean, "input")

    # mean^2 via gamma(2.0)
    mean_sq_node = graph.create_child("gegl:gamma")
    mean_sq_node.set_property("value", 2.0)
    blur_mean.connect_to("output", mean_sq_node, "input")

    # Branch 2: lum^2 via gamma(2.0), then box-blur = E[x^2]
    sq_node = graph.create_child("gegl:gamma")
    sq_node.set_property("value", 2.0)
    src.connect_to("output", sq_node, "input")

    blur_sq = graph.create_child("gegl:box-blur")
    blur_sq.set_property("radius", radius)
    sq_node.connect_to("output", blur_sq, "input")

    # E[x^2] - mean^2 via gegl:subtract (output = input - aux in GEGL)
    # We want blur_sq - mean_sq_node, so: input=blur_sq, aux=mean_sq_node
    sub_node = graph.create_child("gegl:subtract")
    blur_sq.connect_to("output", sub_node, "input")
    mean_sq_node.connect_to("output", sub_node, "aux")

    # sqrt via gamma(0.5) — note: gamma on negative values gives NaN,
    # so we first clamp negatives to 0 using gegl:rgb-clip (clips to [0,1])
    # For luminance data in [0,1] the variance should be non-negative,
    # but floating point can give tiny negatives, so clamp first.
    clip_node = graph.create_child("gegl:rgb-clip")
    clip_node.set_property("low-limit", 0.0)
    clip_node.set_property("high-limit", 1.0)
    sub_node.connect_to("output", clip_node, "input")

    sqrt_node = graph.create_child("gegl:gamma")
    sqrt_node.set_property("value", 0.5)
    clip_node.connect_to("output", sqrt_node, "input")

    write = graph.create_child("gegl:write-buffer")
    write.set_property("buffer", out_buf)
    sqrt_node.connect_to("output", write, "input")

    write.process()

    raw = out_buf.get(rect, 1.0, "Y float", Gegl.AbyssPolicy.CLAMP)
    contrast = array('f')
    contrast.frombytes(raw)

    contrast_min = min(contrast)
    contrast_max = max(contrast)

    return contrast, contrast_min, contrast_max

def convolve2d(flat_in, kernel_flat, width, height):
    # Use GEGL's native convolution-matrix operation (runs in optimised C).
    # The op is hardcoded to 5x5; each cell is a separate named property:
    # row 1 = a1..a5, row 2 = b1..b5, row 3 = c1..c5,
    # row 4 = d1..d5, row 5 = e1..e5  (row-major, matches kernel_flat order).
    # divisor=1.0 and normalize=False so weights are used as-is.

    rect = Gegl.Rectangle.new(0, 0, width, height)

    in_buf = Gegl.Buffer.new("Y float", 0, 0, width, height)
    in_buf.set(rect, "Y float", flat_in.tobytes())

    out_buf = Gegl.Buffer.new("Y float", 0, 0, width, height)

    graph = Gegl.Node()

    src_node = graph.create_child("gegl:buffer-source")
    src_node.set_property("buffer", in_buf)

    conv_node = graph.create_child("gegl:convolution-matrix")
    prop_names = [f"{r}{c}" for r in ('a','b','c','d','e') for c in range(1, 6)]
    for name, val in zip(prop_names, kernel_flat):
        conv_node.set_property(name, float(val))
    conv_node.set_property("divisor", 1.0)
    conv_node.set_property("normalize", False)

    write_node = graph.create_child("gegl:write-buffer")
    write_node.set_property("buffer", out_buf)

    src_node.connect_to("output", conv_node, "input")
    conv_node.connect_to("output", write_node, "input")
    write_node.process()

    raw = out_buf.get(rect, 1.0, "Y float", Gegl.AbyssPolicy.CLAMP)
    out = array('f')
    out.frombytes(raw)
    return out


class AdaptiveDeconvolution(Gimp.PlugIn):
    def __init__(self):
        self.original_pixels = None
        self.process_rect = None
        self.update_x = 0
        self.update_y = 0
        self.channels = None
        self.babl_format = None
        self.has_alpha = None
        self.is_gray = None
        self.presets = None
        self.preset_combo = None
        self.timer_id = None
        self.cached_lum = None
        self.cached_original_lum = None
        self.cached_contrast_norm = None
        self.cached_sqrt_contrast_norm = None
        self.cached_correction = None
        self.cached_oklab_linear = None
        self.cached_max_val = None
        self.cached_conv = None
        self.cached_relative = None
        self.cached_adaptive = None
        self.cached_corr_minus_one = None
        self.cached_contrast = None
        self.cached_contrast_min = None
        self.cached_contrast_max = None
        self.cached_oklab = None
        self.cached_flat_rgb = None
        self.cached_alpha_flat = None
        self.cached_corr_minus_one_red = None
        self.cached_corr_minus_one_green = None
        self.cached_corr_minus_one_blue = None
        self.cached_max_val_red = None
        self.cached_max_val_green = None
        self.cached_max_val_blue = None

    def do_set_i18n(self, name):
        return False

    def do_query_procedures(self):
        return ['plug-in-adaptive-deconvolution']

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.adaptive_deconvolution, None)
        procedure.set_menu_label('Adaptive _Deconvolution...')
        procedure.set_documentation('Apply adaptive Lucy-Richardson deconvolution on luminance channel.',
                                    'Sharpen image using adaptive deconvolution algorithm.',
                                    name)
        procedure.add_menu_path('<Image>/Filters/Enhance')
        procedure.set_attribution('Con Kolivas', 'Con Kolivas', '2026')
        procedure.set_image_types('RGB*, GRAY*')

        # Add arguments
        procedure.add_double_argument('amount', '_Amount', 'Sharpening amount', 0.0, 200.0, 10.0, GObject.ParamFlags.READWRITE)
        procedure.add_boolean_argument('adaptive', 'A_daptive', 'Use adaptive contrast', True, GObject.ParamFlags.READWRITE)
        procedure.add_boolean_argument('oklab', '_OKLab', 'Process luminance using OKLab instead of processing RGB channels separately', True, GObject.ParamFlags.READWRITE)
        procedure.add_boolean_argument('preview', '_Preview', 'Apply preview on canvas', False, GObject.ParamFlags.READWRITE)

        return procedure

    def load_presets(self):
        preset_file = os.path.expanduser('~/.config/GIMP/3.0/adaptive-deconvolution-presets.json')
        if os.path.exists(preset_file):
            with open(preset_file, 'r') as f:
                return json.load(f)
        return {}

    def save_presets(self):
        preset_file = os.path.expanduser('~/.config/GIMP/3.0/adaptive-deconvolution-presets.json')
        preset_dir = os.path.dirname(preset_file)
        os.makedirs(preset_dir, exist_ok=True)
        with open(preset_file, 'w') as f:
            json.dump(self.presets, f)

    def set_combo_to_name(self, combo, name):
        model = combo.get_model()
        for i, row in enumerate(model):
            if row[0] == name:
                combo.set_active(i)
                return

    def on_preset_changed(self, combo, config, dialog, image, drawable):
        name = combo.get_active_text()
        if name:
            vals = self.presets[name]
            config.set_property('amount', vals['amount'])
            config.set_property('adaptive', vals['adaptive'])
            config.set_property('oklab', vals.get('oklab', True))
        self.debounce_update(dialog, image, drawable, config)

    def on_save_preset(self, dialog, config, image, drawable):
        name_dlg = Gtk.Dialog(title="Save Preset As", transient_for=dialog, flags=0)
        name_dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Save", Gtk.ResponseType.OK)
        box = name_dlg.get_content_area()
        box.pack_start(Gtk.Label(label="Preset name:"), False, False, 0)
        entry = Gtk.Entry()
        box.pack_start(entry, False, False, 0)
        name_dlg.show_all()
        response = name_dlg.run()
        if response == Gtk.ResponseType.OK:
            name = entry.get_text().strip()
            if name:
                is_new = name not in self.presets
                self.presets[name] = {
                    'amount': config.get_property('amount'),
                    'adaptive': config.get_property('adaptive'),
                    'oklab': config.get_property('oklab')
                }
                self.save_presets()
                if is_new:
                    self.preset_combo.append_text(name)
                self.set_combo_to_name(self.preset_combo, name)
        name_dlg.destroy()

    def on_delete_preset(self, config, dialog, image, drawable):
        name = self.preset_combo.get_active_text()
        if name and name != 'Default' and name != 'Last':
            del self.presets[name]
            self.save_presets()
            active = self.preset_combo.get_active()
            self.preset_combo.remove(active)
            self.set_combo_to_name(self.preset_combo, 'Last')
            self.on_preset_changed(self.preset_combo, config, dialog, image, drawable)

    def compute_process_rect(self, image, drawable):
        _, off_x, off_y = drawable.get_offsets()
        full_rect = Gegl.Rectangle.new(0, 0, drawable.get_width(), drawable.get_height())
        success, non_empty, img_x1, img_y1, img_x2, img_y2 = Gimp.Selection.bounds(image)
        if non_empty:
            sel_rect_img = Gegl.Rectangle.new(img_x1, img_y1, img_x2 - img_x1, img_y2 - img_y1)
            layer_sel_rect = Gegl.Rectangle.new(img_x1 - off_x, img_y1 - off_y, img_x2 - img_x1, img_y2 - img_y1)
            process_rect = Gegl.Rectangle.intersection(full_rect, layer_sel_rect)
            update_x = img_x1
            update_y = img_y1
            update_w = img_x2 - img_x1
            update_h = img_y2 - img_y1
        else:
            process_rect = full_rect
            update_x = off_x
            update_y = off_y
            update_w = full_rect.width
            update_h = full_rect.height
        return process_rect, update_x, update_y, update_w, update_h

    def get_flat_array_from_buffer(self, buffer, process_rect, babl_format):
        raw = buffer.get(process_rect, 1.0, babl_format, Gegl.AbyssPolicy.CLAMP)
        pixel_array = array('f')
        pixel_array.frombytes(raw)
        return pixel_array

    def get_params(self, drawable):
        buffer = drawable.get_buffer()
        has_alpha = drawable.has_alpha()
        is_gray = drawable.is_gray()
        if is_gray:
            if has_alpha:
                babl_format = "Y~A float"
                channels = 2
            else:
                babl_format = "Y~ float"
                channels = 1
        else:
            if has_alpha:
                babl_format = "R~G~B~A float"
                channels = 4
            else:
                babl_format = "R~G~B~ float"
                channels = 3
        return buffer, has_alpha, babl_format, channels, is_gray

    def apply_to_buffer(self, drawable, flat_array, process_rect, babl_format, update_x, update_y, update_w, update_h, push_undo=True):
        raw_out = flat_array.tobytes()
        shadow_buffer = drawable.get_shadow_buffer()
        shadow_buffer.set(process_rect, babl_format, raw_out)
        shadow_buffer.flush()
        drawable.merge_shadow(push_undo)
        drawable.update(update_x, update_y, update_w, update_h)
        Gimp.displays_flush()

    def process_pixels(self, pixel_array, process_rect, channels, has_alpha, amount, adaptive, oklab, is_gray):
        #profiler = cProfile.Profile()
        #profiler.enable()

        strength = amount / 3.14
        no_contrast = not adaptive
        width = process_rect.width
        height = process_rect.height
        num_pixels = width * height

        is_preview = self.original_pixels is not None

        if is_preview:
            if self.cached_oklab != oklab:
                self.cached_lum = None
                self.cached_oklab = oklab

        if is_preview and self.cached_lum is None:
            alpha_flat = None
            flat_rgb = None
            lum = None
            max_lum = 0.0

            if is_gray:
                if has_alpha:
                    lum = array('f', [pixel_array[p * channels] for p in range(num_pixels)])
                    alpha_flat = array('f', [pixel_array[p * channels + 1] for p in range(num_pixels)])
                else:
                    lum = pixel_array[:]
                max_lum = max(lum) if lum else 0.0
                flat_rgb = None
            else:
                if has_alpha:
                    alpha_flat = array('f', [pixel_array[p * channels + 3] for p in range(num_pixels)])
                    flat_rgb = array('f', [pixel_array[p * channels + c] for p in range(num_pixels) for c in range(3)])
                else:
                    alpha_flat = None
                    flat_rgb = pixel_array[:]
                lum, max_lum = linear_rgb2lum(flat_rgb, width, height)

            self.cached_alpha_flat = alpha_flat
            self.cached_flat_rgb = flat_rgb
            self.cached_lum = lum
            self.cached_original_lum = array('f', lum[:])
            self.cached_contrast, self.cached_contrast_min, self.cached_contrast_max = std_windowed(self.cached_lum, width, height, (7, 7))
            denom = self.cached_contrast_max - self.cached_contrast_min + 1e-10
            self.cached_contrast_norm = array('f', [(c - self.cached_contrast_min) / denom for c in self.cached_contrast])

            self.cached_adaptive = adaptive
            if adaptive:
                self.cached_sqrt_contrast_norm = array('f', [math.sqrt(c) for c in self.cached_contrast_norm])
            else:
                self.cached_sqrt_contrast_norm = None

            if not oklab and not is_gray:
                red = self.cached_flat_rgb[::3]
                green = self.cached_flat_rgb[1::3]
                blue = self.cached_flat_rgb[2::3]

                self.cached_max_val_red = min(2 * max(red), 1.0)
                current = array('f', red)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [red[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                self.cached_corr_minus_one_red = array('f', [correction[k] - 1 for k in range(num_pixels)])

                self.cached_max_val_green = min(2 * max(green), 1.0)
                current = array('f', green)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [green[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                self.cached_corr_minus_one_green = array('f', [correction[k] - 1 for k in range(num_pixels)])

                self.cached_max_val_blue = min(2 * max(blue), 1.0)
                current = array('f', blue)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [blue[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                self.cached_corr_minus_one_blue = array('f', [correction[k] - 1 for k in range(num_pixels)])

                self.cached_max_val = None
                self.cached_corr_minus_one = None
            else:
                self.cached_max_val_red = None
                self.cached_max_val_green = None
                self.cached_max_val_blue = None
                self.cached_corr_minus_one_red = None
                self.cached_corr_minus_one_green = None
                self.cached_corr_minus_one_blue = None

                self.cached_max_val = min(2 * max_lum, 1.0)
                current = array('f', self.cached_lum)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [self.cached_lum[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                self.cached_corr_minus_one = array('f', [correction[k] - 1 for k in range(num_pixels)])

            if not is_gray:
                self.cached_oklab_linear = linearrgb_to_oklab(self.cached_flat_rgb, width, height)

        if is_preview:
            if self.cached_adaptive != adaptive:
                self.cached_adaptive = adaptive
                if adaptive:
                    self.cached_sqrt_contrast_norm = array('f', [math.sqrt(c) for c in self.cached_contrast_norm])
                else:
                    self.cached_sqrt_contrast_norm = None

            original_lum = self.cached_original_lum
            contrast_norm = self.cached_contrast_norm
            sqrt_contrast_norm = self.cached_sqrt_contrast_norm if adaptive else None
            alpha_flat = self.cached_alpha_flat
            flat_rgb = self.cached_flat_rgb
            if not oklab and not is_gray:
                corr_minus_one_red = self.cached_corr_minus_one_red
                corr_minus_one_green = self.cached_corr_minus_one_green
                corr_minus_one_blue = self.cached_corr_minus_one_blue
                max_val_red = self.cached_max_val_red
                max_val_green = self.cached_max_val_green
                max_val_blue = self.cached_max_val_blue
                corr_minus_one = None
                max_val = None
                oklab_linear = None
            else:
                corr_minus_one = self.cached_corr_minus_one
                max_val = self.cached_max_val
                oklab_linear = self.cached_oklab_linear if not is_gray else None
                corr_minus_one_red = None
                corr_minus_one_green = None
                corr_minus_one_blue = None
                max_val_red = None
                max_val_green = None
                max_val_blue = None
        else:
            alpha_flat = None
            flat_rgb = None
            lum = None
            max_lum = 0.0

            if is_gray:
                if has_alpha:
                    lum = array('f', [pixel_array[p * channels] for p in range(num_pixels)])
                    alpha_flat = array('f', [pixel_array[p * channels + 1] for p in range(num_pixels)])
                else:
                    lum = pixel_array[:]
                max_lum = max(lum) if lum else 0.0
                flat_rgb = None
            else:
                if has_alpha:
                    alpha_flat = array('f', [pixel_array[p * channels + 3] for p in range(num_pixels)])
                    flat_rgb = array('f', [pixel_array[p * channels + c] for p in range(num_pixels) for c in range(3)])
                else:
                    alpha_flat = None
                    flat_rgb = pixel_array[:]
                lum, max_lum = linear_rgb2lum(flat_rgb, width, height)

            original_lum = array('f', lum[:])
            contrast, contrast_min, contrast_max = std_windowed(lum, width, height, (7, 7))
            denom = contrast_max - contrast_min + 1e-10
            contrast_norm = array('f', [(c - contrast_min) / denom for c in contrast])
            sqrt_contrast_norm = None
            if adaptive:
                sqrt_contrast_norm = array('f', [math.sqrt(c) for c in contrast_norm])

            if not oklab and not is_gray:
                red = flat_rgb[::3]
                green = flat_rgb[1::3]
                blue = flat_rgb[2::3]

                max_val_red = min(2 * max(red), 1.0)
                current = array('f', red)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [red[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                corr_minus_one_red = array('f', [correction[k] - 1 for k in range(num_pixels)])

                max_val_green = min(2 * max(green), 1.0)
                current = array('f', green)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [green[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                corr_minus_one_green = array('f', [correction[k] - 1 for k in range(num_pixels)])

                max_val_blue = min(2 * max(blue), 1.0)
                current = array('f', blue)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [blue[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                corr_minus_one_blue = array('f', [correction[k] - 1 for k in range(num_pixels)])

                max_val = None
                corr_minus_one = None
                oklab_linear = None
            else:
                max_val_red = None
                max_val_green = None
                max_val_blue = None
                corr_minus_one_red = None
                corr_minus_one_green = None
                corr_minus_one_blue = None

                max_val = min(2 * max_lum, 1.0)
                current = array('f', lum)
                conv = convolve2d(current, psf_flat, width, height)
                relative = array('f', [lum[k] / (conv[k] + 1e-12) for k in range(num_pixels)])
                correction = convolve2d(relative, psf_mirror_flat, width, height)
                corr_minus_one = array('f', [correction[k] - 1 for k in range(num_pixels)])

                oklab_linear = linearrgb_to_oklab(flat_rgb, width, height) if not is_gray else None

        if not oklab and not is_gray:
            if no_contrast:
                damped_red = array('f', [1 + strength * corr_minus_one_red[k] for k in range(num_pixels)])
                damped_green = array('f', [1 + strength * corr_minus_one_green[k] for k in range(num_pixels)])
                damped_blue = array('f', [1 + strength * corr_minus_one_blue[k] for k in range(num_pixels)])
            else:
                damped_red = array('f', [1 + strength * sqrt_contrast_norm[k] * corr_minus_one_red[k] for k in range(num_pixels)])
                damped_green = array('f', [1 + strength * sqrt_contrast_norm[k] * corr_minus_one_green[k] for k in range(num_pixels)])
                damped_blue = array('f', [1 + strength * sqrt_contrast_norm[k] * corr_minus_one_blue[k] for k in range(num_pixels)])

            if is_preview:
                original_red = self.cached_flat_rgb[::3]
                original_green = self.cached_flat_rgb[1::3]
                original_blue = self.cached_flat_rgb[2::3]
            else:
                original_red = flat_rgb[::3]
                original_green = flat_rgb[1::3]
                original_blue = flat_rgb[2::3]

            sharpened_red = array('f', [original_red[p] * (damped_red[p] * damped_red[p] * damped_red[p]) for p in range(num_pixels)])
            for p in range(num_pixels):
                if sharpened_red[p] > max_val_red:
                    sharpened_red[p] = max_val_red

            sharpened_green = array('f', [original_green[p] * (damped_green[p] * damped_green[p] * damped_green[p]) for p in range(num_pixels)])
            for p in range(num_pixels):
                if sharpened_green[p] > max_val_green:
                    sharpened_green[p] = max_val_green

            sharpened_blue = array('f', [original_blue[p] * (damped_blue[p] * damped_blue[p] * damped_blue[p]) for p in range(num_pixels)])
            for p in range(num_pixels):
                if sharpened_blue[p] > max_val_blue:
                    sharpened_blue[p] = max_val_blue

            rgb_sharp = array('f', [0.0] * (num_pixels * 3))
            for p in range(num_pixels):
                idx = p * 3
                rgb_sharp[idx] = sharpened_red[p]
                rgb_sharp[idx + 1] = sharpened_green[p]
                rgb_sharp[idx + 2] = sharpened_blue[p]
        else:
            if no_contrast:
                damped_correction = array('f', [1 + strength * corr_minus_one[k] for k in range(num_pixels)])
            else:
                damped_correction = array('f', [1 + strength * sqrt_contrast_norm[k] * corr_minus_one[k] for k in range(num_pixels)])

            ratio = damped_correction

            if is_gray:
                sharpened_lum = array('f', [original_lum[p] * (ratio[p] * ratio[p] * ratio[p]) if original_lum[p] * (ratio[p] * ratio[p] * ratio[p]) < max_val else max_val for p in range(num_pixels)])
                if alpha_flat is not None:
                    for p in range(num_pixels):
                        if alpha_flat[p] == 0:
                            sharpened_lum[p] = 0.0
                    out_flat = array('f', [0.0] * (num_pixels * channels))
                    for p in range(num_pixels):
                        idx = p * channels
                        out_flat[idx] = sharpened_lum[p]
                        out_flat[idx + 1] = alpha_flat[p]
                else:
                    out_flat = sharpened_lum
            else:
                rgb_sharp = oklab_to_linearrgb(oklab_linear, ratio, width, height, max_val)

        if not is_gray:
            if alpha_flat is not None:
                for p in range(num_pixels):
                    if alpha_flat[p] == 0:
                        idx = p * 3
                        rgb_sharp[idx] = 0.0
                        rgb_sharp[idx + 1] = 0.0
                        rgb_sharp[idx + 2] = 0.0
                out_flat = array('f', [0.0] * (num_pixels * channels))
                for p in range(num_pixels):
                    idx3 = p * 3
                    idx4 = p * channels
                    out_flat[idx4] = rgb_sharp[idx3]
                    out_flat[idx4 + 1] = rgb_sharp[idx3 + 1]
                    out_flat[idx4 + 2] = rgb_sharp[idx3 + 2]
                    out_flat[idx4 + 3] = alpha_flat[p]
            else:
                out_flat = rgb_sharp

        #profiler.disable()
        #s = io.StringIO()
        #stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        #stats.print_stats(20)
        #Gimp.message("Profile Stats:\n" + s.getvalue())

        return out_flat

    def debounce_update(self, dialog, image, drawable, config):
        if self.timer_id is not None:
            GLib.source_remove(self.timer_id)
            self.timer_id = None
        def delayed_update():
            self.update_preview(None, dialog, image, drawable, config)
            self.timer_id = None
            return False
        self.timer_id = GLib.timeout_add(500, delayed_update)

    def update_preview(self, widget, dialog, image, drawable, config):
        preview = config.get_property('preview')
        amount = config.get_property('amount')
        adaptive = config.get_property('adaptive')
        oklab = config.get_property('oklab')

        process_rect, update_x, update_y, update_w, update_h = self.compute_process_rect(image, drawable)
        buffer, has_alpha, babl_format, channels, is_gray = self.get_params(drawable)

        if preview:
            if self.original_pixels is None:
                self.process_rect = process_rect
                self.update_x = update_x
                self.update_y = update_y
                self.update_w = update_w
                self.update_h = update_h
                self.has_alpha = has_alpha
                self.is_gray = is_gray
                self.babl_format = babl_format
                self.channels = channels
                self.original_pixels = self.get_flat_array_from_buffer(buffer, process_rect, babl_format)
            out_flat = self.process_pixels(self.original_pixels, self.process_rect, self.channels, self.has_alpha, amount, adaptive, oklab, self.is_gray)
            self.apply_to_buffer(drawable, out_flat, self.process_rect, self.babl_format, self.update_x, self.update_y, self.update_w, self.update_h, push_undo=False)
        else:
            if self.original_pixels is not None:
                self.apply_to_buffer(drawable, self.original_pixels, self.process_rect, self.babl_format, self.update_x, self.update_y, self.update_w, self.update_h, push_undo=False)
                self.original_pixels = None
                self.process_rect = None
                self.cached_lum = None
                self.cached_original_lum = None
                self.cached_contrast_norm = None
                self.cached_sqrt_contrast_norm = None
                self.cached_correction = None
                self.cached_oklab_linear = None
                self.cached_max_val = None
                self.cached_conv = None
                self.cached_relative = None
                self.cached_adaptive = None
                self.cached_corr_minus_one = None
                self.cached_contrast = None
                self.cached_contrast_min = None
                self.cached_contrast_max = None
                self.cached_oklab = None
                self.cached_flat_rgb = None
                self.cached_alpha_flat = None
                self.cached_corr_minus_one_red = None
                self.cached_corr_minus_one_green = None
                self.cached_corr_minus_one_blue = None
                self.cached_max_val_red = None
                self.cached_max_val_green = None
                self.cached_max_val_blue = None

    def adaptive_deconvolution(self, procedure, run_mode, image, drawables, config, run_data):
        n_drawables = len(drawables)
        if n_drawables != 1:
            error = 'This plug-in works only on one drawable.'
            return procedure.new_return_values(Gimp.PDBStatusType.CALL_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, error))

        drawable = drawables[0]
        if not drawable.is_valid():
            error = 'Invalid drawable.'
            return procedure.new_return_values(Gimp.PDBStatusType.CALL_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, error))

        if run_mode == Gimp.RunMode.INTERACTIVE:
            try:
                image.undo_group_start()

                # Get image type early
                _, _, _, _, is_gray = self.get_params(drawable)
                if is_gray:
                    config.set_property('oklab', False)

                dialog = GimpUi.ProcedureDialog(procedure=procedure, config=config)

                content_area = dialog.get_content_area()

                # Load presets
                self.presets = self.load_presets()
                if 'Default' not in self.presets:
                    self.presets['Default'] = {'amount': 10.0, 'adaptive': True, 'oklab': False}
                    self.save_presets()
                if 'Last' not in self.presets:
                    self.presets['Last'] = self.presets['Default'].copy()
                    self.save_presets()

                # Preset box
                preset_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
                preset_label = Gtk.Label(label="Preset:")
                preset_box.pack_start(preset_label, False, False, 0)
                self.preset_combo = Gtk.ComboBoxText()
                preset_names = sorted(self.presets.keys())
                for name in preset_names:
                    self.preset_combo.append_text(name)
                preset_box.pack_start(self.preset_combo, True, True, 0)
                save_button = Gtk.Button(label="Save...")
                save_button.connect('clicked', lambda b: self.on_save_preset(dialog, config, image, drawable))
                delete_button = Gtk.Button(label="Delete")
                delete_button.connect('clicked', lambda b: self.on_delete_preset(config, dialog, image, drawable))
                preset_box.pack_start(save_button, False, False, 0)
                preset_box.pack_start(delete_button, False, False, 0)
                content_area.pack_start(preset_box, False, False, 0)

                # Set initial preset
                self.set_combo_to_name(self.preset_combo, 'Last')
                self.on_preset_changed(self.preset_combo, config, dialog, image, drawable)

                # Connect preset changed
                self.preset_combo.connect('changed', lambda c: self.on_preset_changed(c, config, dialog, image, drawable))

                # Create spin scale for amount
                amount_spin = dialog.get_spin_scale('amount', 1.0)
                content_area.pack_start(amount_spin, False, False, 0)

                # Adaptive checkbox
                adaptive_check = Gtk.CheckButton(label='A_daptive', use_underline=True)
                adaptive_check.set_active(config.get_property('adaptive'))
                adaptive_check.set_tooltip_text('Use adaptive contrast')
                content_area.pack_start(adaptive_check, False, False, 0)

                # OKLab checkbox
                oklab_check = Gtk.CheckButton(label='_OKLab', use_underline=True)
                oklab_check.set_active(config.get_property('oklab'))
                oklab_check.set_tooltip_text('Process luminance using OKLab instead of processing RGB channels separately')
                if is_gray:
                    oklab_check.set_sensitive(False)
                content_area.pack_start(oklab_check, False, False, 0)

                # Preview checkbox
                preview_check = Gtk.CheckButton(label='_Preview', use_underline=True)
                preview_check.set_active(config.get_property('preview'))
                preview_check.set_tooltip_text('Apply delayed preview on canvas')
                content_area.pack_start(preview_check, False, False, 0)

                dialog.show_all()

                # Customize amount slider
                amount_spin.set_digits(0)
                amount_spin.set_increments(1.0, 10.0)

                # Connect signals to debounce update
                immediate_update = lambda widget: self.update_preview(widget, dialog, image, drawable, config)
                debounce_func = lambda widget: self.debounce_update(dialog, image, drawable, config)

                amount_spin.connect('value-changed', debounce_func)
                adaptive_check.connect('toggled', lambda w: config.set_property('adaptive', w.get_active()) or debounce_func(w))
                oklab_check.connect('toggled', lambda w: config.set_property('oklab', w.get_active()) or debounce_func(w))

                # Sync preview checkbox with config
                def on_preview_toggled(widget):
                    config.set_property('preview', widget.get_active())
                    immediate_update(widget)
                preview_check.connect('toggled', on_preview_toggled)

                # Initial preview
                self.debounce_update(dialog, image, drawable, config)

                is_ok_pressed = dialog.run()

                # Cancel timer if running
                if self.timer_id is not None:
                    GLib.source_remove(self.timer_id)
                    self.timer_id = None

                if is_ok_pressed:
                    self.presets['Last'] = {
                        'amount': config.get_property('amount'),
                        'adaptive': config.get_property('adaptive'),
                        'oklab': config.get_property('oklab')
                    }
                    self.save_presets()

                    # If preview was enabled, revert to original (no undo) then re-apply final with undo
                    if config.get_property('preview') and self.original_pixels is not None:
                        self.apply_to_buffer(drawable, self.original_pixels, self.process_rect, self.babl_format,
                                             self.update_x, self.update_y, self.update_w, self.update_h, push_undo=False)
                        self.original_pixels = None
                        self.cached_lum = None
                        self.cached_original_lum = None
                        self.cached_contrast_norm = None
                        self.cached_sqrt_contrast_norm = None
                        self.cached_correction = None
                        self.cached_oklab_linear = None
                        self.cached_max_val = None
                        self.cached_conv = None
                        self.cached_relative = None
                        self.cached_adaptive = None
                        self.cached_corr_minus_one = None
                        self.cached_contrast = None
                        self.cached_contrast_min = None
                        self.cached_contrast_max = None
                        self.cached_oklab = None
                        self.cached_flat_rgb = None
                        self.cached_alpha_flat = None
                        self.cached_corr_minus_one_red = None
                        self.cached_corr_minus_one_green = None
                        self.cached_corr_minus_one_blue = None
                        self.cached_max_val_red = None
                        self.cached_max_val_green = None
                        self.cached_max_val_blue = None

                    # Apply final result with undo enabled
                    process_rect, update_x, update_y, update_w, update_h = self.compute_process_rect(image, drawable)
                    buffer, has_alpha, babl_format, channels, is_gray = self.get_params(drawable)
                    pixel_array = self.get_flat_array_from_buffer(buffer, process_rect, babl_format)
                    out_flat = self.process_pixels(pixel_array, process_rect, channels, has_alpha,
                                                   config.get_property('amount'), config.get_property('adaptive'), config.get_property('oklab'), is_gray)
                    self.apply_to_buffer(drawable, out_flat, process_rect, babl_format,
                                         update_x, update_y, update_w, update_h, push_undo=True)
                else:
                    # Cancel: revert if preview was used (no undo)
                    if config.get_property('preview') and self.original_pixels is not None:
                        self.apply_to_buffer(drawable, self.original_pixels, self.process_rect, self.babl_format,
                                             self.update_x, self.update_y, self.update_w, self.update_h, push_undo=False)
                        self.original_pixels = None
                        self.process_rect = None
                        self.cached_lum = None
                        self.cached_original_lum = None
                        self.cached_contrast_norm = None
                        self.cached_sqrt_contrast_norm = None
                        self.cached_correction = None
                        self.cached_oklab_linear = None
                        self.cached_max_val = None
                        self.cached_conv = None
                        self.cached_relative = None
                        self.cached_adaptive = None
                        self.cached_corr_minus_one = None
                        self.cached_contrast = None
                        self.cached_contrast_min = None
                        self.cached_contrast_max = None
                        self.cached_oklab = None
                        self.cached_flat_rgb = None
                        self.cached_alpha_flat = None
                        self.cached_corr_minus_one_red = None
                        self.cached_corr_minus_one_green = None
                        self.cached_corr_minus_one_blue = None
                        self.cached_max_val_red = None
                        self.cached_max_val_green = None
                        self.cached_max_val_blue = None

                image.undo_group_end()
                Gimp.displays_flush()
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS if is_ok_pressed else Gimp.PDBStatusType.CANCEL)

            except Exception as e:
                # Gimp.message("Error in interactive mode:\n" + traceback.format_exc())
                if self.original_pixels is not None:
                    self.apply_to_buffer(drawable, self.original_pixels, self.process_rect, self.babl_format,
                                         self.update_x, self.update_y, self.update_w, self.update_h, push_undo=False)
                image.undo_group_end()
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, str(e)))

        # Non-interactive mode
        amount = config.get_property('amount')
        adaptive = config.get_property('adaptive')
        oklab = config.get_property('oklab')

        image.undo_group_start()

        try:
            process_rect, update_x, update_y, update_w, update_h = self.compute_process_rect(image, drawable)
            buffer, has_alpha, babl_format, channels, is_gray = self.get_params(drawable)
            if is_gray:
                oklab = False
            pixel_array = self.get_flat_array_from_buffer(buffer, process_rect, babl_format)
            out_flat = self.process_pixels(pixel_array, process_rect, channels, has_alpha, amount, adaptive, oklab, is_gray)
            self.apply_to_buffer(drawable, out_flat, process_rect, babl_format,
                                 update_x, update_y, update_w, update_h, push_undo=True)

        except Exception as e:
            # Gimp.message("Error in processing:\n" + traceback.format_exc())
            image.undo_group_end()
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, str(e)))

        image.undo_group_end()
        Gimp.displays_flush()
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS)

Gimp.main(AdaptiveDeconvolution.__gtype__, sys.argv)
