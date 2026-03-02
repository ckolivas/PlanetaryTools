#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Wavelet Denoise plugin for GIMP v3 by Con Kolivas <kernel@kolivas.org>

import sys
import re
import json
import os
import gi
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')
gi.require_version('Gtk', '3.0')
from gi.repository import Gimp
from gi.repository import GimpUi
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gtk
import traceback

class WaveletDenoise(Gimp.PlugIn):
    def __init__(self):
        self.preview_layer = None
        self.wavelet_group = None
        self.timer_id = None
        self.presets = None
        self.preset_combo = None

    def do_set_i18n(self, name):
        return False

    def do_query_procedures(self):
        return ['plug-in-wavelet-denoise']

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.wavelet_denoise, None)
        procedure.set_menu_label('Wavelet _Denoise...')
        procedure.set_documentation(
            'Denoise image using wavelet decomposition with independent controls for fine, medium, and coarse detail scales.',
            'Performs wavelet decompose internally and applies Gaussian blur to each scale layer. '
            'Blurring a wavelet scale suppresses noise at that frequency while preserving structure at other scales.',
            name)
        procedure.add_menu_path('<Image>/Filters/Enhance')
        procedure.set_attribution('Con Kolivas', 'Con Kolivas', '2026')
        procedure.set_image_types('RGB*, GRAY*')

        procedure.add_double_argument('fine',    '_Fine',    'Blur radius for fine detail (scale 1)',   0.0, 20.0, 3.0, GObject.ParamFlags.READWRITE)
        procedure.add_double_argument('medium',  '_Medium',  'Blur radius for medium detail (scale 2)', 0.0, 20.0, 1.0, GObject.ParamFlags.READWRITE)
        procedure.add_double_argument('coarse',  'Coars_e',  'Blur radius for coarse detail (scale 3)', 0.0, 20.0, 0.0, GObject.ParamFlags.READWRITE)
        procedure.add_boolean_argument('preview', '_Preview', 'Apply preview on canvas',               True,       GObject.ParamFlags.READWRITE)

        return procedure

    def load_presets(self):
        preset_file = os.path.expanduser('~/.config/GIMP/3.0/wavelet-denoise-presets.json')
        if os.path.exists(preset_file):
            with open(preset_file, 'r') as f:
                return json.load(f)
        return {}

    def save_presets(self):
        preset_file = os.path.expanduser('~/.config/GIMP/3.0/wavelet-denoise-presets.json')
        os.makedirs(os.path.dirname(preset_file), exist_ok=True)
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
            config.set_property('fine',   vals['fine'])
            config.set_property('medium', vals['medium'])
            config.set_property('coarse', vals['coarse'])

    def on_save_preset(self, dialog, config, image, drawable):
        name_dlg = Gtk.Dialog(title='Save Preset As', transient_for=dialog, flags=0)
        name_dlg.add_buttons('Cancel', Gtk.ResponseType.CANCEL, 'Save', Gtk.ResponseType.OK)
        box = name_dlg.get_content_area()
        box.pack_start(Gtk.Label(label='Preset name:'), False, False, 0)
        entry = Gtk.Entry()
        box.pack_start(entry, False, False, 0)
        name_dlg.show_all()
        response = name_dlg.run()
        if response == Gtk.ResponseType.OK:
            name = entry.get_text().strip()
            if name:
                is_new = name not in self.presets
                self.presets[name] = {
                    'fine':   config.get_property('fine'),
                    'medium': config.get_property('medium'),
                    'coarse': config.get_property('coarse'),
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

    def find_wavelet_group(self, items):
        for item in items:
            if item.is_group():
                children = item.get_children()
                if len(children) == 4 and \
                   re.match(r'^Scale 1( #\d+)?$', children[0].get_name()) and \
                   re.match(r'^Scale 2( #\d+)?$', children[1].get_name()) and \
                   re.match(r'^Scale 3( #\d+)?$', children[2].get_name()) and \
                   re.match(r'^Residual( #\d+)?$', children[3].get_name()):
                    return item
                found = self.find_wavelet_group(children)
                if found:
                    return found
        return None

    def apply_gaussian_blur(self, drawable, std_dev):
        """Blur a wavelet scale layer by std_dev pixels.
        Blurring the scale suppresses noise at that frequency band.
        A std_dev of 0 is a no-op."""
        if std_dev <= 0.0:
            return
        filter_ = Gimp.DrawableFilter.new(drawable, 'gegl:gaussian-blur', 'Gaussian Blur')
        config = filter_.get_config()
        config.set_property('std-dev-x', std_dev)
        config.set_property('std-dev-y', std_dev)
        filter_.update()
        drawable.append_filter(filter_)
        drawable.merge_filters()

    def get_merged_from_group(self, image, group):
        save_vis = []
        all_layers = image.get_layers()
        for layer in all_layers:
            save_vis.append(layer.get_visible())
            layer.set_visible(False)
        group.set_visible(True)
        for child in group.get_children():
            child.set_visible(True)
        merged = Gimp.Layer.new_from_visible(image, image, 'merged')
        for i, layer in enumerate(all_layers):
            layer.set_visible(save_vis[i])
        return merged

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
        amounts = [config.get_property('fine'), config.get_property('medium'), config.get_property('coarse')]

        if not preview:
            if self.preview_layer is not None:
                image.remove_layer(self.preview_layer)
                self.preview_layer = None
            drawable.set_visible(True)
            Gimp.displays_flush()
            return

        try:
            if self.preview_layer is not None:
                image.remove_layer(self.preview_layer)
                self.preview_layer = None

            group_copy = self.wavelet_group.copy()
            image.insert_layer(group_copy, None, 0)
            children = group_copy.get_children()
            scales = children[0:3]
            group_copy.set_visible(True)
            for child in children:
                child.set_visible(True)

            for i in range(3):
                self.apply_gaussian_blur(scales[i], amounts[i])

            merged = self.get_merged_from_group(image, group_copy)
            image.insert_layer(merged, None, 0)
            image.remove_layer(group_copy)

            _, offset_x, offset_y = drawable.get_offsets()
            merged.set_offsets(offset_x, offset_y)
            merged.set_mode(drawable.get_mode())
            merged.set_name(drawable.get_name() + ' (preview)')
            parent = drawable.get_parent()
            pos = image.get_item_position(drawable)
            image.reorder_item(merged, parent, pos)
            drawable.set_visible(False)
            merged.set_visible(True)
            self.preview_layer = merged

            Gimp.displays_flush()

        except Exception:
            Gimp.message('Error in update_preview:\n' + traceback.format_exc())

    def wavelet_denoise(self, procedure, run_mode, image, drawables, config, run_data):
        if len(drawables) != 1:
            return procedure.new_return_values(Gimp.PDBStatusType.CALL_ERROR,
                GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                    'This plug-in works only on one drawable.'))

        drawable = drawables[0]
        if not drawable.is_valid():
            return procedure.new_return_values(Gimp.PDBStatusType.CALL_ERROR,
                GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                    'Invalid drawable.'))

        if run_mode == Gimp.RunMode.INTERACTIVE:
            try:
                image.undo_group_start()

                # Decompose a temporary copy so the original is untouched during dialog
                temp = drawable.copy()
                parent = drawable.get_parent()
                pos = image.get_item_position(drawable)
                image.insert_layer(temp, parent, pos)
                temp.set_visible(True)

                pdb = Gimp.get_pdb()
                decompose_proc = pdb.lookup_procedure('plug-in-wavelet-decompose')
                if decompose_proc is None:
                    image.remove_layer(temp)
                    return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                        GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                            'Wavelet decompose plug-in not found. Please install it.'))

                decompose_config = decompose_proc.create_config()
                decompose_config.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
                decompose_config.set_property('image', image)
                decompose_config.set_core_object_array('drawables', [temp])
                decompose_config.set_property('scales', 3)
                decompose_config.set_property('create-group', True)
                decompose_config.set_property('create-masks', False)
                result = decompose_proc.run(decompose_config)

                if result.index(0) != Gimp.PDBStatusType.SUCCESS:
                    image.remove_layer(temp)
                    return result

                self.wavelet_group = self.find_wavelet_group(image.get_layers())
                if self.wavelet_group is None:
                    image.remove_layer(temp)
                    return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                        GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                            'Wavelet decompose group not found.'))

                self.wavelet_group.set_visible(False)
                for child in self.wavelet_group.get_children():
                    child.set_visible(False)
                image.remove_layer(temp)

                dialog = GimpUi.ProcedureDialog(procedure=procedure, config=config)
                content_area = dialog.get_content_area()

                # Presets
                self.presets = self.load_presets()
                if 'Default' not in self.presets:
                    self.presets['Default'] = {'fine': 3.0, 'medium': 1.0, 'coarse': 0.0}
                    self.save_presets()
                if 'Last' not in self.presets:
                    self.presets['Last'] = self.presets['Default'].copy()
                    self.save_presets()

                preset_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
                preset_box.pack_start(Gtk.Label(label='Preset:'), False, False, 0)
                self.preset_combo = Gtk.ComboBoxText()
                for name in sorted(self.presets.keys()):
                    self.preset_combo.append_text(name)
                preset_box.pack_start(self.preset_combo, True, True, 0)
                save_btn = Gtk.Button(label='Save...')
                save_btn.connect('clicked', lambda b: self.on_save_preset(dialog, config, image, drawable))
                del_btn = Gtk.Button(label='Delete')
                del_btn.connect('clicked', lambda b: self.on_delete_preset(config, dialog, image, drawable))
                preset_box.pack_start(save_btn, False, False, 0)
                preset_box.pack_start(del_btn,  False, False, 0)
                content_area.pack_start(preset_box, False, False, 0)

                self.set_combo_to_name(self.preset_combo, 'Last')
                self.on_preset_changed(self.preset_combo, config, dialog, image, drawable)
                self.preset_combo.connect('changed',
                    lambda c: self.on_preset_changed(c, config, dialog, image, drawable))

                fine_spin = dialog.get_spin_scale('fine', 1.0)
                fine_spin.set_tooltip_text('Gaussian blur radius for fine detail scale (pixels)')
                content_area.pack_start(fine_spin, False, False, 0)

                medium_spin = dialog.get_spin_scale('medium', 1.0)
                medium_spin.set_tooltip_text('Gaussian blur radius for medium detail scale (pixels)')
                content_area.pack_start(medium_spin, False, False, 0)

                coarse_spin = dialog.get_spin_scale('coarse', 1.0)
                coarse_spin.set_tooltip_text('Gaussian blur radius for coarse detail scale (pixels)')
                content_area.pack_start(coarse_spin, False, False, 0)

                preview_check = Gtk.CheckButton(label='_Preview', use_underline=True)
                preview_check.set_active(config.get_property('preview'))
                preview_check.set_tooltip_text('Apply delayed preview on canvas')
                content_area.pack_start(preview_check, False, False, 0)

                dialog.show_all()

                for spin in [fine_spin, medium_spin, coarse_spin]:
                    spin.set_digits(1)
                    spin.set_increments(0.1, 1.0)

                debounce_update  = lambda w: self.debounce_update(dialog, image, drawable, config)
                immediate_update = lambda w: self.update_preview(w, dialog, image, drawable, config)

                fine_spin.connect('value-changed',   debounce_update)
                medium_spin.connect('value-changed',  debounce_update)
                coarse_spin.connect('value-changed',  debounce_update)

                def on_preview_toggled(widget):
                    config.set_property('preview', widget.get_active())
                    immediate_update(widget)
                preview_check.connect('toggled', on_preview_toggled)

                self.update_preview(None, dialog, image, drawable, config)

                is_ok_pressed = dialog.run()

                if self.timer_id is not None:
                    GLib.source_remove(self.timer_id)
                    self.timer_id = None

                if self.preview_layer is not None:
                    image.remove_layer(self.preview_layer)
                    self.preview_layer = None
                drawable.set_visible(True)
                Gimp.displays_flush()
                dialog.destroy()

                if not is_ok_pressed:
                    if self.wavelet_group is not None:
                        image.remove_layer(self.wavelet_group)
                    image.undo_group_end()
                    return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)

                self.presets['Last'] = {
                    'fine':   config.get_property('fine'),
                    'medium': config.get_property('medium'),
                    'coarse': config.get_property('coarse'),
                }
                self.save_presets()

                amounts  = [config.get_property('fine'), config.get_property('medium'), config.get_property('coarse')]
                children = self.wavelet_group.get_children()
                scales   = children[0:3]
                self.wavelet_group.set_visible(True)
                for child in children:
                    child.set_visible(True)
                for i in range(3):
                    self.apply_gaussian_blur(scales[i], amounts[i])

                merged_layer = self.get_merged_from_group(image, self.wavelet_group)
                image.remove_layer(self.wavelet_group)

                _, offset_x, offset_y = drawable.get_offsets()
                merged_layer.set_offsets(offset_x, offset_y)
                merged_layer.set_mode(drawable.get_mode())
                merged_layer.set_visible(drawable.get_visible())
                merged_layer.set_name(drawable.get_name())
                original_position = image.get_item_position(drawable)
                original_parent   = drawable.get_parent()
                image.remove_layer(drawable)
                image.insert_layer(merged_layer, original_parent, original_position)

                image.undo_group_end()
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

            except Exception:
                Gimp.message('Error in interactive mode:\n' + traceback.format_exc())
                if self.timer_id is not None:
                    GLib.source_remove(self.timer_id)
                    self.timer_id = None
                if self.preview_layer is not None:
                    image.remove_layer(self.preview_layer)
                if self.wavelet_group is not None:
                    image.remove_layer(self.wavelet_group)
                drawable.set_visible(True)
                Gimp.displays_flush()
                image.undo_group_end()
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                    GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                        traceback.format_exc()))

        # Non-interactive mode
        amounts = [config.get_property('fine'), config.get_property('medium'), config.get_property('coarse')]

        image.undo_group_start()
        try:
            original_position = image.get_item_position(drawable)
            original_parent   = drawable.get_parent()

            temp = drawable.copy()
            image.insert_layer(temp, original_parent, original_position)
            temp.set_visible(True)

            pdb = Gimp.get_pdb()
            decompose_proc = pdb.lookup_procedure('plug-in-wavelet-decompose')
            decompose_config = decompose_proc.create_config()
            decompose_config.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
            decompose_config.set_property('image', image)
            decompose_config.set_core_object_array('drawables', [temp])
            decompose_config.set_property('scales', 3)
            decompose_config.set_property('create-group', True)
            decompose_config.set_property('create-masks', False)
            result = decompose_proc.run(decompose_config)

            if result.index(0) != Gimp.PDBStatusType.SUCCESS:
                image.remove_layer(temp)
                return result

            group = self.find_wavelet_group(image.get_layers())
            if group is None:
                image.remove_layer(temp)
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                    GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                        'Wavelet decompose group not found.'))

            children = group.get_children()
            if len(children) != 4:
                image.remove_layer(temp)
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                    GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0,
                        'Unexpected number of layers in decomposition.'))

            scales = children[0:3]
            group.set_visible(True)
            for child in children:
                child.set_visible(True)
            for i in range(3):
                self.apply_gaussian_blur(scales[i], amounts[i])

            merged_layer = self.get_merged_from_group(image, group)
            image.remove_layer(group)
            image.remove_layer(temp)

            _, offset_x, offset_y = drawable.get_offsets()
            merged_layer.set_offsets(offset_x, offset_y)
            merged_layer.set_mode(drawable.get_mode())
            merged_layer.set_visible(drawable.get_visible())
            merged_layer.set_name(drawable.get_name())
            image.remove_layer(drawable)
            image.insert_layer(merged_layer, original_parent, original_position)

        except Exception:
            Gimp.message('Error in processing:\n' + traceback.format_exc())
            raise
        finally:
            image.undo_group_end()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

Gimp.main(WaveletDenoise.__gtype__, sys.argv)
