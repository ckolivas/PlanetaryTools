#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Wavelet Sharpen plugin for GIMP v3 by Con Kolivas <kernel@kolivas.org>

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

class WaveletSharpen(Gimp.PlugIn):
    def __init__(self):
        self.preview_layer = None
        self.wavelet_group = None
        self.timer_id = None
        self.presets = None
        self.preset_combo = None

    def do_set_i18n(self, name):
        return False

    def do_query_procedures(self):
        return ['plug-in-wavelet-sharpen']

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.wavelet_sharpen, None)
        procedure.set_menu_label('Wavelet _Sharpen...')
        procedure.set_documentation('Sharpen image using wavelet decomposition with independent controls for fine, medium, and coarse details.',
                                    'Performs wavelet decompose internally and applies unsharp mask to each scale.',
                                    name)
        procedure.add_menu_path('<Image>/Filters/Enhance')
        procedure.set_attribution('Con Kolivas', 'Con Kolivas', '2026')
        procedure.set_image_types('RGB*, GRAY*')

        # Add arguments
        procedure.add_double_argument('fine', '_Fine', 'Amount for fine details (scale 1)', 0.0, 300.0, 16.0, GObject.ParamFlags.READWRITE)
        procedure.add_double_argument('medium', '_Medium', 'Amount for medium details (scale 2)', 0.0, 300.0, 8.0, GObject.ParamFlags.READWRITE)
        procedure.add_double_argument('coarse', 'Coars_e', 'Amount for coarse details (scale 3)', 0.0, 300.0, 1.0, GObject.ParamFlags.READWRITE)
        procedure.add_boolean_argument('preview', '_Preview', 'Apply preview on canvas', True, GObject.ParamFlags.READWRITE)

        return procedure

    def load_presets(self):
        preset_file = os.path.expanduser('~/.config/GIMP/3.0/wavelet-sharpen-presets.json')
        if os.path.exists(preset_file):
            with open(preset_file, 'r') as f:
                return json.load(f)
        return {}

    def save_presets(self):
        preset_file = os.path.expanduser('~/.config/GIMP/3.0/wavelet-sharpen-presets.json')
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
            config.set_property('fine', vals['fine'])
            config.set_property('medium', vals['medium'])
            config.set_property('coarse', vals['coarse'])

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
                    'fine': config.get_property('fine'),
                    'medium': config.get_property('medium'),
                    'coarse': config.get_property('coarse')
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

    def apply_unsharp_mask(self, drawable, std_dev, scale, threshold):
        filter_ = Gimp.DrawableFilter.new(drawable, "gegl:unsharp-mask", "Unsharp Mask")
        config = filter_.get_config()
        config.set_property("std-dev", std_dev)
        config.set_property("scale", scale)
        config.set_property("threshold", threshold)
        filter_.update()
        drawable.append_filter(filter_)
        drawable.merge_filters()

    def get_merged_from_group(self, image, group):
        # Save visibilities
        save_vis = []
        all_layers = image.get_layers()
        for layer in all_layers:
            save_vis.append(layer.get_visible())
            layer.set_visible(False)

        # Show group and children
        group.set_visible(True)
        for child in group.get_children():
            child.set_visible(True)

        # Create new layer from visible
        merged = Gimp.Layer.new_from_visible(image, image, "merged")

        # Restore visibilities
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

            # Duplicate the wavelet group
            group_copy = self.wavelet_group.copy()
            image.insert_layer(group_copy, None, 0)

            # Get duplicated children
            children = group_copy.get_children()
            scales = children[0:3]

            # Set visible for operations
            group_copy.set_visible(True)
            for child in children:
                child.set_visible(True)

            # Apply unsharp mask to duplicated scales
            for i in range(3):
                self.apply_unsharp_mask(scales[i], 16.0, amounts[i], 0.0)

            # Merge using new method
            merged = self.get_merged_from_group(image, group_copy)

            # Insert the merged layer
            image.insert_layer(merged, None, 0)

            # Remove the group copy
            image.remove_layer(group_copy)

            # Set properties and position above original
            _, offset_x, offset_y = drawable.get_offsets()
            merged.set_offsets(offset_x, offset_y)
            merged.set_mode(drawable.get_mode())
            merged.set_name(drawable.get_name() + " (preview)")
            parent = drawable.get_parent()
            pos = image.get_item_position(drawable)
            image.reorder_item(merged, parent, pos)
            drawable.set_visible(False)
            merged.set_visible(True)
            self.preview_layer = merged

            Gimp.displays_flush()

        except Exception as e:
            Gimp.message("Error in update_preview:\n" + traceback.format_exc())

    def wavelet_sharpen(self, procedure, run_mode, image, drawables, config, run_data):
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

                # Create temp duplicate for decomposition
                temp = drawable.copy()
                parent = drawable.get_parent()
                pos = image.get_item_position(drawable)
                image.insert_layer(temp, parent, pos)
                temp.set_visible(True)

                # Perform decomposition
                pdb = Gimp.get_pdb()
                decompose_proc = pdb.lookup_procedure('plug-in-wavelet-decompose')
                if decompose_proc is None:
                    Gimp.message("Wavelet decompose plug-in not found. Please install it.")
                    image.remove_layer(temp)
                    return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, "Wavelet decompose plug-in not found."))

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

                # Find the wavelet layer group recursively
                self.wavelet_group = self.find_wavelet_group(image.get_layers())

                if self.wavelet_group is None:
                    image.remove_layer(temp)
                    return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, "Wavelet decompose group not found."))

                # Hide the group and children
                self.wavelet_group.set_visible(False)
                for child in self.wavelet_group.get_children():
                    child.set_visible(False)

                # Remove temp
                image.remove_layer(temp)

                dialog = GimpUi.ProcedureDialog(procedure=procedure, config=config)

                content_area = dialog.get_content_area()

                # Load presets
                self.presets = self.load_presets()
                if 'Default' not in self.presets:
                    self.presets['Default'] = {'fine': 16.0, 'medium': 8.0, 'coarse': 1.0}
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

                # Create spin scales
                fine_spin = dialog.get_spin_scale('fine', 1.0)
                fine_spin.set_tooltip_text('Amount for fine details (scale 1)')
                content_area.pack_start(fine_spin, False, False, 0)

                medium_spin = dialog.get_spin_scale('medium', 1.0)
                medium_spin.set_tooltip_text('Amount for medium details (scale 2)')
                content_area.pack_start(medium_spin, False, False, 0)

                coarse_spin = dialog.get_spin_scale('coarse', 1.0)
                coarse_spin.set_tooltip_text('Amount for coarse details (scale 3)')
                content_area.pack_start(coarse_spin, False, False, 0)

                # Create preview checkbox
                preview_check = Gtk.CheckButton(label='_Preview', use_underline=True)
                preview_check.set_active(config.get_property('preview'))
                preview_check.set_tooltip_text('Apply delayed preview on canvas')
                content_area.pack_start(preview_check, False, False, 0)

                dialog.show_all()

                # Customize sliders
                for spin in [fine_spin, medium_spin, coarse_spin]:
                    spin.set_digits(1)
                    spin.set_increments(0.1, 10.0)

                # Connect signals for live preview
                immediate_update = lambda widget: self.update_preview(widget, dialog, image, drawable, config)
                debounce_update = lambda widget: self.debounce_update(dialog, image, drawable, config)

                fine_spin.connect('value-changed', debounce_update)
                medium_spin.connect('value-changed', debounce_update)
                coarse_spin.connect('value-changed', debounce_update)
                preview_check.connect('toggled', lambda widget: immediate_update(widget))

                # Sync preview checkbox with config
                def on_preview_toggled(widget):
                    config.set_property('preview', widget.get_active())
                    immediate_update(widget)
                preview_check.connect('toggled', on_preview_toggled)

                # Initial preview if enabled
                self.update_preview(None, dialog, image, drawable, config)

                is_ok_pressed = dialog.run()

                # Cancel timer if running
                if self.timer_id is not None:
                    GLib.source_remove(self.timer_id)
                    self.timer_id = None

                # Clean up preview
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
                    'fine': config.get_property('fine'),
                    'medium': config.get_property('medium'),
                    'coarse': config.get_property('coarse')
                }
                self.save_presets()

                # Apply final amounts to original scales
                children = self.wavelet_group.get_children()
                scales = children[0:3]
                amounts = [config.get_property('fine'), config.get_property('medium'), config.get_property('coarse')]
                self.wavelet_group.set_visible(True)
                for child in children:
                    child.set_visible(True)
                for i in range(3):
                    self.apply_unsharp_mask(scales[i], 16.0, amounts[i], 0.0)

                # Merge using new method
                merged_layer = self.get_merged_from_group(image, self.wavelet_group)

                # Remove the wavelet group
                image.remove_layer(self.wavelet_group)

                # Replace original: record position, remove original, insert
                # merged at exactly that position. Avoids the double position
                # shift caused by insert-at-0 followed by reorder_item.
                _, offset_x, offset_y = drawable.get_offsets()
                merged_layer.set_offsets(offset_x, offset_y)
                merged_layer.set_mode(drawable.get_mode())
                merged_layer.set_visible(drawable.get_visible())
                merged_layer.set_name(drawable.get_name())
                original_position = image.get_item_position(drawable)
                original_parent = drawable.get_parent()
                image.remove_layer(drawable)
                image.insert_layer(merged_layer, original_parent, original_position)

                image.undo_group_end()
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

            except Exception as e:
                Gimp.message("Error in interactive mode:\n" + traceback.format_exc())
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
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, str(e)))

        # Non-interactive mode
        fine = config.get_property('fine')
        medium = config.get_property('medium')
        coarse = config.get_property('coarse')
        amounts = [fine, medium, coarse]

        image.undo_group_start()

        try:
            original_position = image.get_item_position(drawable)
            original_parent = drawable.get_parent()

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
                error = 'Wavelet decompose group not found.'
                image.remove_layer(temp)
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, error))

            children = group.get_children()
            if len(children) != 4:
                error = 'Unexpected number of layers in decomposition.'
                image.remove_layer(temp)
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error.new_literal(GLib.quark_from_string('gimp-plug-in-error'), 0, error))

            scales = children[0:3]

            group.set_visible(True)
            for child in children:
                child.set_visible(True)

            for i in range(3):
                    self.apply_unsharp_mask(scales[i], 16.0, amounts[i], 0.0)

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

        except Exception as e:
            Gimp.message("Error in processing:\n" + traceback.format_exc())
            raise
        finally:
            image.undo_group_end()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

Gimp.main(WaveletSharpen.__gtype__, sys.argv)
