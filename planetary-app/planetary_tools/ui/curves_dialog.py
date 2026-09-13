"""Interactive GIMP-style curves editor shared by the canvas and batch tools."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout, QWidget,
)

from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.filters.curves import (
    CHANNELS, N_SAMPLES, change_curve_mode, curve_samples, default_curves_params,
    identity_curve, map_samples, normalize_curves_params,
)
from planetary_tools.ui.dialogs import _FilterDialog

_COLOURS = {"Value": "#dedede", "Red": "#ef6868", "Green": "#73d985",
            "Blue": "#80a9ff", "Alpha": "#b4b4b4"}


class CurveEditor(QWidget):
    changed = pyqtSignal()
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.curve = identity_curve()
        self.channel = "Value"
        self.selected = -1
        self.histogram = np.zeros(N_SAMPLES)
        self.logarithmic = True
        self._dragging = False
        self._last = None
        self.setMinimumSize(300, 245)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Click to add a point; drag to adjust. Delete or right-click removes a point.")
        self.setAccessibleName("Curves input and output graph")

    def sizeHint(self):
        return QSize(360, 280)

    def plot_rect(self):
        return QRectF(32, 12, max(1, self.width()-48), max(1, self.height()-42))

    def _position(self, x, y):
        r = self.plot_rect()
        return QPointF(r.left()+x*r.width(), r.bottom()-y*r.height())

    def _values(self, position):
        r = self.plot_rect()
        return (float(np.clip((position.x()-r.left())/r.width(), 0, 1)),
                float(np.clip((r.bottom()-position.y())/r.height(), 0, 1)))

    def set_curve(self, curve, channel):
        self.curve = deepcopy(curve)
        self.channel = channel
        self.selected = -1
        self._dragging = False
        self.selection_changed.emit()
        self.update()

    def _notify(self):
        self.update()
        self.selection_changed.emit()
        self.changed.emit()

    def add_point(self, x, y):
        if self.curve['mode'] != 'smooth':
            return
        points = self.curve['points']
        nearby = next((i for i, p in enumerate(points) if abs(p[0]-x) < 1e-6), None)
        if nearby is not None:
            self.selected = nearby
            points[nearby][1] = float(y)
        else:
            points.append([float(x), float(y), 'smooth'])
            points.sort(key=lambda p: p[0])
            self.selected = next(i for i, p in enumerate(points) if p[0] == x)
        self._notify()

    def move_selected(self, x, y):
        if self.selected < 0 or self.curve['mode'] != 'smooth':
            return
        points = self.curve['points']
        lower = points[self.selected-1][0]+1e-6 if self.selected else 0
        upper = points[self.selected+1][0]-1e-6 if self.selected+1 < len(points) else 1
        points[self.selected][:2] = [float(np.clip(x, lower, upper)), float(np.clip(y, 0, 1))]
        self._notify()

    def delete_selected(self):
        if self.curve['mode'] == 'smooth' and self.selected >= 0 and len(self.curve['points']) > 2:
            del self.curve['points'][self.selected]
            self.selected = -1
            self._notify()

    def _draw_freehand(self, x, y):
        index = int(x*(N_SAMPLES-1)+.5)
        previous, value = self._last if self._last is not None else (index, y)
        values = np.linspace(value, y, abs(index-previous)+1)
        if index < previous:
            values = values[::-1]
        self.curve['samples'][min(index, previous):max(index, previous)+1] = values.tolist()
        self._last = (index, y)
        self._notify()

    def mousePressEvent(self, event):
        if not self.plot_rect().contains(event.position()):
            return
        self.setFocus()
        x, y = self._values(event.position())
        if self.curve['mode'] == 'smooth':
            distances = [(self._position(p[0], p[1])-event.position()).manhattanLength()
                         for p in self.curve['points']]
            closest = int(np.argmin(distances))
            if distances[closest] < 14:
                self.selected = closest
                self.selection_changed.emit()
                self.update()
            elif event.button() == Qt.MouseButton.LeftButton:
                self.add_point(x, y)
            else:
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.delete_selected()
                return
        elif event.button() == Qt.MouseButton.LeftButton:
            self._last = None
            self._draw_freehand(x, y)
        self._dragging = event.button() == Qt.MouseButton.LeftButton

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        x, y = self._values(event.position())
        if self.curve['mode'] == 'smooth':
            self.move_selected(x, y)
        else:
            self._draw_freehand(x, y)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self.mouseMoveEvent(event)
        self._dragging = False
        self._last = None

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        elif self.selected >= 0 and event.key() in (
            Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
        ):
            x, y, _ = self.curve['points'][self.selected]
            delta = 10/255 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1/255
            self.move_selected(x + delta * ((event.key() == Qt.Key.Key_Right)-(event.key() == Qt.Key.Key_Left)),
                               y + delta * ((event.key() == Qt.Key.Key_Up)-(event.key() == Qt.Key.Key_Down)))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.plot_rect()
        painter.fillRect(r, QColor('#171a20'))
        heights = np.log1p(self.histogram) if self.logarithmic else self.histogram.copy()
        if heights.max() > 0:
            heights = heights/heights.max()
            path = QPainterPath(self._position(0, 0))
            for i, height in enumerate(heights):
                path.lineTo(self._position(i/255, float(height)))
            path.lineTo(self._position(1, 0))
            painter.fillPath(path, QColor('#3d424d'))
        painter.setPen(QPen(QColor('#555a64'), 1))
        for fraction in np.linspace(0, 1, 5):
            painter.drawLine(self._position(fraction, 0), self._position(fraction, 1))
            painter.drawLine(self._position(0, fraction), self._position(1, fraction))
        painter.setPen(QPen(QColor('#808590'), 1, Qt.PenStyle.DashLine))
        painter.drawLine(self._position(0, 0), self._position(1, 1))
        path = QPainterPath()
        for i, value in enumerate(curve_samples(self.curve)):
            pos = self._position(i/255, float(value))
            path.moveTo(pos) if i == 0 else path.lineTo(pos)
        painter.setPen(QPen(QColor(_COLOURS[self.channel]), 2))
        painter.drawPath(path)
        if self.curve['mode'] == 'smooth':
            for i, (x, y, kind) in enumerate(self.curve['points']):
                painter.setBrush(QColor('#ffffff' if i == self.selected else _COLOURS[self.channel]))
                pos = self._position(x, y)
                if kind == 'corner':
                    painter.drawRect(QRectF(pos.x()-4, pos.y()-4, 8, 8))
                else:
                    painter.drawEllipse(pos, 5, 5)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(QRectF(r.left(), r.bottom()+5, r.width(), 22), Qt.AlignmentFlag.AlignCenter, 'Input')
        painter.drawText(QRectF(0, r.bottom()-16, 28, 20), Qt.AlignmentFlag.AlignRight, '0')
        painter.drawText(QRectF(0, r.top(), 28, 20), Qt.AlignmentFlag.AlignRight, '255')
        painter.drawText(QRectF(r.right()-28, r.bottom()+5, 32, 22), Qt.AlignmentFlag.AlignRight, '255')
        painter.save()
        painter.translate(12, r.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-40, -10, 80, 20), Qt.AlignmentFlag.AlignCenter, 'Output')
        painter.restore()


class CurvesDialog(_FilterDialog):
    filter_id = 'curves'
    supports_presets = True

    def __init__(self, parent=None):
        self._settings = default_curves_params()
        self._source = None
        self._canvas = None
        self._loading = False
        super().__init__('Curves', parent)

    def _build_filter_params(self):
        self.channel = QComboBox()
        self.channel.addItems(CHANNELS)
        self.channel.model().item(4).setEnabled(False)
        self.channel.currentTextChanged.connect(self._load_channel)
        self._form.addRow('Channel', self.channel)
        self.trc = QComboBox()
        self.trc.addItem('Perceptual (sRGB)', 'perceptual')
        self.trc.addItem('Linear light', 'linear')
        self.trc.currentIndexChanged.connect(self._trc_changed)
        self._form.addRow('Work in', self.trc)
        self.editor = CurveEditor()
        self.editor.changed.connect(self._curve_changed)
        self.editor.selection_changed.connect(self._sync_point)
        self._form.addRow(self.editor)
        self.mode = QComboBox()
        self.mode.addItems(['Smooth', 'Freehand'])
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.log_hist = QCheckBox('Log histogram')
        self.log_hist.setChecked(True)
        self.log_hist.toggled.connect(self._hist_scale_changed)
        row = QHBoxLayout()
        row.addWidget(self.mode)
        row.addWidget(self.log_hist)
        self._form.addRow(row)
        row = QHBoxLayout()
        self.input = QDoubleSpinBox()
        self.output = QDoubleSpinBox()
        for label, spin in (('Input', self.input), ('Output', self.output)):
            spin.setRange(0, 255)
            spin.setDecimals(2)
            spin.setSingleStep(1)
            spin.valueChanged.connect(self._point_edited)
            row.addWidget(QLabel(label))
            row.addWidget(spin)
        self._form.addRow(row)
        self.point_type = QComboBox()
        self.point_type.addItems(['Smooth', 'Corner'])
        self.point_type.currentIndexChanged.connect(self._point_type_changed)
        self.delete = QPushButton('Delete point')
        self.delete.clicked.connect(self.editor.delete_selected)
        row = QHBoxLayout()
        row.addWidget(self.point_type)
        row.addWidget(self.delete)
        self._form.addRow(row)
        self.pick = QCheckBox('Pick a curve point from the image')
        self.pick.setToolTip('Enable, then click the image to add its input level to this curve.')
        self.pick.setEnabled(False)
        self._form.addRow(self.pick)
        reset = QPushButton('Reset channel')
        reset_all = QPushButton('Reset all')
        reset.clicked.connect(self._reset_channel)
        reset_all.clicked.connect(self._reset_all)
        row = QHBoxLayout()
        row.addWidget(reset)
        row.addWidget(reset_all)
        self._form.addRow(row)
        self.set_help_text('Click the graph to add points and drag to reshape the curve. '
                           'Delete or right-click a point to remove it. Arrow keys move a selected point; '
                           'Shift moves ten levels. Value maps all RGB channels after their individual curves. '
                           'Perceptual is GIMP’s default; Linear light works on linear pixel values.')
        self._sync_point()

    def _load_channel(self, *_):
        if not hasattr(self, 'editor'):
            return
        ch = self.channel.currentText()
        curve = self._settings['channels'][ch]
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0 if curve['mode'] == 'smooth' else 1)
        self.mode.blockSignals(False)
        self.editor.set_curve(curve, ch)
        self.pick.setEnabled(self._canvas is not None and curve['mode'] == 'smooth')
        self._update_histogram()

    def _curve_changed(self):
        self._settings['channels'][self.channel.currentText()] = deepcopy(self.editor.curve)
        self.params_changed.emit()

    def _trc_changed(self):
        self._settings['trc'] = self.trc.currentData()
        self._update_histogram()
        self.params_changed.emit()

    def _mode_changed(self):
        ch = self.channel.currentText()
        self._settings['channels'][ch] = change_curve_mode(
            self.editor.curve, 'smooth' if self.mode.currentIndex() == 0 else 'freehand')
        self._load_channel()
        self.params_changed.emit()

    def _sync_point(self):
        if not hasattr(self, 'point_type'):
            return
        enabled = self.editor.selected >= 0 and self.editor.curve['mode'] == 'smooth'
        for widget in (self.input, self.output, self.point_type):
            widget.setEnabled(enabled)
        self.delete.setEnabled(enabled and len(self.editor.curve.get('points', [])) > 2)
        if enabled:
            x, y, kind = self.editor.curve['points'][self.editor.selected]
            self._loading = True
            self.input.setValue(x*255)
            self.output.setValue(y*255)
            self.point_type.setCurrentIndex(0 if kind == 'smooth' else 1)
            self._loading = False

    def _point_edited(self):
        if not self._loading:
            self.editor.move_selected(self.input.value()/255, self.output.value()/255)

    def _point_type_changed(self):
        if not self._loading and self.editor.selected >= 0:
            self.editor.curve['points'][self.editor.selected][2] = self.point_type.currentText().lower()
            self.editor._notify()

    def _hist_scale_changed(self, checked):
        self.editor.logarithmic = checked
        self.editor.update()

    def _update_histogram(self):
        if self._source is None:
            return
        data = self._source
        ch = self.channel.currentText()
        if data.ndim == 2 or data.shape[-1] == 1:
            values = data.ravel()
        elif ch == 'Value':
            values = data[..., :3].max(axis=-1).ravel()
        else:
            values = data[..., CHANNELS.index(ch)-1].ravel()
        values = values[::max(1, int(np.ceil(values.size/250_000)))]
        if self._settings['trc'] == 'perceptual' and ch != 'Alpha':
            values = linear_to_srgb(values)
        values = values[np.isfinite(values)]
        self.editor.histogram = np.histogram(np.clip(values, 0, 1), bins=256, range=(0, 1))[0]
        self.editor.update()

    def set_image_channels(self, is_grayscale, has_alpha=False):
        for index in (1, 2, 3):
            self.channel.model().item(index).setEnabled(not is_grayscale)
        self.channel.model().item(4).setEnabled(has_alpha)
        if (is_grayscale and self.channel.currentIndex() in (1, 2, 3)) or (
            not has_alpha and self.channel.currentIndex() == 4
        ):
            self.channel.setCurrentIndex(0)

    def set_input_brightness(self, data, is_grayscale, **kwargs):
        super().set_input_brightness(data, is_grayscale, **kwargs)
        self._source = np.asarray(data)
        self.set_image_channels(is_grayscale or data.ndim == 2 or data.shape[-1] == 1,
                                data.ndim == 3 and data.shape[-1] == 4)
        self._update_histogram()

    def _reset_channel(self):
        self._settings['channels'][self.channel.currentText()] = identity_curve()
        self._load_channel()
        self.params_changed.emit()

    def _reset_all(self):
        self.set_params(default_curves_params())
        self.params_changed.emit()

    def get_params(self) -> dict[str, Any]:
        return deepcopy(self._settings)

    def set_params(self, params):
        settings = normalize_curves_params(params)
        self._settings = settings
        self.trc.blockSignals(True)
        self.trc.setCurrentIndex(0 if settings['trc'] == 'perceptual' else 1)
        self.trc.blockSignals(False)
        self._load_channel()

    def attach_canvas(self, canvas):
        self._canvas = canvas
        canvas.viewport().installEventFilter(self)
        self.pick.setEnabled(self.editor.curve['mode'] == 'smooth')

    def detach_canvas(self):
        if self._canvas is not None:
            self._canvas.viewport().removeEventFilter(self)
            self._canvas = None

    def eventFilter(self, obj, event):
        if (self._canvas is not None and obj is self._canvas.viewport()
                and self.pick.isChecked() and self.pick.isEnabled() and self._source is not None
                and event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton):
            point = self._canvas.mapToScene(event.position().toPoint())
            x, y = int(np.floor(point.x())), int(np.floor(point.y()))
            if 0 <= y < self._source.shape[0] and 0 <= x < self._source.shape[1]:
                pixel = self._source[y, x]
                ch = self.channel.currentText()
                value = (float(pixel) if np.ndim(pixel) == 0 else float(pixel[0]) if len(pixel) == 1
                         else float(np.max(pixel[:3])) if ch == 'Value' else float(pixel[CHANNELS.index(ch)-1]))
                if self._settings['trc'] == 'perceptual' and ch != 'Alpha':
                    value = float(linear_to_srgb(np.array(value)))
                if np.isfinite(value):
                    value = float(np.clip(value, 0, 1))
                    self.editor.add_point(value, float(map_samples(np.array(value), curve_samples(self.editor.curve))))
                self.pick.setChecked(False)
            return True
        return super().eventFilter(obj, event)


def edit_curves_params(params, is_grayscale, parent=None, preset_name=None):
    dialog = QDialog(parent)
    dialog.setWindowTitle('Curves')
    layout = QVBoxLayout(dialog)
    editor = CurvesDialog(dialog)
    editor.set_params(params)
    editor.set_image_channels(is_grayscale)
    editor.preview.hide()
    editor._input_label.hide()
    editor._output_label.hide()
    editor.pick.hide()
    editor._preset_combo.blockSignals(True)
    editor._set_combo_to(preset_name or '(None)')
    editor._preset_combo.blockSignals(False)
    layout.addWidget(editor)
    editor.accepted.connect(dialog.accept)
    editor.rejected.connect(dialog.reject)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    name = editor._preset_combo.currentText()
    return editor.get_params(), None if name == '(None)' else name
