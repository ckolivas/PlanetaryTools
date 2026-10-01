"""Spectrum-guided FFT denoise controls and automatic initial settings."""
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import QCheckBox, QLabel, QPushButton, QWidget

from planetary_tools.filters.fft_denoise import (
    MAX_FREQUENCY, analyse_spectrum, auto_parameters, transfer,
)
from planetary_tools.ui.dialogs import _FilterDialog


class SpectrumGraph(QWidget):
    markers_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 160)
        self.spectrum = None
        self.settings = dict(start=60., feather=40., curve=-2.)
        self._drag = None
        self.setToolTip('Blue: input power. Red: predicted filtered power. '
                        'Drag the green start or red feather-end marker. '
                        '100% Nyquist is 0.5 cycles/pixel; diagonal frequencies extend to 141%.')

    def plot_rect(self):
        return QRectF(12, 25, self.width()-24, self.height()-57)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.plot_rect()
        p.fillRect(r, QColor('#161b23'))
        p.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        p.drawText(12, 17, 'Power spectrum (log)')
        p.drawText(12, self.height()-7, 'Frequency (% Nyquist)')
        for tick in (0, 50, 100, 140):
            x = r.left()+tick/MAX_FREQUENCY*r.width()
            p.setPen(QColor('#323b48'))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(self.palette().color(QPalette.ColorRole.WindowText))
            p.drawText(QPointF(x-5, r.bottom()+14), str(tick))
        if self.spectrum is not None:
            s = self.spectrum
            gain = transfer(s.frequency, **self.settings)
            peak = max(float(s.power.max()), 1e-20)
            # Common log range, with 100 dB dynamic range; no separate scaling.
            for power, colour in ((s.power, '#78adff'), (s.power*gain**2, '#ff7777')):
                heights = np.clip((10*np.log10(np.maximum(power/peak, 1e-10))+100)/100, 0, 1)
                path = QPainterPath()
                for i, (f, value) in enumerate(zip(s.frequency, heights)):
                    point = QPointF(r.left()+f/MAX_FREQUENCY*r.width(), r.bottom()-value*r.height())
                    path.moveTo(point) if i == 0 else path.lineTo(point)
                p.setPen(QPen(QColor(colour), 1.4))
                p.drawPath(path)
        for f, colour in ((self.settings['start'], '#78db88'),
                          (self.settings['start']+self.settings['feather'], '#ff7777')):
            x = r.left()+min(f, MAX_FREQUENCY)/MAX_FREQUENCY*r.width()
            p.setPen(QPen(QColor(colour), 1.5))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.plot_rect().contains(event.position()):
            return
        r = self.plot_rect()
        f = (event.position().x()-r.left())/r.width()*MAX_FREQUENCY
        start = self.settings['start']
        end = min(MAX_FREQUENCY, start+self.settings['feather'])
        self._drag = 'start' if abs(f-start) <= abs(f-end) else 'end'
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        r = self.plot_rect()
        f = float(np.clip((event.position().x()-r.left())/r.width()*MAX_FREQUENCY, 0, MAX_FREQUENCY))
        start, feather = self.settings['start'], self.settings['feather']
        if self._drag == 'start':
            start = f
        else:
            feather = max(.5, f-start)
        self.markers_changed.emit(start, feather)

    def mouseReleaseEvent(self, event):
        self._drag = None


class FftDenoiseDialog(_FilterDialog):
    filter_id = 'fft_denoise'
    supports_presets = True
    supports_clamp = True

    def __init__(self, parent=None):
        self._spectrum = None
        self._auto_selected = False
        self._last_output = None
        super().__init__('FFT Denoise', parent)
        self.set_help_text('Use after sharpening to reduce amplified noise with a smooth FFT cutoff. '
                           'Start keeps lower frequencies unchanged; Feather is the transition width. '
                           'Negative Curve retains more detail, positive Curve removes more noise. '
                           'Auto estimates a curve from the current image and recalculates for each image in a batch. '
                           'Inspect fine detail and removed signal before applying, especially tiny moons or stars.')

    def _build_filter_params(self):
        self.graph = SpectrumGraph()
        self._form.addRow(self.graph)
        self.start = self._add_double('Start (% Nyquist)', 60, 0, 141.4, 1)
        self.feather = self._add_double('Feather width (%)', 40, .5, 141.4, 1)
        self.curve = self._add_double('Curve', -2, -5, 5, .1)
        for spin in (self.start, self.feather, self.curve):
            spin.setKeyboardTracking(False)
            spin.valueChanged.disconnect()
            spin.valueChanged.connect(self._manual_changed)
        self.start.setToolTip('Frequencies below this point are retained. 100% = 0.5 cycles/pixel.')
        self.feather.setToolTip('Width from the start to complete suppression. Wider transitions reduce ringing.')
        self.curve.setToolTip('Negative values retain more of the transition; positive values suppress more.')
        self.graph.markers_changed.connect(self._move_markers)
        self.auto_button = QPushButton('Auto')
        self.auto_button.setToolTip(
            "Choose starting points for this image. Saved Auto presets recalculate for each batch image. "
            "Adjusting the curve switches back to manual settings."
        )
        self.auto_button.setEnabled(False)
        self.auto_button.clicked.connect(self._auto)
        self._form.addRow(self.auto_button)
        self.note = QLabel('Auto estimates a starting point; inspect fine detail.')
        self.note.setWordWrap(True)
        self._form.addRow(self.note)
        self.show_difference = QCheckBox('Show removed signal')
        self.show_difference.toggled.connect(self._update_difference)
        self._form.addRow(self.show_difference)
        self.difference = QLabel()
        self.difference.setFixedSize(320, 96)
        self.difference.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.difference.setToolTip('Amplified signed input minus output: red = removed, cyan = added. '
                                   'Visible edges indicate detail loss or ringing. Display only; never saved to the image.')
        self.difference.hide()
        self._form.addRow(self.difference)

    def _refresh_graph(self):
        self.graph.settings = {key: getattr(self, key).value() for key in ('start', 'feather', 'curve')}
        self.graph.update()

    def _move_markers(self, start, feather):
        self.start.blockSignals(True)
        self.feather.blockSignals(True)
        self.start.setValue(start)
        self.feather.setValue(feather)
        self.start.blockSignals(False)
        self.feather.blockSignals(False)
        self._manual_changed()

    def _manual_changed(self):
        # Clear Auto before notifying the preview, so the edited values are used.
        self._auto_selected = False
        self._refresh_graph()
        self.params_changed.emit()

    def _auto(self):
        if self._spectrum is not None:
            self._auto_selected = True
            settings = auto_parameters(self._spectrum)
            for key, value in settings.items():
                spin = getattr(self, key)
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self._refresh_graph()
        self.params_changed.emit()

    def set_input_brightness(self, data, is_grayscale, **kwargs):
        super().set_input_brightness(data, is_grayscale, **kwargs)
        self._spectrum = analyse_spectrum(data)
        self.graph.spectrum = self._spectrum
        self.auto_button.setEnabled(True)
        h, w = self._spectrum.shape
        self.note.setText(f'Analysis: {w} × {h} native pixels. Auto is a starting point; inspect fine detail.')
        if self._auto_selected:
            self._auto()
        self._refresh_graph()

    def update_histogram_display(self, data):
        self._last_output = data
        self._update_difference()

    def _update_difference(self):
        self.difference.setVisible(self.show_difference.isChecked())
        if not self.show_difference.isChecked():
            return
        source = getattr(self, '_input_data', None)
        if source is None or self._last_output is None:
            self.difference.setText('Enable preview to see removed signal')
            return
        # Bound display-only allocations; the canvas retains full resolution.
        stride = max(1, int(np.ceil(max(source.shape[:2])/640)))
        delta = source[::stride, ::stride] - self._last_output[::stride, ::stride]
        if delta.ndim == 3:
            delta = np.mean(delta[..., :3], axis=2)
        scale = max(float(np.percentile(np.abs(delta), 99.5)), 1e-8)
        level = np.clip(delta/scale, -1, 1)
        rgb = np.stack((np.maximum(level, 0), np.maximum(-level, 0), np.maximum(-level, 0)), axis=-1)
        rgb = np.ascontiguousarray(rgb*255, dtype=np.uint8)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.difference.setPixmap(QPixmap.fromImage(img).scaled(
            self.difference.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def get_params(self):
        params = super().get_params()
        params.update({key: getattr(self, key).value() for key in ('start', 'feather', 'curve')})
        params['auto'] = self._auto_selected
        return params

    def set_params(self, params):
        super().set_params(params)
        for key in ('start', 'feather', 'curve'):
            spin = getattr(self, key)
            spin.blockSignals(True)
            spin.setValue(params.get(key, spin.value()))
            spin.blockSignals(False)
        self._auto_selected = bool(params.get('auto', False))
        if self._auto_selected and self._spectrum is not None:
            self._auto()
        self._refresh_graph()
