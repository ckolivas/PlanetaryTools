"""Caption controls and system-font rendering for the current image."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PyQt6.QtCore import QPointF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPixmap, QTextLayout, QTextOption
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFontComboBox,
    QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QSpinBox, QVBoxLayout,
)

from planetary_tools.core.colour import srgb_to_linear

POSITIONS = ('Top left', 'Top centre', 'Top right',
             'Bottom left', 'Bottom centre', 'Bottom right')
WEB_COLOURS = tuple(name for name in QColor.colorNames() if name != 'transparent')


@dataclass(frozen=True)
class CaptionOverlay:
    x: int
    y: int
    coverage: np.ndarray
    colour: np.ndarray


def make_caption_overlay(width: int, height: int, text: str, font: QFont,
                         colour: QColor, position: str, border: int) -> CaptionOverlay | None:
    """Render on the GUI thread, positioning visible ink inside the inset."""
    if position not in POSITIONS or border < 0:
        raise ValueError('Choose a caption position and a non-negative border.')
    if not text.strip():
        return None
    available_w, available_h = width - 2*border, height - 2*border
    if available_w < 1 or available_h < 1:
        raise ValueError('The border leaves no room for a caption.')
    align = (Qt.AlignmentFlag.AlignLeft if position.endswith('left') else
             Qt.AlignmentFlag.AlignRight if position.endswith('right') else Qt.AlignmentFlag.AlignHCenter)
    option = QTextOption(align)
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layouts = []
    total_height = 0.
    padding = max(4, font.pixelSize())  # Bearings, accents and underline ink.
    for paragraph in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        layout = QTextLayout(paragraph or ' ', font)
        layout.setTextOption(option)
        layout.beginLayout()
        try:
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(available_w)
                line.setPosition(QPointF(0, total_height))
                total_height += line.height()
                if total_height > available_h + 2*padding:
                    raise ValueError('Caption is too tall. Reduce the font size, border or number of lines.')
        finally:
            layout.endLayout()
        layouts.append(layout)
    image = QImage(available_w + 2*padding, math.ceil(total_height) + 2*padding,
                   QImage.Format.Format_RGBA8888)
    if image.isNull():
        raise ValueError('Unable to render a caption at this size.')
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(Qt.GlobalColor.white)
        for layout in layouts:
            layout.draw(painter, QPointF(padding, padding))
    finally:
        painter.end()
    pixels = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), dtype=np.uint8)
    alpha = pixels.reshape(image.height(), image.bytesPerLine())[:, :image.width()*4].reshape(
        image.height(), image.width(), 4)[..., 3]
    rows, columns = np.nonzero(alpha)
    if not rows.size:
        return None
    coverage = alpha[rows.min():rows.max()+1, columns.min():columns.max()+1].astype(np.float32) / 255
    ink_h, ink_w = coverage.shape
    if ink_w > available_w or ink_h > available_h:
        raise ValueError('Caption does not fit. Reduce the font size, border or amount of text.')
    x = border if position.endswith('left') else width-border-ink_w if position.endswith('right') else (width-ink_w)//2
    y = border if position.startswith('Top') else height-border-ink_h
    rgb = srgb_to_linear(np.array(colour.getRgbF()[:3], dtype=np.float32))
    coverage.setflags(write=False)
    rgb.setflags(write=False)
    return CaptionOverlay(x, y, coverage, rgb)


def apply_caption(data: np.ndarray, overlay: CaptionOverlay | None) -> np.ndarray:
    """Composite in linear light, leaving pixels outside the glyphs exact."""
    source = np.asarray(data, dtype=np.float32)
    if overlay is None:
        return source.copy()
    if source.ndim == 2 and np.all(overlay.colour == overlay.colour[0]):
        result = source.copy()
        colour = overlay.colour[0]
        coverage = overlay.coverage
    else:
        result = np.repeat(source[..., None], 3, axis=2) if source.ndim == 2 else source.copy()
        colour = overlay.colour
        coverage = overlay.coverage[..., None]
    h, w = overlay.coverage.shape
    region = result[overlay.y:overlay.y+h, overlay.x:overlay.x+w]
    ink = overlay.coverage > 0
    blended = region*(1-coverage) + colour*coverage
    region[ink] = blended[ink]
    return result


class CaptionDialog(QDialog):
    params_changed = pyqtSignal()
    preview_toggled = pyqtSignal(bool)

    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Caption')
        self._width, self._height = width, height
        self._overlay = None
        root = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText('Type your caption here. Use Enter for a new line.')
        self.text.setMinimumHeight(110)
        root.addWidget(QLabel('Caption text'))
        root.addWidget(self.text)
        form = QFormLayout()
        root.addLayout(form)
        self.position = QComboBox()
        self.position.addItems(POSITIONS)
        self.position.setCurrentText('Bottom left')
        form.addRow('Position', self.position)
        self.font_family = QFontComboBox()
        self.font_family.setCurrentFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
        form.addRow('Font', self.font_family)
        self.font_size = QSpinBox()
        self.font_size.setRange(1, 2048)
        self.font_size.setValue(12)
        self.font_size.setSuffix(' px')
        self.font_size.setKeyboardTracking(False)
        self.font_size.setToolTip('Font size in image pixels, independent of canvas zoom and display scaling.')
        form.addRow('Font size', self.font_size)
        style = QHBoxLayout()
        self.bold, self.underline, self.italic = (QCheckBox(label) for label in ('Bold', 'Underline', 'Italic'))
        for box in (self.bold, self.underline, self.italic):
            style.addWidget(box)
        form.addRow('Style', style)
        self.colour = QComboBox()
        for name in WEB_COLOURS:
            swatch = QPixmap(20, 16)
            swatch.fill(QColor(name))
            self.colour.addItem(QIcon(swatch), f'{name}  {QColor(name).name()}', name)
        self.colour.setCurrentIndex(self.colour.findData('white'))
        self.colour.setMaxVisibleItems(16)
        form.addRow('Text colour', self.colour)
        self.border = QSpinBox()
        self.border.setRange(0, max(0, (min(width, height)-1)//2))
        self.border.setValue(min(10, self.border.maximum()))
        self.border.setSuffix(' px')
        self.border.setKeyboardTracking(False)
        self.border.setToolTip('Distance from the visible text to the nearest top/bottom and left/right image edges. Centre positions are horizontally centred.')
        form.addRow('Border', self.border)
        self.message = QLabel()
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        self.preview = QCheckBox('Preview on canvas')
        self.preview.setChecked(True)
        self.preview.toggled.connect(self.preview_toggled.emit)
        root.addWidget(self.preview)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self._restore_settings()
        self.text.textChanged.connect(self._update_overlay)
        self.position.currentIndexChanged.connect(self._update_overlay)
        self.font_family.currentFontChanged.connect(self._update_overlay)
        self.font_size.valueChanged.connect(self._update_overlay)
        self.colour.currentIndexChanged.connect(self._update_overlay)
        self.border.valueChanged.connect(self._update_overlay)
        for box in (self.bold, self.underline, self.italic):
            box.toggled.connect(self._update_overlay)
        self.resize(420, self.sizeHint().height())
        self._update_overlay()

    def _restore_settings(self):
        settings = QSettings()

        def value(key, default):
            try:
                return settings.value(f'caption/{key}', default, type=type(default))
            except (TypeError, ValueError):
                return default

        self.text.setPlainText(value('text', ''))
        position = value('position', 'Bottom left')
        if position in POSITIONS:
            self.position.setCurrentText(position)
        family = value('font_family', self.font_family.currentFont().family())
        if self.font_family.findText(family) >= 0:
            self.font_family.setCurrentFont(QFont(family))
        self.font_size.setValue(value('font_size', 12))
        for key in ('bold', 'underline', 'italic'):
            getattr(self, key).setChecked(value(key, False))
        colour_index = self.colour.findData(value('colour', 'white'))
        if colour_index >= 0:
            self.colour.setCurrentIndex(colour_index)
        # The spin box limits the saved inset to the new image dimensions.
        self.border.setValue(value('border', 10))
        self.preview.setChecked(value('preview', True))

    def save_settings(self):
        """Remember the last applied caption, including its text, across restarts."""
        settings = QSettings()
        values = {
            'text': self.text.toPlainText(),
            'position': self.position.currentText(),
            'font_family': self.font_family.currentFont().family(),
            'font_size': self.font_size.value(),
            'bold': self.bold.isChecked(),
            'underline': self.underline.isChecked(),
            'italic': self.italic.isChecked(),
            'colour': self.colour.currentData(),
            'border': self.border.value(),
            'preview': self.preview.isChecked(),
        }
        for key, value in values.items():
            settings.setValue(f'caption/{key}', value)
        settings.sync()

    def _update_overlay(self, *_):
        font = QFont(self.font_family.currentFont())
        font.setPixelSize(self.font_size.value())
        font.setBold(self.bold.isChecked())
        font.setUnderline(self.underline.isChecked())
        font.setItalic(self.italic.isChecked())
        try:
            self._overlay = make_caption_overlay(self._width, self._height, self.text.toPlainText(),
                font, QColor(self.colour.currentData()), self.position.currentText(), self.border.value())
            self.message.setText('Text wraps within the border. OK adds it to the image; Undo removes it.')
        except ValueError as exc:
            self._overlay = None
            self.message.setText(str(exc))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self.has_caption())
        self.params_changed.emit()

    def has_caption(self) -> bool:
        return self._overlay is not None

    def build_filter_func(self):
        overlay = self._overlay
        return lambda data, grayscale: apply_caption(data, overlay)
