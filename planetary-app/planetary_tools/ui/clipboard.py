"""Copy the image to, and paste an image from, the system clipboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QMimeData
from PyQt6.QtGui import QImage

from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.core.document import ImageDocument
from planetary_tools.io.loader import document_from_samples, supported_extensions


def qimage_from_document(doc: ImageDocument) -> QImage:
    """The image as 16-bit sRGB, as a 16-bit PNG save would write it."""
    srgb = np.clip(linear_to_srgb(doc.data), 0.0, 1.0)
    height, width = srgb.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint16)
    rgba[..., :3] = (srgb * 65535.0 + 0.5).astype(np.uint16)  # As save_image rounds.
    rgba[..., 3] = 65535
    image = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBX64)
    # QImage borrows the buffer: copy it before ``rgba`` is freed.
    return image.copy()


def _is_16_bit(image: QImage) -> bool:
    return image.format() in (
        QImage.Format.Format_RGBX64, QImage.Format.Format_RGBA64,
        QImage.Format.Format_RGBA64_Premultiplied, QImage.Format.Format_Grayscale16,
    )


def document_from_qimage(image: QImage) -> ImageDocument:
    """An untitled document from a pasted image; transparency is dropped."""
    if image.isNull():
        raise ValueError("The clipboard image is empty.")
    if _is_16_bit(image):
        converted = image.convertToFormat(QImage.Format.Format_RGBX64)
        dtype, channels = np.uint16, 4
    else:
        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        dtype, channels = np.uint8, 3
    width, height = converted.width(), converted.height()
    row = np.frombuffer(converted.constBits().asstring(converted.sizeInBytes()), dtype=dtype)
    # Rows can be padded: take each row's stride, then the visible pixels.
    stride = converted.bytesPerLine() // np.dtype(dtype).itemsize
    samples = row.reshape(height, stride)[:, :width * channels].reshape(height, width, channels)
    return document_from_samples(np.ascontiguousarray(samples[..., :3]))


def image_file_from_mime(mime: QMimeData | None) -> Path | None:
    """A copied image file (as from a file manager) that PT can open."""
    if mime is None or not mime.hasUrls():
        return None
    extensions = set(supported_extensions())
    for url in mime.urls():
        if url.isLocalFile():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in extensions and path.is_file():
                return path
    return None
