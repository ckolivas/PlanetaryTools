"""Shared file-type choices for image input dialogs."""

from planetary_tools.io.loader import supported_extensions


def image_file_filters() -> str:
    extensions = supported_extensions()
    choices = [f"All Images ({' '.join('*' + ext for ext in extensions)})"]
    for label, suffixes in (
        ("TIFF", (".tif", ".tiff")),
        ("PNG", (".png",)),
        ("JPEG", (".jpg", ".jpeg")),
        ("FITS", (".fits", ".fit", ".fts")),
        ("BMP", (".bmp",)),
        ("WebP", (".webp",)),
    ):
        patterns = " ".join('*' + ext for ext in suffixes if ext in extensions)
        if patterns:
            choices.append(f"{label} ({patterns})")
    choices.append("All Files (*)")
    return ";;".join(choices)
