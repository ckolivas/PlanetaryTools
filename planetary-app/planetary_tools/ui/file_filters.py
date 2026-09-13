"""Shared file-type choices for image input dialogs."""

from pathlib import Path

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


def save_path_for_filter(path: str, selected_filter: str) -> str:
    """Make an image filename agree with a specific chosen output format."""
    for label, extensions in (
        ("PNG", (".png",)),
        ("JPEG", (".jpg", ".jpeg")),
        ("TIFF", (".tif", ".tiff")),
    ):
        if selected_filter.startswith(label):
            result = Path(path)
            return str(result if result.suffix.lower() in extensions
                       else result.with_suffix(extensions[0]))
    # All Files deliberately lets the filename determine the encoder.
    return path
