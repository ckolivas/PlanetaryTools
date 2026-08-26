"""Write a looping animation (GIF / APNG / WebP) from a sequence of stills."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.io.loader import load_image

ProgressFn = Callable[[int, int, str], None]

FORMATS = ("gif", "apng", "webp")
GIF_QUALITIES = ("best", "high", "medium", "low")
FORMAT_SUFFIX = {"gif": ".gif", "apng": ".png", "webp": ".webp"}

_GIF_PRESETS: dict[str, tuple[int, Image.Dither]] = {
    "best": (256, Image.Dither.FLOYDSTEINBERG),
    "high": (256, Image.Dither.NONE),
    "medium": (128, Image.Dither.FLOYDSTEINBERG),
    "low": (64, Image.Dither.FLOYDSTEINBERG),
}

_ANIM_SUFFIXES = {".gif", ".png", ".webp", ".apng"}


@dataclass(frozen=True)
class AnimationResult:
    path: Path
    frames: int
    width: int
    height: int
    duration_ms: int
    fps_requested: float
    fmt: str


def natural_sort_key(path: Path) -> tuple:
    """Sort key so ``img_2.png`` precedes ``img_10.png``."""
    name = path.name.lower()
    parts = tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))
    return parts + (str(path).lower(),)


def duration_ms(fmt: str, fps: float) -> int:
    """Frame delay in milliseconds for ``fmt`` at ``fps``.

    GIF stores delay in hundredths of a second, so the value is snapped to
    10 ms. APNG and WebP use a 1 ms tick.
    """
    fps = float(fps)
    if fps <= 0:
        raise ValueError("Frame rate must be positive.")
    if fmt == "gif":
        return max(10, int(round(100.0 / fps)) * 10)
    if fmt in ("apng", "webp"):
        return max(1, int(round(1000.0 / fps)))
    raise ValueError(f"Unknown animation format: {fmt}")


def apply_format_suffix(path: Path | str, fmt: str) -> Path:
    """Replace a known animation suffix, or append the format's suffix."""
    if fmt not in FORMAT_SUFFIX:
        raise ValueError(f"Unknown animation format: {fmt}")
    p = Path(path)
    suffix = FORMAT_SUFFIX[fmt]
    if p.suffix.lower() in _ANIM_SUFFIXES:
        return p.with_suffix(suffix)
    if p.suffix:
        return p.with_suffix(suffix)
    return Path(str(p) + suffix)


def _to_uint8_srgb(data: np.ndarray) -> np.ndarray:
    srgb = np.clip(linear_to_srgb(np.asarray(data, dtype=np.float32)), 0.0, 1.0)
    if srgb.ndim == 2:
        srgb = np.stack([srgb, srgb, srgb], axis=-1)
    elif srgb.ndim == 3 and srgb.shape[2] >= 3:
        srgb = srgb[..., :3]
    else:
        raise ValueError(f"Unsupported frame shape: {srgb.shape}")
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def centre_pad_uint8(frame: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Centre ``frame`` on a black ``canvas_w``×``canvas_h`` uint8 RGB canvas."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported frame shape: {arr.shape}")
    arr = arr[..., :3]
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h == canvas_h and w == canvas_w:
        return np.asarray(arr, dtype=np.uint8)
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y0 = (canvas_h - h) // 2
    x0 = (canvas_w - w) // 2
    if y0 < 0 or x0 < 0 or y0 + h > canvas_h or x0 + w > canvas_w:
        raise ValueError("Frame is larger than the animation canvas.")
    out[y0 : y0 + h, x0 : x0 + w] = arr
    return out


def pad_frames(frames: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Centre-pad every frame onto a canvas of max width × max height."""
    arrays = [np.asarray(f) for f in frames]
    if not arrays:
        raise ValueError("Need at least one frame.")
    canvas_h = max(int(a.shape[0]) for a in arrays)
    canvas_w = max(int(a.shape[1]) for a in arrays)
    return [centre_pad_uint8(a, canvas_w, canvas_h) for a in arrays]


def expand_back_and_forth(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Append the sequence in reverse, omitting both endpoints so a loop does not hitch.

    ``A B C D E`` becomes ``A B C D E D C B``, which loops as a ping-pong.
    Two-frame sequences are unchanged (``A B`` already ping-pongs when looped).
    """
    if len(frames) < 3:
        return list(frames)
    return list(frames) + list(reversed(frames[1:-1]))


def _quantize_gif(im: Image.Image, colors: int, dither: Image.Dither) -> Image.Image:
    try:
        return im.quantize(
            colors=colors,
            method=Image.Quantize.MAXCOVERAGE,
            dither=dither,
        )
    except Exception:
        return im.quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=dither,
        )


def encode_frames(
    frames: list[np.ndarray],
    output: str | Path,
    *,
    fps: float,
    fmt: str,
    gif_quality: str = "best",
    back_and_forth: bool = True,
) -> AnimationResult:
    """Write already-padded uint8 RGB frames to ``output``."""
    if len(frames) < 2:
        raise ValueError("Need at least two frames to write an animation.")
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unknown animation format: {fmt}")
    if back_and_forth:
        frames = expand_back_and_forth(frames)
    delay = duration_ms(fmt, fps)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    pil_rgb = [Image.fromarray(np.asarray(f, dtype=np.uint8), mode="RGB") for f in frames]
    h, w = frames[0].shape[:2]

    if fmt == "gif":
        quality = gif_quality.lower()
        if quality not in _GIF_PRESETS:
            raise ValueError(f"Unknown GIF quality: {gif_quality}")
        colors, dither = _GIF_PRESETS[quality]
        paletted = [_quantize_gif(im, colors, dither) for im in pil_rgb]
        delays = [delay] * len(paletted)
        paletted[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=paletted[1:],
            duration=delays,
            loop=0,
            optimize=True,
            # Do not restore to background after each frame. Disposal 2 flashes
            # a blank canvas at the loop wrap, which looks like a pause.
            disposal=1,
        )
    elif fmt == "apng":
        delays = [delay] * len(pil_rgb)
        pil_rgb[0].save(
            path,
            format="PNG",
            save_all=True,
            append_images=pil_rgb[1:],
            duration=delays,
            loop=0,
            default_image=False,
            disposal=0,
        )
    else:
        delays = [delay] * len(pil_rgb)
        pil_rgb[0].save(
            path,
            format="WEBP",
            save_all=True,
            append_images=pil_rgb[1:],
            duration=delays,
            loop=0,
            lossless=True,
            quality=100,
            method=6,
        )

    return AnimationResult(
        path=path,
        frames=len(frames),
        width=w,
        height=h,
        duration_ms=delay,
        fps_requested=float(fps),
        fmt=fmt,
    )


def write_animation(
    paths: list[Path] | list[str],
    output: str | Path,
    *,
    fps: float,
    fmt: str,
    gif_quality: str = "best",
    back_and_forth: bool = True,
    on_progress: ProgressFn | None = None,
) -> AnimationResult:
    """Load stills, pad to a common canvas, and write a looping animation."""
    files = [Path(p) for p in paths]
    if len(files) < 2:
        raise ValueError("Select at least two images.")
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unknown animation format: {fmt}")

    total = len(files) + 1
    loaded: list[np.ndarray] = []
    for i, path in enumerate(files):
        if on_progress is not None:
            on_progress(i, total, f"Loading {path.name}")
        doc = load_image(path)
        loaded.append(_to_uint8_srgb(doc.data))

    padded = pad_frames(loaded)
    if on_progress is not None:
        on_progress(len(files), total, "Writing")
    result = encode_frames(
        padded,
        output,
        fps=fps,
        fmt=fmt,
        gif_quality=gif_quality,
        back_and_forth=back_and_forth,
    )
    if on_progress is not None:
        on_progress(total, total, "Done")
    return result
