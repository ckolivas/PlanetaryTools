"""Write an animation (GIF / APNG / WebP / MP4) from a sequence of stills."""

from __future__ import annotations

import math
import os
import re
from fractions import Fraction
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import av
import numpy as np
from PIL import Image

from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.core.animation_interpolation import build_timeline, interpolate_timeline
from planetary_tools.io.loader import load_image

ProgressFn = Callable[[int, int, str], None]
Frame = TypeVar("Frame")

FORMATS = ("gif", "apng", "webp", "mp4")
GIF_QUALITIES = ("best", "high", "medium", "low")
FORMAT_SUFFIX = {"gif": ".gif", "apng": ".png", "webp": ".webp", "mp4": ".mp4"}

_GIF_PRESETS: dict[str, tuple[int, Image.Dither]] = {
    "best": (256, Image.Dither.FLOYDSTEINBERG),
    "high": (256, Image.Dither.NONE),
    "medium": (128, Image.Dither.FLOYDSTEINBERG),
    "low": (64, Image.Dither.FLOYDSTEINBERG),
}

_ANIM_SUFFIXES = {".gif", ".png", ".webp", ".apng", ".mp4"}


@dataclass(frozen=True)
class AnimationResult:
    path: Path
    frames: int
    width: int
    height: int
    duration_ms: float
    fps_requested: float
    fmt: str


def natural_sort_key(path: Path) -> tuple:
    """Sort key so ``img_2.png`` precedes ``img_10.png``."""
    name = path.name.lower()
    parts = tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))
    return parts + (str(path).lower(),)


def duration_ms(fmt: str, fps: float) -> float:
    """Frame delay in milliseconds for ``fmt`` at ``fps``.

    GIF stores delay in hundredths of a second, so the value is snapped to
    10 ms. APNG and WebP use a 1 ms tick. MP4 uses the requested frame rate
    directly; its reported delay is not rounded to whole milliseconds.
    """
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be positive.")
    if fmt == "gif":
        return max(10, int(round(100.0 / fps)) * 10)
    if fmt in ("apng", "webp"):
        return max(1, int(round(1000.0 / fps)))
    if fmt == "mp4":
        return 1000.0 / fps
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


def expand_back_and_forth(frames: list[Frame]) -> list[Frame]:
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


def _encode_mp4(frames: list[np.ndarray], path: Path, fps: float, crf: int) -> None:
    """Encode RGB using the bundled FFmpeg libraries, without chroma subsampling."""
    if not isinstance(crf, (int, np.integer)) or not 0 <= crf <= 51:
        raise ValueError("MP4 constant quality must be an integer from 0 (lossless) to 51.")
    h, w = frames[0].shape[:2]
    if any(frame.shape != (h, w, 3) for frame in frames):
        raise ValueError("MP4 frames must be RGB images with matching dimensions.")

    # Publish only a completed video, preserving an existing output on failure.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".mp4", dir=path.parent)
    os.close(fd)
    try:
        try:
            rate = Fraction(str(float(fps)))
            with av.open(temporary, mode='w', format='mp4', options={'movflags': '+faststart'}) as container:
                stream = container.add_stream('libx264rgb', rate=rate)
                stream.width = w
                stream.height = h
                stream.pix_fmt = 'rgb24'
                stream.options = {'crf': str(crf), 'preset': 'medium'}
                for index, pixels in enumerate(frames):
                    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels, dtype=np.uint8), format='rgb24')
                    frame.pts = index
                    frame.time_base = 1 / rate
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        except (av.FFmpegError, ValueError) as exc:
            raise RuntimeError(f"MP4 export failed: {exc}") from exc
        if Path(temporary).stat().st_size == 0:
            raise RuntimeError("MP4 export failed: FFmpeg produced an empty video.")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def encode_frames(
    frames: list[np.ndarray],
    output: str | Path,
    *,
    fps: float,
    fmt: str,
    gif_quality: str = "best",
    mp4_crf: int = 0,
    back_and_forth: bool = True,
) -> AnimationResult:
    """Write already-padded uint8 RGB frames to ``output``."""
    if len(frames) < 2:
        raise ValueError("Need at least two frames to write an animation.")
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unknown animation format: {fmt}")
    delay = duration_ms(fmt, fps)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    if fmt == "mp4":
        sequence = expand_back_and_forth(frames) if back_and_forth else frames
        _encode_mp4(sequence, path, fps, mp4_crf)
        return AnimationResult(path, len(sequence), w, h, delay, float(fps), fmt)

    pil_rgb = [Image.fromarray(np.asarray(f, dtype=np.uint8), mode="RGB") for f in frames]

    if fmt == "gif":
        quality = gif_quality.lower()
        if quality not in _GIF_PRESETS:
            raise ValueError(f"Unknown GIF quality: {gif_quality}")
        colors, dither = _GIF_PRESETS[quality]
        paletted = [_quantize_gif(im, colors, dither) for im in pil_rgb]
        # Prepare each source once; the return trip reuses the same pixels
        # and palette rather than repeating expensive GIF quantization.
        if back_and_forth:
            paletted = expand_back_and_forth(paletted)
        frame_count = len(paletted)
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
        if back_and_forth:
            pil_rgb = expand_back_and_forth(pil_rgb)
        frame_count = len(pil_rgb)
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
        if back_and_forth:
            pil_rgb = expand_back_and_forth(pil_rgb)
        frame_count = len(pil_rgb)
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
        frames=frame_count,
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
    mp4_crf: int = 0,
    back_and_forth: bool = True,
    on_progress: ProgressFn | None = None,
    motion_interpolation: bool = False,
    frame_interval_minutes: float = 1.0,
) -> AnimationResult:
    """Load stills, pad to a common canvas, and write a looping animation."""
    files = [Path(p) for p in paths]
    if len(files) < 2:
        raise ValueError("Select at least two images.")
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unknown animation format: {fmt}")

    timeline = build_timeline(files, frame_interval_minutes) if motion_interpolation else None
    if timeline is not None:
        files = [frame.path for frame in timeline.sources]
    interpolation_steps = len(timeline.gaps) if timeline is not None else 0
    total = len(files) + interpolation_steps + 1
    loaded: list[np.ndarray] = []
    for i, path in enumerate(files):
        if on_progress is not None:
            on_progress(i, total, f"Loading {path.name}")
        doc = load_image(path, pin_noise=False)
        loaded.append(_to_uint8_srgb(doc.data))

    padded = pad_frames(loaded)
    if timeline is not None:
        padded = interpolate_timeline(
            padded, timeline,
            on_progress=(lambda current, _, message: on_progress(len(files)+current, total, message))
            if on_progress is not None else None,
        )
    if on_progress is not None:
        on_progress(total-1, total, "Writing")
    result = encode_frames(
        padded,
        output,
        fps=fps,
        fmt=fmt,
        gif_quality=gif_quality,
        mp4_crf=mp4_crf,
        back_and_forth=back_and_forth,
    )
    if on_progress is not None:
        on_progress(total, total, "Done")
    return result
