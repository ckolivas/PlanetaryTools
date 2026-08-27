"""Filter registry — shared definitions for UI, preview, and batch processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from planetary_tools.core.brightness import (
    BrightnessInfo,
    apply_channel_post_process,
    brightness_increase_pct,
    measure_brightness,
)
from planetary_tools.core.noise import (
    absolute_noise,
    estimate_texture_scale,
    is_chromatic,
)
from planetary_tools.filters.adaptive_deconv import adaptive_deconvolution
from planetary_tools.filters.colour_matrix import (
    IDENTITY_MATRIX,
    apply_colour_matrix,
    matrix_from_params,
)
# from planetary_tools.filters.oklab_filters import oklab_luminance
from planetary_tools.filters.levels import apply_levels, default_levels_params
from planetary_tools.core.crop import (
    DEFAULT_BORDER_PX,
    DEFAULT_MIN_BRIGHTNESS_PCT,
    autocrop_rect,
    crop_image,
    rect_from_size_offset,
)
from planetary_tools.core.rotate import rotate_image
from planetary_tools.core.scale import scale_image
from planetary_tools.filters.moon_enhance import moon_enhance
from planetary_tools.filters.saturation import apply_saturation_vibrance
from planetary_tools.filters.stretch import stretch_contrast_oklab
from planetary_tools.filters.wavelet import merge_wavelet_detail, wavelet_denoise, wavelet_sharpen
from planetary_tools.filters.wiener_deconv import wiener_deconvolution

FilterFunc = Callable[[np.ndarray, bool], np.ndarray]

# clamp: scale peak to 100% when highlights clip.
# clamp_low: when clamping highlights, scale min channel to 0% as well.
# When clamp_low is off, negatives are floored to 0% automatically.
CLAMP_PARAM = "clamp"
CLAMP_LOW_PARAM = "clamp_low"
_LEGACY_CLAMP_PARAM = "rescale"

ENHANCE_FILTER_IDS = frozenset({
    "wavelet_sharpen",
    "wavelet_denoise",
    "adaptive_deconv",
    "wiener_deconv",
})

# Moon enhance is deliberately not an "enhance" filter for stats: those noise
# readouts crop to the bright subject (the planet), which this filter does not
# change, so the score would be meaningless.
CLAMP_FILTER_IDS = ENHANCE_FILTER_IDS | frozenset({
    "colour_matrix",
    "saturation_vibrance",
    "merge_wavelet_detail",
    "moon_enhance",
})


@dataclass(frozen=True)
class FilterOutputStats:
    brightness: BrightnessInfo
    brightness_increase_pct: float | None = None
    # Absolute flat-region noise score of the pre-clip filter result.
    noise_level: float | None = None
    # Absolute noise of the unfiltered input (same scale/probes as noise_level).
    # Matches across enhance dialogs for a given document.
    source_noise_level: float | None = None


@dataclass
class FilterDef:
    id: str
    label: str
    batch_enabled: bool = True
    requires_rgb: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)

    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class WaveletSharpenDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return wavelet_sharpen(
            data, is_grayscale,
            params.get("fine", 16.0),
            params.get("medium", 8.0),
            params.get("coarse", 1.0),
            params.get("chunky", 0.0),
            luminance=bool(params.get("luminance", False)) and not is_grayscale,
        )


@dataclass
class WaveletDenoiseDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return wavelet_denoise(
            data, is_grayscale,
            params.get("fine", 3.0),
            params.get("medium", 1.0),
            params.get("coarse", 0.0),
        )


@dataclass
class AdaptiveDeconvDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        oklab = params.get("oklab", True) and not is_grayscale
        return adaptive_deconvolution(
            data, is_grayscale,
            params.get("amount", 10.0),
            params.get("adaptive", True),
            oklab,
        )


@dataclass
class WienerDeconvDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        oklab = params.get("oklab", True) and not is_grayscale
        return wiener_deconvolution(
            data, is_grayscale,
            params.get("amount", 10.0),
            params.get("adaptive", True),
            oklab,
        )


@dataclass
class MoonEnhanceDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return moon_enhance(
            data, is_grayscale,
            params.get("brightness", 25.0),
            params.get("sensitivity", 8.0),
            params.get("radius_scale", 0.5),
            params.get("planet_margin", 0.0),
            int(params.get("max_moons", 12)),
        )


@dataclass
class StretchContrastDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return stretch_contrast_oklab(data, params.get("amount", 100.0))


@dataclass
class ColourMatrixDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return apply_colour_matrix(data, matrix_from_params(params))


@dataclass
class SaturationVibranceDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return apply_saturation_vibrance(
            data,
            params.get("saturation", 1.0),
            params.get("vibrance", 1.0),
        )


@dataclass
class LevelsDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return apply_levels(data, params)


@dataclass
class MergeWaveletDetailDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        secondary = params.get("secondary_data")
        if secondary is None:
            return data
        return merge_wavelet_detail(
            data,
            secondary,
            is_grayscale,
            params.get("n_secondary_scales", 3),
        )


def scale_percents(params: dict[str, Any]) -> tuple[float, float]:
    """Return (width%, height%) for a Scale Image step."""
    percent = float(params.get("percent", 100.0))
    width_percent = float(params.get("width_percent", percent))
    height_percent = float(params.get("height_percent", percent))
    if bool(params.get("maintain_aspect", True)):
        scale = float(params.get("percent", width_percent))
        return scale, scale
    return width_percent, height_percent


@dataclass
class ScaleImageDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        del is_grayscale
        h, w = int(data.shape[0]), int(data.shape[1])
        wp, hp = scale_percents(params)
        new_w = max(1, int(round(w * wp / 100.0)))
        new_h = max(1, int(round(h * hp / 100.0)))
        return scale_image(data, new_w, new_h)


@dataclass
class RotateImageDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        del is_grayscale
        return rotate_image(
            data,
            float(params.get("angle", 0.0)),
            expand=True,
            crop_to_original=bool(params.get("crop_to_original", False)),
        )


def crop_step_summary(params: dict[str, Any]) -> str:
    """Short label for a Crop/Expand Image batch step."""
    if bool(params.get("autocrop", True)):
        border = int(params.get("border_px", DEFAULT_BORDER_PX))
        bright = float(params.get("min_brightness_pct", DEFAULT_MIN_BRIGHTNESS_PCT))
        return f"Autocrop, {border} px, {bright:g}%"
    width = int(params.get("width", 0))
    height = int(params.get("height", 0))
    ox = int(params.get("offset_x", 0))
    oy = int(params.get("offset_y", 0))
    if width <= 0 and height <= 0:
        size = "full"
    elif width <= 0:
        size = f"full × {height}"
    elif height <= 0:
        size = f"{width} × full"
    else:
        size = f"{width} × {height}"
    if ox or oy:
        return f"{size}, offset {ox}, {oy}"
    return size


@dataclass
class CropImageDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        img_h, img_w = int(arr.shape[0]), int(arr.shape[1])
        if bool(params.get("autocrop", True)):
            rect, _found = autocrop_rect(
                arr,
                is_grayscale,
                border_px=int(params.get("border_px", DEFAULT_BORDER_PX)),
                min_brightness_pct=float(
                    params.get("min_brightness_pct", DEFAULT_MIN_BRIGHTNESS_PCT)
                ),
            )
        else:
            crop_w = int(params.get("width", 0)) or img_w
            crop_h = int(params.get("height", 0)) or img_h
            rect = rect_from_size_offset(
                img_w,
                img_h,
                crop_w,
                crop_h,
                int(params.get("offset_x", 0)),
                int(params.get("offset_y", 0)),
            )
        if (
            rect.x == 0
            and rect.y == 0
            and rect.width == img_w
            and rect.height == img_h
        ):
            return arr
        return crop_image(arr, rect.x, rect.y, rect.width, rect.height)


# @dataclass
# class OklabLuminanceDef(FilterDef):
#     def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
#         return oklab_luminance(data)


def _with_defaults(default_params: dict[str, Any]) -> dict[str, Any]:
    clamp_default = default_params.get(
        CLAMP_PARAM,
        default_params.get(_LEGACY_CLAMP_PARAM, True),
    )
    return {
        **default_params,
        CLAMP_PARAM: clamp_default,
        CLAMP_LOW_PARAM: default_params.get(CLAMP_LOW_PARAM, False),
    }


FILTERS: dict[str, FilterDef] = {
    "scale_image": ScaleImageDef(
        id="scale_image",
        label="Scale Image",
        default_params={
            "percent": 100.0,
            "width_percent": 100.0,
            "height_percent": 100.0,
            "maintain_aspect": True,
        },
    ),
    "rotate_image": RotateImageDef(
        id="rotate_image",
        label="Rotate Image",
        default_params={
            "angle": 0.0,
            "crop_to_original": False,
        },
    ),
    "crop_image": CropImageDef(
        id="crop_image",
        label="Crop/Expand Image",
        default_params={
            "autocrop": True,
            "border_px": float(DEFAULT_BORDER_PX),
            "min_brightness_pct": float(DEFAULT_MIN_BRIGHTNESS_PCT),
            "width": 0.0,
            "height": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
    ),
    "wavelet_sharpen": WaveletSharpenDef(
        id="wavelet_sharpen",
        label="Wavelet Sharpen",
        default_params=_with_defaults({
            "fine": 16.0,
            "medium": 8.0,
            "coarse": 1.0,
            "chunky": 0.0,
            "luminance": False,
            "auto": False,
            "target_noise": 3.0,
            "target_contrast": 15.0,
        }),
    ),
    "wavelet_denoise": WaveletDenoiseDef(
        id="wavelet_denoise",
        label="Wavelet Denoise",
        default_params=_with_defaults({"fine": 3.0, "medium": 1.0, "coarse": 0.0}),
    ),
    "adaptive_deconv": AdaptiveDeconvDef(
        id="adaptive_deconv",
        label="Adaptive Deconvolution",
        default_params=_with_defaults({
            "amount": 10.0,
            "adaptive": True,
            "oklab": True,
            "auto": False,
            "target_noise": 3.5,
            "target_contrast": 15.0,
        }),
    ),
    "wiener_deconv": WienerDeconvDef(
        id="wiener_deconv",
        label="Wiener Deconvolution",
        # Kept for a future reimplementation; not listed in batch or the Enhance menu.
        batch_enabled=False,
        default_params=_with_defaults({"amount": 10.0, "adaptive": True, "oklab": True}),
    ),
    "moon_enhance": MoonEnhanceDef(
        id="moon_enhance",
        label="Moon Enhance",
        default_params=_with_defaults({
            "brightness": 25.0,
            "sensitivity": 8.0,
            "radius_scale": 0.5,
            # 0 = auto: max(20, 0.5 × planet equivalent radius) px.
            "planet_margin": 0.0,
            "max_moons": 12,
        }),
    ),
    "stretch_contrast": StretchContrastDef(
        id="stretch_contrast",
        label="Stretch Contrast OKLab",
        requires_rgb=True,
        default_params={"amount": 100.0},
    ),
    "colour_matrix": ColourMatrixDef(
        id="colour_matrix",
        label="Colour Correction Matrix",
        requires_rgb=True,
        default_params=_with_defaults({"matrix": [row[:] for row in IDENTITY_MATRIX]}),
    ),
    "saturation_vibrance": SaturationVibranceDef(
        id="saturation_vibrance",
        label="Saturation & Vibrance",
        requires_rgb=True,
        default_params=_with_defaults({"saturation": 1.0, "vibrance": 1.0}),
    ),
    "levels": LevelsDef(
        id="levels",
        label="Levels",
        requires_rgb=True,
        default_params={"channels": default_levels_params()},
    ),
    "merge_wavelet_detail": MergeWaveletDetailDef(
        id="merge_wavelet_detail",
        label="Merge Wavelet Detail",
        batch_enabled=False,
        default_params=_with_defaults({"n_secondary_scales": 3}),
    ),
    # "oklab_luminance": OklabLuminanceDef(
    #     id="oklab_luminance",
    #     label="OKLab Luminance",
    #     requires_rgb=True,
    #     default_params={},
    # ),
}

# Legacy American-spelling filter id.
_LEGACY_FILTER_IDS = {"color_matrix": "colour_matrix"}
FILTERS["color_matrix"] = FILTERS["colour_matrix"]


def _canonical_filter_id(filter_id: str) -> str:
    return _LEGACY_FILTER_IDS.get(filter_id, filter_id)


def _merge_params(fdef: FilterDef, params: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**fdef.default_params, **(params or {})}
    if CLAMP_PARAM not in merged and _LEGACY_CLAMP_PARAM in merged:
        merged[CLAMP_PARAM] = merged[_LEGACY_CLAMP_PARAM]
    return merged


def _clamp_high_enabled(params: dict[str, Any]) -> bool:
    return bool(params.get(CLAMP_PARAM, params.get(_LEGACY_CLAMP_PARAM, False)))


def _clamp_low_enabled(params: dict[str, Any]) -> bool:
    return bool(params.get(CLAMP_LOW_PARAM, False))


def run_filter_raw(
    filter_id: str,
    data: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run filter core without post-process clamping."""
    fdef = FILTERS[filter_id]
    merged = _merge_params(fdef, params)
    if fdef.requires_rgb and is_grayscale:
        raise ValueError(f"{fdef.label} requires an RGB image.")
    return fdef.apply(data, is_grayscale, merged)


def post_process(
    raw: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any],
    *,
    filter_id: str,
) -> np.ndarray:
    """Highlight clamping and automatic clip-black when clamp-to-0% is off."""
    clampable = _canonical_filter_id(filter_id) in CLAMP_FILTER_IDS
    clamp_high = clampable and _clamp_high_enabled(params)
    clamp_low = clampable and clamp_high and _clamp_low_enabled(params)
    clip_black = clampable and not clamp_low
    return apply_channel_post_process(
        raw,
        is_grayscale,
        clip_black=clip_black,
        clamp_high=clamp_high,
        clamp_low=clamp_low,
    )


def apply_filter(
    filter_id: str,
    data: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    merged = _merge_params(FILTERS[filter_id], params)
    raw = run_filter_raw(filter_id, data, is_grayscale, merged)
    return post_process(raw, is_grayscale, merged, filter_id=filter_id)


def output_brightness_info(
    filter_id: str,
    data: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any] | None = None,
) -> BrightnessInfo:
    """Channel min/max after filter, reflecting optional clamp correction."""
    return output_filter_stats(filter_id, data, is_grayscale, params).brightness


def output_filter_stats(
    filter_id: str,
    data: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any] | None = None,
    *,
    texture_scale: float | None = None,
    chromatic: bool | None = None,
) -> FilterOutputStats:
    """Pre-clip output levels and, for enhance filters, peak/noise metrics.

    Noise residual scales and chromatic mode should be the *session* source
    context (pinned on the document at load). Pass them explicitly so a
    sharpened document still scores with the original stack's PSF probes —
    otherwise re-estimating texture on the sharpened result drops the score
    (e.g. auto 2.99 → deconv 2.70 on crop.png).

    When omitted, both are estimated from ``data`` (batch / tests).
    """
    merged = _merge_params(FILTERS[filter_id], params)
    raw = run_filter_raw(filter_id, data, is_grayscale, merged)
    brightness = measure_brightness(raw, is_grayscale)

    increase: float | None = None
    noise: float | None = None
    source_noise: float | None = None
    if _canonical_filter_id(filter_id) in ENHANCE_FILTER_IDS:
        increase = brightness_increase_pct(data, raw, is_grayscale)
        if texture_scale is None:
            texture_scale = estimate_texture_scale(data, is_grayscale)
        if chromatic is None:
            chromatic = is_chromatic(data, is_grayscale)
        source_noise = absolute_noise(
            data,
            is_grayscale,
            texture_scale=texture_scale,
            chromatic=chromatic,
        )
        # Near-identity filters (e.g. all amounts 0): report source noise
        # exactly so enhance dialogs agree on the same document despite tiny
        # wavelet/deconv floating-point round-trip differences.
        raw_f = np.asarray(raw, dtype=np.float64)
        src_f = np.asarray(data, dtype=np.float64)
        if (
            source_noise is not None
            and raw_f.shape == src_f.shape
            and float(np.max(np.abs(raw_f - src_f))) < 1e-5
        ):
            noise = source_noise
        else:
            noise = absolute_noise(
                raw,
                is_grayscale,
                texture_scale=texture_scale,
                chromatic=chromatic,
            )

    return FilterOutputStats(brightness, increase, noise, source_noise)


def apply_filter_with_stats(
    filter_id: str,
    data: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, BrightnessInfo]:
    merged = _merge_params(FILTERS[filter_id], params)
    raw = run_filter_raw(filter_id, data, is_grayscale, merged)
    info = measure_brightness(raw, is_grayscale)
    result = post_process(raw, is_grayscale, merged, filter_id=filter_id)
    return result, info


def batch_filters() -> list[FilterDef]:
    """Filters available in batch (unique by id; aliases are not listed twice)."""
    seen: set[str] = set()
    out: list[FilterDef] = []
    for fdef in FILTERS.values():
        if not fdef.batch_enabled or fdef.id in seen:
            continue
        seen.add(fdef.id)
        out.append(fdef)
    return out


# Backwards-compatible export for UI modules still importing RESCALE_PARAM.
RESCALE_PARAM = CLAMP_PARAM