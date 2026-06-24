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
from planetary_tools.filters.adaptive_deconv import adaptive_deconvolution
from planetary_tools.filters.color_matrix import (
    IDENTITY_MATRIX,
    apply_color_matrix,
    matrix_from_params,
)
# from planetary_tools.filters.oklab_filters import oklab_luminance
from planetary_tools.filters.levels import apply_levels, default_levels_params
from planetary_tools.filters.saturation import apply_saturation_vibrance
from planetary_tools.filters.stretch import stretch_contrast_oklab
from planetary_tools.filters.wavelet import wavelet_denoise, wavelet_sharpen

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
})

CLAMP_FILTER_IDS = ENHANCE_FILTER_IDS | frozenset({
    "color_matrix",
    "saturation_vibrance",
})


@dataclass(frozen=True)
class FilterOutputStats:
    brightness: BrightnessInfo
    brightness_increase_pct: float | None = None


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
class StretchContrastDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return stretch_contrast_oklab(data)


@dataclass
class ColorMatrixDef(FilterDef):
    def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
        return apply_color_matrix(data, matrix_from_params(params))


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


# @dataclass
# class OklabLuminanceDef(FilterDef):
#     def apply(self, data: np.ndarray, is_grayscale: bool, params: dict[str, Any]) -> np.ndarray:
#         return oklab_luminance(data)


def _with_defaults(default_params: dict[str, Any]) -> dict[str, Any]:
    clamp_default = default_params.get(
        CLAMP_PARAM,
        default_params.get(_LEGACY_CLAMP_PARAM, False),
    )
    return {
        **default_params,
        CLAMP_PARAM: clamp_default,
        CLAMP_LOW_PARAM: default_params.get(CLAMP_LOW_PARAM, False),
    }


FILTERS: dict[str, FilterDef] = {
    "wavelet_sharpen": WaveletSharpenDef(
        id="wavelet_sharpen",
        label="Wavelet Sharpen",
        default_params=_with_defaults({"fine": 16.0, "medium": 8.0, "coarse": 1.0}),
    ),
    "wavelet_denoise": WaveletDenoiseDef(
        id="wavelet_denoise",
        label="Wavelet Denoise",
        default_params=_with_defaults({"fine": 3.0, "medium": 1.0, "coarse": 0.0}),
    ),
    "adaptive_deconv": AdaptiveDeconvDef(
        id="adaptive_deconv",
        label="Adaptive Deconvolution",
        default_params=_with_defaults({"amount": 10.0, "adaptive": True, "oklab": True}),
    ),
    "stretch_contrast": StretchContrastDef(
        id="stretch_contrast",
        label="Stretch Contrast OKLab",
        requires_rgb=True,
        default_params={},
    ),
    "color_matrix": ColorMatrixDef(
        id="color_matrix",
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
    # "oklab_luminance": OklabLuminanceDef(
    #     id="oklab_luminance",
    #     label="OKLab Luminance",
    #     requires_rgb=True,
    #     default_params={},
    # ),
}


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
    clampable = filter_id in CLAMP_FILTER_IDS
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
) -> FilterOutputStats:
    """Pre-clip output levels and, for enhance filters, peak increase."""
    merged = _merge_params(FILTERS[filter_id], params)
    raw = run_filter_raw(filter_id, data, is_grayscale, merged)
    brightness = measure_brightness(raw, is_grayscale)

    increase: float | None = None
    if filter_id in ENHANCE_FILTER_IDS:
        increase = brightness_increase_pct(data, raw, is_grayscale)

    return FilterOutputStats(brightness, increase)


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
    return [f for f in FILTERS.values() if f.batch_enabled]


# Backwards-compatible export for UI modules still importing RESCALE_PARAM.
RESCALE_PARAM = CLAMP_PARAM