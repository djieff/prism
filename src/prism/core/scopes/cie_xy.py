"""Core CIE 1931 xy chromaticity analysis helpers for viewer scope windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from warnings import filterwarnings

import numpy as np

filterwarnings(
    "ignore",
    message='"Matplotlib" related API features are not available:.*',
    module="colour.utilities.verbose",
)
from colour.colorimetry import CCS_ILLUMINANTS, MSDS_CMFS  # noqa: E402
from colour.models import RGB_COLOURSPACES  # noqa: E402

CieXyMode = Literal["A", "B", "A|B"]

DEFAULT_CIE_RGB_SPACE = "sRGB"
CIE_XY_X_RANGE = (0.0, 0.8)
CIE_XY_Y_RANGE = (0.0, 0.9)

PREFERRED_CIE_WHITEPOINTS: tuple[str, ...] = ("D50", "D55", "D60", "D65", "D75", "DCI-P3")


@dataclass(frozen=True)
class CieRgbSpace:
    """RGB colourspace data needed for RGB-to-XYZ conversion."""

    name: str
    primaries_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    whitepoint_xy: tuple[float, float]
    rgb_to_xyz: np.ndarray


@dataclass(frozen=True)
class CieGamutOverlay:
    """Reference RGB gamut boundary for CIE xy rendering."""

    name: str
    primaries_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    whitepoint_xy: tuple[float, float]


@dataclass(frozen=True)
class CieWhitePoint:
    """Reference white point marker for CIE xy rendering."""

    name: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class CieXyTrace:
    """CIE xy chromaticity data for one image side."""

    x_values: np.ndarray
    y_values: np.ndarray
    density: np.ndarray
    color_density: np.ndarray
    source_size: tuple[int, int]
    rgb_space_name: str


@dataclass(frozen=True)
class CieXyViewData:
    """Mode-aware CIE xy payload for UI consumption."""

    mode: CieXyMode
    trace_a: CieXyTrace | None
    trace_b: CieXyTrace | None
    status: str


def available_cie_rgb_space_names() -> tuple[str, ...]:
    """Return package-backed RGB space names in alphabetical order."""
    return tuple(sorted(RGB_COLOURSPACES.keys(), key=str.casefold))


def available_cie_whitepoint_names() -> tuple[str, ...]:
    """Return common CIE 1931 2-degree white point names."""
    illuminants = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]
    return tuple(name for name in PREFERRED_CIE_WHITEPOINTS if name in illuminants)


def get_cie_whitepoint(name: str) -> CieWhitePoint:
    """Return a CIE 1931 2-degree white point from Colour Science."""
    illuminants = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]
    if name not in illuminants:
        raise ValueError(f"Unsupported CIE white point: {name!r}")
    return CieWhitePoint(name=name, xy=_xy_tuple(np.asarray(illuminants[name], dtype=np.float64)))


def get_cie_rgb_space(name: str) -> CieRgbSpace:
    """Return RGB colourspace data from Colour Science."""
    if name not in RGB_COLOURSPACES:
        raise ValueError(f"Unsupported CIE RGB space: {name!r}")
    colourspace = RGB_COLOURSPACES[name]
    primaries = _xy_tuple3(np.asarray(colourspace.primaries, dtype=np.float64))
    whitepoint = _xy_tuple(np.asarray(colourspace.whitepoint, dtype=np.float64))
    matrix = np.ascontiguousarray(colourspace.matrix_RGB_to_XYZ, dtype=np.float32)
    return CieRgbSpace(
        name=str(colourspace.name),
        primaries_xy=primaries,
        whitepoint_xy=whitepoint,
        rgb_to_xyz=matrix,
    )


def available_cie_gamut_overlays() -> tuple[CieGamutOverlay, ...]:
    """Return package-backed gamut overlays in alphabetical order."""
    return tuple(
        CieGamutOverlay(
            name=space.name,
            primaries_xy=space.primaries_xy,
            whitepoint_xy=space.whitepoint_xy,
        )
        for space in (get_cie_rgb_space(name) for name in available_cie_rgb_space_names())
    )


def spectral_locus_xy() -> np.ndarray:
    """Return CIE 1931 2-degree spectral locus xy coordinates."""
    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    xyz = np.asarray(cmfs.values, dtype=np.float64)
    denom = np.sum(xyz, axis=1)
    valid = np.isfinite(denom) & (denom > 0.0)
    xy = xyz[valid, :2] / denom[valid, None]
    return np.ascontiguousarray(xy, dtype=np.float32)


def build_cie_xy_trace(
    buffer_rgb: np.ndarray,
    *,
    rgb_space_name: str = DEFAULT_CIE_RGB_SPACE,
    density_size: int = 256,
    max_samples: int = 65536,
) -> CieXyTrace:
    """Build CIE xy chromaticity data from an RGB float buffer."""
    if density_size <= 0:
        raise ValueError("density_size must be positive")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if buffer_rgb.ndim != 3 or buffer_rgb.shape[2] != 3:
        raise ValueError("Expected image buffer shape (H, W, 3)")

    src_h, src_w = int(buffer_rgb.shape[0]), int(buffer_rgb.shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Expected non-empty image dimensions")

    work = np.asarray(buffer_rgb, dtype=np.float32)
    sample = _sample_rgb_pixels(work, max_samples=max_samples)
    if not np.all(np.isfinite(sample)):
        raise ValueError("CIE xy buffer must contain only finite values")

    rgb_space = get_cie_rgb_space(rgb_space_name)
    clipped = np.clip(sample, 0.0, 1.0)
    xyz = clipped @ rgb_space.rgb_to_xyz.T
    denom = np.sum(xyz, axis=1)
    valid = np.isfinite(denom) & (denom > 0.0)
    x_values = np.ascontiguousarray(xyz[valid, 0] / denom[valid], dtype=np.float32)
    y_values = np.ascontiguousarray(xyz[valid, 1] / denom[valid], dtype=np.float32)
    density, color_density = _density_from_xy_points(
        x_values,
        y_values,
        clipped[valid],
        density_size,
    )

    return CieXyTrace(
        x_values=x_values,
        y_values=y_values,
        density=density,
        color_density=color_density,
        source_size=(src_w, src_h),
        rgb_space_name=rgb_space.name,
    )


def build_cie_xy_view_data(
    mode: CieXyMode,
    buffer_a: np.ndarray | None,
    buffer_b: np.ndarray | None,
    *,
    rgb_space_name: str = DEFAULT_CIE_RGB_SPACE,
    density_size: int = 256,
    max_samples: int = 65536,
) -> CieXyViewData:
    """Build mode-specific CIE xy payload from side A/B buffers."""
    if mode == "A":
        if buffer_a is None:
            return CieXyViewData(mode=mode, trace_a=None, trace_b=None, status="A image not available")
        return CieXyViewData(
            mode=mode,
            trace_a=build_cie_xy_trace(
                buffer_a,
                rgb_space_name=rgb_space_name,
                density_size=density_size,
                max_samples=max_samples,
            ),
            trace_b=None,
            status="CIE xy: A",
        )

    if mode == "B":
        if buffer_b is None:
            return CieXyViewData(mode=mode, trace_a=None, trace_b=None, status="B image not available")
        return CieXyViewData(
            mode=mode,
            trace_a=None,
            trace_b=build_cie_xy_trace(
                buffer_b,
                rgb_space_name=rgb_space_name,
                density_size=density_size,
                max_samples=max_samples,
            ),
            status="CIE xy: B",
        )

    trace_a = (
        build_cie_xy_trace(
            buffer_a,
            rgb_space_name=rgb_space_name,
            density_size=density_size,
            max_samples=max_samples,
        )
        if buffer_a is not None
        else None
    )
    trace_b = (
        build_cie_xy_trace(
            buffer_b,
            rgb_space_name=rgb_space_name,
            density_size=density_size,
            max_samples=max_samples,
        )
        if buffer_b is not None
        else None
    )

    if trace_a is not None and trace_b is not None:
        status = "CIE xy: A|B"
    elif trace_a is not None:
        status = "CIE xy: A (B missing)"
    elif trace_b is not None:
        status = "CIE xy: B (A missing)"
    else:
        status = "Missing image data: A, B"

    return CieXyViewData(
        mode=mode,
        trace_a=trace_a,
        trace_b=trace_b,
        status=status,
    )


def unique_cie_gamut_overlay_names(names: tuple[str | None, ...]) -> tuple[str, ...]:
    """Return supported overlay names once, preserving first-selection order."""
    unique: list[str] = []
    for name in names:
        if not name or name in unique:
            continue
        if name not in RGB_COLOURSPACES:
            raise ValueError(f"Unsupported CIE gamut overlay: {name!r}")
        unique.append(name)
    return tuple(unique)


def _sample_rgb_pixels(buffer_rgb: np.ndarray, *, max_samples: int) -> np.ndarray:
    flat = np.reshape(buffer_rgb, (-1, 3))
    sample_count = flat.shape[0]
    if sample_count <= max_samples:
        return flat
    indices = np.linspace(0, sample_count - 1, max_samples, dtype=np.int64)
    return flat[indices, :]


def _density_from_xy_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    rgb_values: np.ndarray,
    density_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = CIE_XY_X_RANGE
    y_min, y_max = CIE_XY_Y_RANGE
    in_bounds = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values >= x_min)
        & (x_values <= x_max)
        & (y_values >= y_min)
        & (y_values <= y_max)
    )
    x_in = x_values[in_bounds]
    y_in = y_values[in_bounds]
    rgb_in = rgb_values[in_bounds]

    density = np.zeros((density_size, density_size), dtype=np.float32)
    color_sum = np.zeros((density_size, density_size, 3), dtype=np.float32)
    if x_in.size == 0:
        return (
            np.ascontiguousarray(density, dtype=np.float32),
            np.ascontiguousarray(color_sum, dtype=np.float32),
        )

    x_idx = np.rint(((x_in - x_min) / (x_max - x_min)) * float(density_size - 1))
    y_idx = np.rint((1.0 - ((y_in - y_min) / (y_max - y_min))) * float(density_size - 1))
    x_idx = np.clip(x_idx, 0, density_size - 1).astype(np.int64)
    y_idx = np.clip(y_idx, 0, density_size - 1).astype(np.int64)

    np.add.at(density, (y_idx, x_idx), np.float32(1.0))
    np.add.at(color_sum, (y_idx, x_idx), rgb_in)

    color_density = np.zeros_like(color_sum)
    occupied = density > 0.0
    color_density[occupied] = color_sum[occupied] / density[occupied, None]

    max_count = float(np.max(density))
    if max_count > 0.0:
        density = density * np.float32(1.0 / max_count)
        color_density = color_density * density[:, :, None]
    return (
        np.ascontiguousarray(density, dtype=np.float32),
        np.ascontiguousarray(color_density, dtype=np.float32),
    )


def _xy_tuple(values: np.ndarray) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _xy_tuple3(values: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return (_xy_tuple(values[0]), _xy_tuple(values[1]), _xy_tuple(values[2]))
