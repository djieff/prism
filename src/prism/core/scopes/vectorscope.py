"""Core vectorscope analysis helpers for viewer scope windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from prism.core.scopes.waveform_science import (
    DEFAULT_WAVEFORM_SIGNAL_STANDARD,
    WaveformSignalStandard,
    waveform_y_prime_coefficients,
)

VectorscopeMode = Literal["A", "B", "A|B"]


@dataclass(frozen=True)
class VectorscopeTrace:
    """Vectorscope chroma data for one image side."""

    x_values: np.ndarray
    y_values: np.ndarray
    density: np.ndarray
    color_density: np.ndarray
    source_size: tuple[int, int]
    signal_standard: WaveformSignalStandard
    y_prime_coefficients: tuple[float, float, float]
    plot_scale: float


@dataclass(frozen=True)
class VectorscopeViewData:
    """Mode-aware vectorscope payload for UI consumption."""

    mode: VectorscopeMode
    trace_a: VectorscopeTrace | None
    trace_b: VectorscopeTrace | None
    status: str


def build_vectorscope_trace(
    buffer_rgb: np.ndarray,
    *,
    density_size: int = 256,
    max_samples: int = 65536,
    signal_standard: WaveformSignalStandard = DEFAULT_WAVEFORM_SIGNAL_STANDARD,
) -> VectorscopeTrace:
    """Build normalized Cb/Cr vectorscope data from an RGB float buffer."""
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
        raise ValueError("Vectorscope buffer must contain only finite values")

    clipped = np.clip(sample, 0.0, 1.0)
    coefficients = waveform_y_prime_coefficients(signal_standard)
    kr = float(coefficients[0])
    kb = float(coefficients[2])
    plot_scale = vectorscope_chroma_scale(
        (float(coefficients[0]), float(coefficients[1]), float(coefficients[2]))
    )
    y_prime = (
        (coefficients[0] * clipped[:, 0])
        + (coefficients[1] * clipped[:, 1])
        + (coefficients[2] * clipped[:, 2])
    ).astype(np.float32)

    cb = np.float32(0.5) * ((clipped[:, 2] - y_prime) / np.float32(1.0 - kb))
    cr = np.float32(0.5) * ((clipped[:, 0] - y_prime) / np.float32(1.0 - kr))
    x_values = np.ascontiguousarray((cb * np.float32(2.0)) / np.float32(plot_scale), dtype=np.float32)
    y_values = np.ascontiguousarray((cr * np.float32(2.0)) / np.float32(plot_scale), dtype=np.float32)
    density, color_density = _density_from_chroma_points(
        x_values,
        y_values,
        clipped,
        density_size,
    )

    return VectorscopeTrace(
        x_values=x_values,
        y_values=y_values,
        density=density,
        color_density=color_density,
        source_size=(src_w, src_h),
        signal_standard=signal_standard,
        y_prime_coefficients=(
            float(coefficients[0]),
            float(coefficients[1]),
            float(coefficients[2]),
        ),
        plot_scale=plot_scale,
    )


def build_vectorscope_view_data(
    mode: VectorscopeMode,
    buffer_a: np.ndarray | None,
    buffer_b: np.ndarray | None,
    *,
    density_size: int = 256,
    max_samples: int = 65536,
    signal_standard: WaveformSignalStandard = DEFAULT_WAVEFORM_SIGNAL_STANDARD,
) -> VectorscopeViewData:
    """Build mode-specific vectorscope payload from side A/B buffers."""
    if mode == "A":
        if buffer_a is None:
            return VectorscopeViewData(
                mode=mode,
                trace_a=None,
                trace_b=None,
                status="A image not available",
            )
        return VectorscopeViewData(
            mode=mode,
            trace_a=build_vectorscope_trace(
                buffer_a,
                density_size=density_size,
                max_samples=max_samples,
                signal_standard=signal_standard,
            ),
            trace_b=None,
            status="Vectorscope: A",
        )

    if mode == "B":
        if buffer_b is None:
            return VectorscopeViewData(
                mode=mode,
                trace_a=None,
                trace_b=None,
                status="B image not available",
            )
        return VectorscopeViewData(
            mode=mode,
            trace_a=None,
            trace_b=build_vectorscope_trace(
                buffer_b,
                density_size=density_size,
                max_samples=max_samples,
                signal_standard=signal_standard,
            ),
            status="Vectorscope: B",
        )

    trace_a = (
        build_vectorscope_trace(
            buffer_a,
            density_size=density_size,
            max_samples=max_samples,
            signal_standard=signal_standard,
        )
        if buffer_a is not None
        else None
    )
    trace_b = (
        build_vectorscope_trace(
            buffer_b,
            density_size=density_size,
            max_samples=max_samples,
            signal_standard=signal_standard,
        )
        if buffer_b is not None
        else None
    )

    if trace_a is not None and trace_b is not None:
        status = "Vectorscope: A|B"
    elif trace_a is not None:
        status = "Vectorscope: A (B missing)"
    elif trace_b is not None:
        status = "Vectorscope: B (A missing)"
    else:
        status = "Missing image data: A, B"

    return VectorscopeViewData(
        mode=mode,
        trace_a=trace_a,
        trace_b=trace_b,
        status=status,
    )


def _sample_rgb_pixels(buffer_rgb: np.ndarray, *, max_samples: int) -> np.ndarray:
    flat = np.reshape(buffer_rgb, (-1, 3))
    sample_count = flat.shape[0]
    if sample_count <= max_samples:
        return flat
    indices = np.linspace(0, sample_count - 1, max_samples, dtype=np.int64)
    return flat[indices, :]


def vectorscope_chroma_scale(coefficients: tuple[float, float, float]) -> float:
    """Return radius scale that fits full-saturation RGB/cyan/magenta/yellow targets."""
    targets = tuple(
        _chroma_for_rgb(rgb, coefficients)
        for rgb in (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        )
    )
    return max(float(np.hypot(x, y)) for x, y in targets)


def _chroma_for_rgb(
    rgb: tuple[float, float, float],
    coefficients: tuple[float, float, float],
) -> tuple[float, float]:
    r, g, b = rgb
    kr, kg, kb = coefficients
    y_prime = (kr * r) + (kg * g) + (kb * b)
    cb = 0.5 * ((b - y_prime) / (1.0 - kb))
    cr = 0.5 * ((r - y_prime) / (1.0 - kr))
    return cb * 2.0, cr * 2.0


def _density_from_chroma_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    rgb_values: np.ndarray,
    density_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_idx = np.rint(((x_values + np.float32(1.0)) * np.float32(0.5)) * float(density_size - 1))
    y_idx = np.rint(((np.float32(1.0) - ((y_values + np.float32(1.0)) * np.float32(0.5))) * float(density_size - 1)))
    x_idx = np.clip(x_idx, 0, density_size - 1).astype(np.int64)
    y_idx = np.clip(y_idx, 0, density_size - 1).astype(np.int64)

    density = np.zeros((density_size, density_size), dtype=np.float32)
    color_sum = np.zeros((density_size, density_size, 3), dtype=np.float32)
    np.add.at(density, (y_idx, x_idx), np.float32(1.0))
    np.add.at(color_sum, (y_idx, x_idx), rgb_values)

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
