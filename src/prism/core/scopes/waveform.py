"""Core waveform analysis helpers for viewer scope windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from prism.core.scopes.waveform_science import (
    DEFAULT_WAVEFORM_SIGNAL_STANDARD,
    WaveformSignalStandard,
    waveform_y_prime_coefficients,
)

WaveformMode = Literal["A", "B", "A|B"]


@dataclass(frozen=True)
class WaveformTrace:
    """Waveform density data for one image side."""

    x_values: np.ndarray
    density_r: np.ndarray
    density_g: np.ndarray
    density_b: np.ndarray
    density_luma: np.ndarray  # Encoded Y' density; legacy field name retained.
    source_size: tuple[int, int]
    signal_standard: WaveformSignalStandard
    y_prime_coefficients: tuple[float, float, float]


@dataclass(frozen=True)
class WaveformViewData:
    """Mode-aware waveform payload for UI consumption."""

    mode: WaveformMode
    trace_a: WaveformTrace | None
    trace_b: WaveformTrace | None
    status: str


def build_waveform_trace(
    buffer_rgb: np.ndarray,
    *,
    sample_width: int = 512,
    sample_height: int = 256,
    signal_standard: WaveformSignalStandard = DEFAULT_WAVEFORM_SIGNAL_STANDARD,
) -> WaveformTrace:
    """Build normalized waveform density grids from an RGB float buffer."""
    if sample_width <= 0 or sample_height <= 0:
        raise ValueError("sample_width and sample_height must be positive")
    if buffer_rgb.ndim != 3 or buffer_rgb.shape[2] != 3:
        raise ValueError("Expected image buffer shape (H, W, 3)")

    src_h, src_w = int(buffer_rgb.shape[0]), int(buffer_rgb.shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Expected non-empty image dimensions")

    work = np.asarray(buffer_rgb, dtype=np.float32)
    x_idx = np.linspace(0, src_w - 1, sample_width, dtype=np.int32)
    sampled = work[:, x_idx, :]
    clipped = np.clip(sampled, 0.0, 1.0)
    y_idx = np.rint(clipped * float(sample_height - 1)).astype(np.int32)
    row_idx = (sample_height - 1) - y_idx

    density_r = _density_from_row_indices(row_idx[:, :, 0], sample_height, sample_width)
    density_g = _density_from_row_indices(row_idx[:, :, 1], sample_height, sample_width)
    density_b = _density_from_row_indices(row_idx[:, :, 2], sample_height, sample_width)
    y_prime_coefficients = waveform_y_prime_coefficients(signal_standard)
    y_prime = (
        (y_prime_coefficients[0] * clipped[:, :, 0])
        + (y_prime_coefficients[1] * clipped[:, :, 1])
        + (y_prime_coefficients[2] * clipped[:, :, 2])
    ).astype(np.float32)
    y_prime_bins = np.rint(y_prime * float(sample_height - 1)).astype(np.int32)
    y_prime_rows = (sample_height - 1) - y_prime_bins
    density_luma = _density_from_row_indices(y_prime_rows, sample_height, sample_width)
    max_count = float(
        max(
            np.max(density_r),
            np.max(density_g),
            np.max(density_b),
            np.max(density_luma),
        )
    )
    if max_count > 0.0:
        inv = np.float32(1.0 / max_count)
        density_r = density_r * inv
        density_g = density_g * inv
        density_b = density_b * inv
        density_luma = density_luma * inv
    x_values = np.linspace(0.0, 1.0, sample_width, dtype=np.float32)

    return WaveformTrace(
        x_values=x_values,
        density_r=density_r,
        density_g=density_g,
        density_b=density_b,
        density_luma=density_luma,
        source_size=(src_w, src_h),
        signal_standard=signal_standard,
        y_prime_coefficients=(
            float(y_prime_coefficients[0]),
            float(y_prime_coefficients[1]),
            float(y_prime_coefficients[2]),
        ),
    )


def build_waveform_view_data(
    mode: WaveformMode,
    buffer_a: np.ndarray | None,
    buffer_b: np.ndarray | None,
    *,
    sample_width: int = 512,
    sample_height: int = 256,
    signal_standard: WaveformSignalStandard = DEFAULT_WAVEFORM_SIGNAL_STANDARD,
) -> WaveformViewData:
    """Build mode-specific waveform payload from side A/B buffers."""
    if mode == "A":
        if buffer_a is None:
            return WaveformViewData(mode=mode, trace_a=None, trace_b=None, status="A image not available")
        return WaveformViewData(
            mode=mode,
            trace_a=build_waveform_trace(
                buffer_a,
                sample_width=sample_width,
                sample_height=sample_height,
                signal_standard=signal_standard,
            ),
            trace_b=None,
            status="Waveform: A",
        )

    if mode == "B":
        if buffer_b is None:
            return WaveformViewData(mode=mode, trace_a=None, trace_b=None, status="B image not available")
        return WaveformViewData(
            mode=mode,
            trace_a=None,
            trace_b=build_waveform_trace(
                buffer_b,
                sample_width=sample_width,
                sample_height=sample_height,
                signal_standard=signal_standard,
            ),
            status="Waveform: B",
        )

    trace_a = (
        build_waveform_trace(
            buffer_a,
            sample_width=sample_width,
            sample_height=sample_height,
            signal_standard=signal_standard,
        )
        if buffer_a is not None
        else None
    )
    trace_b = (
        build_waveform_trace(
            buffer_b,
            sample_width=sample_width,
            sample_height=sample_height,
            signal_standard=signal_standard,
        )
        if buffer_b is not None
        else None
    )

    if trace_a is not None and trace_b is not None:
        status = "Waveform: A|B"
    elif trace_a is not None:
        status = "Waveform: A (B missing)"
    elif trace_b is not None:
        status = "Waveform: B (A missing)"
    else:
        status = "Missing image data: A, B"

    return WaveformViewData(
        mode=mode,
        trace_a=trace_a,
        trace_b=trace_b,
        status=status,
    )


def _density_from_row_indices(
    row_indices: np.ndarray, sample_height: int, sample_width: int
) -> np.ndarray:
    """Return raw per-bin counts for one channel's row indices."""
    density = np.zeros((sample_height, sample_width), dtype=np.float32)
    for col in range(sample_width):
        counts = np.bincount(row_indices[:, col], minlength=sample_height)
        if counts.sum() > 0:
            density[:, col] = counts.astype(np.float32)
    return density
