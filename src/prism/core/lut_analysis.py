"""Core analysis helpers for LUT inspection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MONOTONIC_TOLERANCE = -1e-6


@dataclass(frozen=True)
class LutAnalysisSummary:
    """UI-agnostic summary of plotted LUT sample data."""

    sample_count: int
    channel_count: int
    output_min: tuple[float, ...]
    output_max: tuple[float, ...]
    has_values_below_zero: bool
    has_values_above_one: bool
    monotonic_channels: tuple[bool, ...]


def summarize_lut_samples(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    channels: int,
) -> LutAnalysisSummary | None:
    """Return summary metrics for LUT plot samples, or `None` for empty data."""
    if x_values.size == 0 or y_values.size == 0:
        return None
    if y_values.ndim != 2:
        raise ValueError("LUT sample values must be a two-dimensional array")

    channel_count = min(max(int(channels), 1), int(y_values.shape[1]))
    y_used = y_values[:, :channel_count]
    if y_used.size == 0:
        return None

    y_min = np.min(y_used, axis=0)
    y_max = np.max(y_used, axis=0)
    diffs = np.diff(y_used, axis=0)

    return LutAnalysisSummary(
        sample_count=int(x_values.shape[0]),
        channel_count=channel_count,
        output_min=tuple(float(value) for value in y_min),
        output_max=tuple(float(value) for value in y_max),
        has_values_below_zero=bool(np.any(y_used < 0.0)),
        has_values_above_one=bool(np.any(y_used > 1.0)),
        monotonic_channels=tuple(
            bool(np.all(diffs[:, channel] >= MONOTONIC_TOLERANCE))
            for channel in range(channel_count)
        ),
    )
