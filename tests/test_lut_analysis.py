"""Tests for LUT inspection analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from prism.core.lut_analysis import summarize_lut_samples


def test_summarize_lut_samples_reports_min_max_range_and_monotonicity() -> None:
    x = np.asarray([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    y = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.75, 0.25, 0.5],
            [0.5, 0.75, 0.25],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_lut_samples(x, y, channels=3)

    assert summary is not None
    assert summary.sample_count == 4
    assert summary.channel_count == 3
    assert summary.output_min == (0.0, 0.0, 0.0)
    assert summary.output_max == (1.0, 1.0, 1.0)
    assert summary.has_values_below_zero is False
    assert summary.has_values_above_one is False
    assert summary.monotonic_channels == (False, True, False)


def test_summarize_lut_samples_detects_values_outside_unit_range() -> None:
    x = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    y = np.asarray(
        [
            [-0.1, 0.0, 0.0],
            [0.5, 1.2, 0.5],
            [1.0, 1.0, 1.1],
        ],
        dtype=np.float32,
    )

    summary = summarize_lut_samples(x, y, channels=3)

    assert summary is not None
    assert summary.has_values_below_zero is True
    assert summary.has_values_above_one is True
    assert summary.output_min == pytest.approx((-0.1, 0.0, 0.0))
    assert summary.output_max == pytest.approx((1.0, 1.2, 1.1))


def test_summarize_lut_samples_clamps_channel_count_to_available_values() -> None:
    x = np.asarray([0.0, 1.0], dtype=np.float32)
    y = np.asarray([[0.25], [0.75]], dtype=np.float32)

    summary = summarize_lut_samples(x, y, channels=3)

    assert summary is not None
    assert summary.channel_count == 1
    assert summary.output_min == (0.25,)
    assert summary.output_max == (0.75,)
    assert summary.monotonic_channels == (True,)


def test_summarize_lut_samples_returns_none_for_empty_data() -> None:
    x = np.asarray([], dtype=np.float32)
    y = np.asarray([], dtype=np.float32)

    assert summarize_lut_samples(x, y, channels=3) is None


def test_summarize_lut_samples_rejects_non_2d_values() -> None:
    x = np.asarray([0.0, 1.0], dtype=np.float32)
    y = np.asarray([0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="two-dimensional"):
        summarize_lut_samples(x, y, channels=1)
