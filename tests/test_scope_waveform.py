"""Tests for waveform scope core helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from prism.core.scopes.waveform import (
    WaveformMode,
    build_waveform_trace,
    build_waveform_view_data,
)
from prism.core.scopes.waveform_science import DEFAULT_WAVEFORM_SIGNAL_STANDARD


def test_build_waveform_trace_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected image buffer shape"):
        build_waveform_trace(np.zeros((4, 4), dtype=np.float32))


def test_build_waveform_trace_shape_and_source_size() -> None:
    buffer_rgb = np.zeros((3, 5, 3), dtype=np.float32)
    trace = build_waveform_trace(buffer_rgb, sample_width=4, sample_height=6)

    assert trace.x_values.shape == (4,)
    assert trace.density_r.shape == (6, 4)
    assert trace.density_g.shape == (6, 4)
    assert trace.density_b.shape == (6, 4)
    assert trace.density_luma.shape == (6, 4)
    assert trace.source_size == (5, 3)
    assert trace.signal_standard == DEFAULT_WAVEFORM_SIGNAL_STANDARD
    assert np.allclose(trace.y_prime_coefficients, (0.2126, 0.7152, 0.0722))


def test_build_waveform_trace_maps_black_to_bottom_and_white_to_top() -> None:
    black = np.zeros((4, 4, 3), dtype=np.float32)
    white = np.ones((4, 4, 3), dtype=np.float32)

    black_trace = build_waveform_trace(black, sample_width=4, sample_height=8)
    white_trace = build_waveform_trace(white, sample_width=4, sample_height=8)

    assert np.allclose(black_trace.density_r[-1, :], 1.0)
    assert np.allclose(black_trace.density_g[-1, :], 1.0)
    assert np.allclose(black_trace.density_b[-1, :], 1.0)
    assert np.allclose(black_trace.density_luma[-1, :], 1.0)
    assert np.allclose(white_trace.density_r[0, :], 1.0)
    assert np.allclose(white_trace.density_g[0, :], 1.0)
    assert np.allclose(white_trace.density_b[0, :], 1.0)
    assert np.allclose(white_trace.density_luma[0, :], 1.0)


def test_build_waveform_trace_default_bt709_matches_legacy_y_prime_bins() -> None:
    buffer_rgb = np.asarray(
        [
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.25, 0.5, 0.75),
                (1.0, 1.0, 1.0),
            ]
        ],
        dtype=np.float32,
    )
    sample_height = 1001

    trace = build_waveform_trace(
        buffer_rgb,
        sample_width=buffer_rgb.shape[1],
        sample_height=sample_height,
    )
    legacy_y_prime = (
        (0.2126 * buffer_rgb[:, :, 0])
        + (0.7152 * buffer_rgb[:, :, 1])
        + (0.0722 * buffer_rgb[:, :, 2])
    ).astype(np.float32)
    expected_rows = (sample_height - 1) - np.rint(
        legacy_y_prime * float(sample_height - 1)
    ).astype(np.int32)

    for column, row in enumerate(expected_rows[0]):
        assert trace.density_luma[row, column] == pytest.approx(1.0)
        assert np.count_nonzero(trace.density_luma[:, column]) == 1


def test_build_waveform_trace_bt2020_uses_selected_y_prime_coefficients() -> None:
    buffer_rgb = np.asarray(
        [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]],
        dtype=np.float32,
    )
    sample_height = 10001

    trace = build_waveform_trace(
        buffer_rgb,
        sample_width=3,
        sample_height=sample_height,
        signal_standard="ITU-R BT.2020",
    )

    assert trace.signal_standard == "ITU-R BT.2020"
    assert np.allclose(trace.y_prime_coefficients, (0.2627, 0.6780, 0.0593))
    expected_bins = np.rint(
        np.asarray(trace.y_prime_coefficients, dtype=np.float32)
        * float(sample_height - 1)
    ).astype(np.int32)
    for column, signal_bin in enumerate(expected_bins):
        row = (sample_height - 1) - signal_bin
        assert trace.density_luma[row, column] == pytest.approx(1.0)


def test_changing_signal_standard_changes_only_y_prime_density() -> None:
    buffer_rgb = np.random.default_rng(23).random((12, 16, 3), dtype=np.float32)

    bt709 = build_waveform_trace(
        buffer_rgb,
        sample_width=16,
        sample_height=64,
        signal_standard="ITU-R BT.709",
    )
    bt2020 = build_waveform_trace(
        buffer_rgb,
        sample_width=16,
        sample_height=64,
        signal_standard="ITU-R BT.2020",
    )

    assert np.array_equal(bt709.density_r, bt2020.density_r)
    assert np.array_equal(bt709.density_g, bt2020.density_g)
    assert np.array_equal(bt709.density_b, bt2020.density_b)
    assert not np.array_equal(bt709.density_luma, bt2020.density_luma)


def test_build_waveform_view_data_mode_a_uses_only_a() -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    b = np.ones((2, 2, 3), dtype=np.float32)
    view = build_waveform_view_data("A", a, b, sample_width=2, sample_height=4)

    assert view.mode == "A"
    assert view.trace_a is not None
    assert view.trace_b is None
    assert view.status == "Waveform: A"


def test_build_waveform_view_data_mode_b_missing_buffer() -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    view = build_waveform_view_data("B", a, None, sample_width=2, sample_height=4)

    assert view.mode == "B"
    assert view.trace_a is None
    assert view.trace_b is None
    assert view.status == "B image not available"


def test_build_waveform_view_data_mode_ab_requires_both_buffers() -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    view = build_waveform_view_data("A|B", a, None, sample_width=2, sample_height=4)

    assert view.mode == "A|B"
    assert view.trace_a is not None
    assert view.trace_b is None
    assert view.status == "Waveform: A (B missing)"


@pytest.mark.parametrize("mode", ["A", "B", "A|B"])
def test_build_waveform_view_data_threads_signal_standard(mode: str) -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    b = np.ones((2, 2, 3), dtype=np.float32)

    view = build_waveform_view_data(
        cast(WaveformMode, mode),
        a,
        b,
        sample_width=2,
        sample_height=4,
        signal_standard="ITU-R BT.2020",
    )

    traces = [trace for trace in (view.trace_a, view.trace_b) if trace is not None]
    assert traces
    assert all(trace.signal_standard == "ITU-R BT.2020" for trace in traces)
