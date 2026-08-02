"""Tests for vectorscope scope core helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from prism.core.scopes.vectorscope import (
    VectorscopeMode,
    build_vectorscope_trace,
    build_vectorscope_view_data,
    vectorscope_chroma_scale,
)
from prism.core.scopes.waveform_science import DEFAULT_WAVEFORM_SIGNAL_STANDARD


def test_build_vectorscope_trace_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected image buffer shape"):
        build_vectorscope_trace(np.zeros((4, 4), dtype=np.float32))


def test_build_vectorscope_trace_rejects_non_finite_values() -> None:
    buffer_rgb = np.zeros((2, 2, 3), dtype=np.float32)
    buffer_rgb[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_vectorscope_trace(buffer_rgb)


def test_build_vectorscope_trace_shape_and_source_size() -> None:
    buffer_rgb = np.zeros((3, 5, 3), dtype=np.float32)
    trace = build_vectorscope_trace(buffer_rgb, density_size=16)

    assert trace.x_values.shape == (15,)
    assert trace.y_values.shape == (15,)
    assert trace.density.shape == (16, 16)
    assert trace.color_density.shape == (16, 16, 3)
    assert trace.density.dtype == np.float32
    assert trace.color_density.dtype == np.float32
    assert trace.density.flags.c_contiguous
    assert trace.color_density.flags.c_contiguous
    assert trace.source_size == (5, 3)
    assert trace.signal_standard == DEFAULT_WAVEFORM_SIGNAL_STANDARD
    assert np.allclose(trace.y_prime_coefficients, (0.2126, 0.7152, 0.0722))
    assert trace.plot_scale == pytest.approx(
        vectorscope_chroma_scale(trace.y_prime_coefficients)
    )


def test_build_vectorscope_trace_decimates_samples_deterministically() -> None:
    buffer_rgb = np.random.default_rng(17).random((20, 20, 3), dtype=np.float32)

    first = build_vectorscope_trace(buffer_rgb, density_size=32, max_samples=37)
    second = build_vectorscope_trace(buffer_rgb, density_size=32, max_samples=37)

    assert first.x_values.shape == (37,)
    assert np.array_equal(first.x_values, second.x_values)
    assert np.array_equal(first.y_values, second.y_values)
    assert np.array_equal(first.density, second.density)


def test_build_vectorscope_trace_maps_neutral_values_to_center() -> None:
    buffer_rgb = np.asarray(
        [
            [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
            [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0)],
        ],
        dtype=np.float32,
    )

    trace = build_vectorscope_trace(buffer_rgb, density_size=33)

    assert np.allclose(trace.x_values, 0.0, atol=1e-6)
    assert np.allclose(trace.y_values, 0.0, atol=1e-6)
    assert trace.density[16, 16] == pytest.approx(1.0)
    assert np.count_nonzero(trace.density) == 1


def test_build_vectorscope_trace_bt709_primary_coordinates() -> None:
    buffer_rgb = np.asarray(
        [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]],
        dtype=np.float32,
    )

    trace = build_vectorscope_trace(buffer_rgb, density_size=65)
    scale = trace.plot_scale

    assert trace.x_values == pytest.approx(
        np.asarray([-0.2291442, -0.7708558, 1.0], dtype=np.float32) / scale,
        abs=1e-6,
    )
    assert trace.y_values == pytest.approx(
        np.asarray([1.0, -0.9083058, -0.09169418], dtype=np.float32) / scale,
        abs=1e-6,
    )
    assert np.all(np.hypot(trace.x_values, trace.y_values) > 0.75)
    assert np.all(np.hypot(trace.x_values, trace.y_values) <= 1.0)


def test_build_vectorscope_trace_bt2020_uses_selected_signal_standard() -> None:
    buffer_rgb = np.asarray(
        [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]],
        dtype=np.float32,
    )

    bt709 = build_vectorscope_trace(
        buffer_rgb,
        density_size=65,
        signal_standard="ITU-R BT.709",
    )
    bt2020 = build_vectorscope_trace(
        buffer_rgb,
        density_size=65,
        signal_standard="ITU-R BT.2020",
    )

    assert bt2020.signal_standard == "ITU-R BT.2020"
    assert np.allclose(bt2020.y_prime_coefficients, (0.2627, 0.6780, 0.0593))
    assert not np.array_equal(bt709.x_values, bt2020.x_values)
    assert not np.array_equal(bt709.y_values, bt2020.y_values)


def test_build_vectorscope_trace_density_is_normalized() -> None:
    buffer_rgb = np.asarray(
        [
            [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)],
            [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0)],
            [(0.5, 0.5, 0.5), (0.0, 0.0, 1.0)],
        ],
        dtype=np.float32,
    )

    trace = build_vectorscope_trace(buffer_rgb, density_size=65)

    assert float(np.max(trace.density)) == pytest.approx(1.0)
    assert np.all(trace.density >= 0.0)
    assert np.all(trace.color_density >= 0.0)
    assert np.all(trace.color_density <= 1.0)
    assert np.count_nonzero(trace.density) == 3


def test_build_vectorscope_trace_color_density_preserves_source_color() -> None:
    buffer_rgb = np.asarray(
        [[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]],
        dtype=np.float32,
    )

    trace = build_vectorscope_trace(buffer_rgb, density_size=65)
    occupied = np.argwhere(trace.density > 0.0)

    assert occupied.shape == (1, 2)
    row, col = occupied[0]
    assert trace.color_density[row, col, 0] == pytest.approx(1.0)
    assert trace.color_density[row, col, 1] == pytest.approx(0.0)
    assert trace.color_density[row, col, 2] == pytest.approx(0.0)


def test_build_vectorscope_view_data_mode_a_uses_only_a() -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    b = np.ones((2, 2, 3), dtype=np.float32)
    view = build_vectorscope_view_data("A", a, b, density_size=8)

    assert view.mode == "A"
    assert view.trace_a is not None
    assert view.trace_b is None
    assert view.status == "Vectorscope: A"


def test_build_vectorscope_view_data_mode_b_missing_buffer() -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    view = build_vectorscope_view_data("B", a, None, density_size=8)

    assert view.mode == "B"
    assert view.trace_a is None
    assert view.trace_b is None
    assert view.status == "B image not available"


def test_build_vectorscope_view_data_mode_ab_supports_missing_side() -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    view = build_vectorscope_view_data("A|B", a, None, density_size=8)

    assert view.mode == "A|B"
    assert view.trace_a is not None
    assert view.trace_b is None
    assert view.status == "Vectorscope: A (B missing)"


@pytest.mark.parametrize("mode", ["A", "B", "A|B"])
def test_build_vectorscope_view_data_threads_signal_standard(mode: str) -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    b = np.ones((2, 2, 3), dtype=np.float32)

    view = build_vectorscope_view_data(
        cast(VectorscopeMode, mode),
        a,
        b,
        density_size=8,
        signal_standard="ITU-R BT.2020",
    )

    traces = [trace for trace in (view.trace_a, view.trace_b) if trace is not None]
    assert traces
    assert all(trace.signal_standard == "ITU-R BT.2020" for trace in traces)
