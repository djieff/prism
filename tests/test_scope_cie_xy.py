"""Tests for CIE xy scope core helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from prism.core.scopes.cie_xy import (
    CIE_XY_X_RANGE,
    CIE_XY_Y_RANGE,
    CieXyMode,
    available_cie_gamut_overlays,
    available_cie_rgb_space_names,
    available_cie_whitepoint_names,
    build_cie_xy_trace,
    build_cie_xy_view_data,
    get_cie_rgb_space,
    get_cie_whitepoint,
    spectral_locus_xy,
    unique_cie_gamut_overlay_names,
)


def test_build_cie_xy_trace_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected image buffer shape"):
        build_cie_xy_trace(np.zeros((4, 4), dtype=np.float32))


def test_build_cie_xy_trace_rejects_non_finite_values() -> None:
    buffer_rgb = np.zeros((2, 2, 3), dtype=np.float32)
    buffer_rgb[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_cie_xy_trace(buffer_rgb)


def test_build_cie_xy_trace_rejects_unknown_rgb_space() -> None:
    buffer_rgb = np.ones((1, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported CIE RGB space"):
        build_cie_xy_trace(buffer_rgb, rgb_space_name="Not A Space")


def test_build_cie_xy_trace_shape_and_source_size() -> None:
    buffer_rgb = np.ones((3, 5, 3), dtype=np.float32)
    trace = build_cie_xy_trace(buffer_rgb, density_size=16)

    assert trace.x_values.shape == (15,)
    assert trace.y_values.shape == (15,)
    assert trace.density.shape == (16, 16)
    assert trace.color_density.shape == (16, 16, 3)
    assert trace.density.dtype == np.float32
    assert trace.color_density.dtype == np.float32
    assert trace.density.flags.c_contiguous
    assert trace.color_density.flags.c_contiguous
    assert trace.source_size == (5, 3)
    assert trace.rgb_space_name == "sRGB"


def test_build_cie_xy_trace_decimates_samples_deterministically() -> None:
    buffer_rgb = np.random.default_rng(17).random((20, 20, 3), dtype=np.float32)

    first = build_cie_xy_trace(buffer_rgb, density_size=32, max_samples=37)
    second = build_cie_xy_trace(buffer_rgb, density_size=32, max_samples=37)

    assert first.x_values.shape == (37,)
    assert np.array_equal(first.x_values, second.x_values)
    assert np.array_equal(first.y_values, second.y_values)
    assert np.array_equal(first.density, second.density)


def test_build_cie_xy_trace_maps_neutral_values_to_whitepoint() -> None:
    buffer_rgb = np.asarray(
        [
            [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
            [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0)],
        ],
        dtype=np.float32,
    )

    trace = build_cie_xy_trace(buffer_rgb, density_size=33, rgb_space_name="sRGB")

    assert trace.x_values.shape == (3,)
    assert trace.y_values.shape == (3,)
    assert np.allclose(trace.x_values, 0.3127, atol=5e-4)
    assert np.allclose(trace.y_values, 0.3290, atol=5e-4)
    assert float(np.max(trace.density)) == pytest.approx(1.0)
    assert np.count_nonzero(trace.density) == 1


def test_build_cie_xy_trace_srgb_primaries_land_on_srgb_primary_xy() -> None:
    buffer_rgb = np.asarray(
        [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]],
        dtype=np.float32,
    )

    trace = build_cie_xy_trace(buffer_rgb, density_size=65, rgb_space_name="sRGB")

    assert trace.x_values == pytest.approx(np.asarray([0.64, 0.30, 0.15]), abs=5e-4)
    assert trace.y_values == pytest.approx(np.asarray([0.33, 0.60, 0.06]), abs=5e-4)
    assert np.count_nonzero(trace.density) == 3


def test_build_cie_xy_trace_uses_selected_rgb_space() -> None:
    buffer_rgb = np.asarray([[(1.0, 0.0, 0.0)]], dtype=np.float32)

    srgb = build_cie_xy_trace(buffer_rgb, density_size=65, rgb_space_name="sRGB")
    rec2020 = build_cie_xy_trace(buffer_rgb, density_size=65, rgb_space_name="ITU-R BT.2020")

    assert srgb.rgb_space_name == "sRGB"
    assert rec2020.rgb_space_name == "ITU-R BT.2020"
    assert srgb.x_values[0] == pytest.approx(0.64, abs=5e-4)
    assert rec2020.x_values[0] == pytest.approx(0.708, abs=5e-4)
    assert not np.array_equal(srgb.x_values, rec2020.x_values)


def test_build_cie_xy_trace_clamps_out_of_range_rgb() -> None:
    buffer_rgb = np.asarray([[(-1.0, 2.0, 0.0), (0.0, 1.0, 0.0)]], dtype=np.float32)

    trace = build_cie_xy_trace(buffer_rgb, density_size=65, rgb_space_name="sRGB")

    assert trace.x_values == pytest.approx(np.asarray([0.30, 0.30]), abs=5e-4)
    assert trace.y_values == pytest.approx(np.asarray([0.60, 0.60]), abs=5e-4)
    assert np.count_nonzero(trace.density) == 1


def test_build_cie_xy_trace_ignores_zero_black_chromaticity() -> None:
    buffer_rgb = np.zeros((2, 2, 3), dtype=np.float32)

    trace = build_cie_xy_trace(buffer_rgb, density_size=16)

    assert trace.x_values.shape == (0,)
    assert trace.y_values.shape == (0,)
    assert np.count_nonzero(trace.density) == 0
    assert np.count_nonzero(trace.color_density) == 0


def test_get_cie_rgb_space_returns_overlay_contract_values() -> None:
    p3 = get_cie_rgb_space("Display P3")
    rec2020 = get_cie_rgb_space("ITU-R BT.2020")

    assert np.asarray(p3.primaries_xy) == pytest.approx(
        np.asarray(((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)))
    )
    assert p3.whitepoint_xy == pytest.approx((0.3127, 0.3290))
    assert p3.rgb_to_xyz.shape == (3, 3)
    assert np.asarray(rec2020.primaries_xy) == pytest.approx(
        np.asarray(((0.708, 0.292), (0.17, 0.797), (0.131, 0.046)))
    )


def test_available_cie_rgb_spaces_and_overlays_are_alphabetical() -> None:
    names = available_cie_rgb_space_names()
    overlays = available_cie_gamut_overlays()

    assert names == tuple(sorted(names, key=str.casefold))
    assert {
        "ARRI Wide Gamut 3",
        "ARRI Wide Gamut 4",
        "ACEScg",
        "ACES2065-1",
        "DRAGONcolor",
        "DRAGONcolor2",
        "REDWideGamutRGB",
        "S-Gamut",
        "S-Gamut3",
        "S-Gamut3.Cine",
        "Venice S-Gamut3",
        "Venice S-Gamut3.Cine",
    }.issubset(names)
    assert tuple(overlay.name for overlay in overlays) == tuple(
        get_cie_rgb_space(name).name for name in names
    )


def test_available_cie_whitepoints_and_lookup_use_colour_science_values() -> None:
    assert available_cie_whitepoint_names() == ("D50", "D55", "D60", "D65", "D75", "DCI-P3")

    d60 = get_cie_whitepoint("D60")
    d65 = get_cie_whitepoint("D65")

    assert d60.name == "D60"
    assert d60.xy == pytest.approx((0.32161671, 0.33761992))
    assert d65.xy == pytest.approx((0.3127, 0.3290))


def test_get_cie_whitepoint_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported CIE white point"):
        get_cie_whitepoint("Not A White Point")


def test_unique_cie_gamut_overlay_names_de_duplicates_and_skips_empty_values() -> None:
    assert unique_cie_gamut_overlay_names(
        ("sRGB", None, "Display P3", "sRGB", "", "ITU-R BT.2020")
    ) == ("sRGB", "Display P3", "ITU-R BT.2020")


def test_unique_cie_gamut_overlay_names_rejects_unknown_overlay() -> None:
    with pytest.raises(ValueError, match="Unsupported CIE gamut overlay"):
        unique_cie_gamut_overlay_names(("sRGB", "Not A Space"))


def test_spectral_locus_xy_uses_cie_1931_data_inside_plot_bounds() -> None:
    locus = spectral_locus_xy()

    assert locus.shape == (471, 2)
    assert locus.dtype == np.float32
    assert locus.flags.c_contiguous
    assert float(np.min(locus[:, 0])) >= CIE_XY_X_RANGE[0]
    assert float(np.max(locus[:, 0])) <= CIE_XY_X_RANGE[1]
    assert float(np.min(locus[:, 1])) >= CIE_XY_Y_RANGE[0]
    assert float(np.max(locus[:, 1])) <= CIE_XY_Y_RANGE[1]


def test_build_cie_xy_view_data_mode_a_uses_only_a() -> None:
    a = np.ones((2, 2, 3), dtype=np.float32)
    b = np.zeros((2, 2, 3), dtype=np.float32)
    view = build_cie_xy_view_data("A", a, b, density_size=8, rgb_space_name="sRGB")

    assert view.mode == "A"
    assert view.trace_a is not None
    assert view.trace_b is None
    assert view.status == "CIE xy: A"


def test_build_cie_xy_view_data_mode_b_missing_buffer() -> None:
    a = np.ones((2, 2, 3), dtype=np.float32)
    view = build_cie_xy_view_data("B", a, None, density_size=8)

    assert view.mode == "B"
    assert view.trace_a is None
    assert view.trace_b is None
    assert view.status == "B image not available"


def test_build_cie_xy_view_data_mode_ab_supports_missing_side() -> None:
    a = np.ones((2, 2, 3), dtype=np.float32)
    view = build_cie_xy_view_data("A|B", a, None, density_size=8)

    assert view.mode == "A|B"
    assert view.trace_a is not None
    assert view.trace_b is None
    assert view.status == "CIE xy: A (B missing)"


@pytest.mark.parametrize("mode", ["A", "B", "A|B"])
def test_build_cie_xy_view_data_threads_rgb_space(mode: str) -> None:
    a = np.ones((2, 2, 3), dtype=np.float32)
    b = np.ones((2, 2, 3), dtype=np.float32)

    view = build_cie_xy_view_data(
        cast(CieXyMode, mode),
        a,
        b,
        density_size=8,
        rgb_space_name="Display P3",
    )

    traces = [trace for trace in (view.trace_a, view.trace_b) if trace is not None]
    assert traces
    assert all(trace.rgb_space_name == "Display P3" for trace in traces)
