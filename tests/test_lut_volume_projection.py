"""Tests for LUT volume projection helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prism.core.lut_volume_projection import (
    DEFAULT_VOLUME_SAMPLE_LIMIT,
    build_lut_volume_point_cloud,
    project_lut_volume,
    project_lut_volume_point_cloud,
)
from prism.io.lut_loader import load_lut_volume_data


def _identity_cube(size: int) -> np.ndarray:
    values = np.zeros((size, size, size, 3), dtype=np.float32)
    for z in range(size):
        for y in range(size):
            for x in range(size):
                values[z, y, x] = [
                    x / float(size - 1),
                    y / float(size - 1),
                    z / float(size - 1),
                ]
    return values


def test_build_lut_volume_point_cloud_preserves_zyx_axis_order() -> None:
    cube = _identity_cube(2)

    point_cloud = build_lut_volume_point_cloud(cube)

    assert point_cloud.total_point_count == 8
    assert point_cloud.input_rgb.shape == (8, 3)
    assert point_cloud.output_rgb.shape == (8, 3)
    np.testing.assert_allclose(point_cloud.input_rgb, point_cloud.output_rgb)
    np.testing.assert_allclose(point_cloud.output_rgb[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(point_cloud.output_rgb[2], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(point_cloud.output_rgb[4], [0.0, 0.0, 1.0])


def test_build_lut_volume_point_cloud_decimates_deterministically() -> None:
    cube = _identity_cube(5)

    point_cloud = build_lut_volume_point_cloud(cube, sample_limit=20)

    assert point_cloud.total_point_count == 125
    assert point_cloud.output_rgb.shape[0] == 18
    np.testing.assert_array_equal(point_cloud.sample_indices, np.arange(0, 125, 7))


def test_build_lut_volume_point_cloud_keeps_33_cube_under_default_limit() -> None:
    cube = _identity_cube(33)

    point_cloud = build_lut_volume_point_cloud(cube)

    assert point_cloud.total_point_count == 33**3
    assert point_cloud.output_rgb.shape[0] == 33**3
    assert point_cloud.output_rgb.shape[0] < DEFAULT_VOLUME_SAMPLE_LIMIT


def test_project_lut_volume_point_cloud_supports_plane_modes() -> None:
    point_cloud = build_lut_volume_point_cloud(_identity_cube(2))

    rg = project_lut_volume_point_cloud(point_cloud, mode="RG plane")
    rb = project_lut_volume_point_cloud(point_cloud, mode="RB plane")
    gb = project_lut_volume_point_cloud(point_cloud, mode="GB plane")

    np.testing.assert_allclose(rg.xy, point_cloud.output_rgb[:, [0, 1]])
    np.testing.assert_allclose(rb.xy, point_cloud.output_rgb[:, [0, 2]])
    np.testing.assert_allclose(gb.xy, point_cloud.output_rgb[:, [1, 2]])


def test_project_lut_volume_isometric_returns_finite_normalized_coordinates() -> None:
    projection = project_lut_volume(_identity_cube(3), mode="RGB isometric")

    assert projection.mode == "RGB isometric"
    assert projection.xy.shape == (27, 2)
    assert projection.colors_rgb.shape == (27, 3)
    assert np.all(np.isfinite(projection.xy))
    assert np.min(projection.xy) >= 0.0
    assert np.max(projection.xy) <= 1.0


def test_project_lut_volume_clips_output_positions_and_display_colors() -> None:
    cube = _identity_cube(2)
    cube[0, 0, 0] = [-1.0, 0.5, 2.0]

    projection = project_lut_volume(cube, mode="RG plane")

    np.testing.assert_allclose(projection.xy[0], [0.0, 0.5])
    np.testing.assert_allclose(projection.colors_rgb[0], [0.0, 0.5, 1.0])


def test_project_lut_volume_can_project_input_lattice_positions() -> None:
    cube = _identity_cube(2)
    cube[:] = 0.25

    projection = project_lut_volume(cube, mode="RG plane", use_output_positions=False)

    np.testing.assert_allclose(projection.xy[1], [1.0, 0.0])
    np.testing.assert_allclose(projection.colors_rgb[1], [0.25, 0.25, 0.25])


def test_project_lut_volume_rejects_invalid_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="shaped"):
        project_lut_volume(np.zeros((2, 2, 3), dtype=np.float32))

    bad = _identity_cube(2)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        project_lut_volume(bad)

    with pytest.raises(ValueError, match="sample_limit"):
        build_lut_volume_point_cloud(_identity_cube(2), sample_limit=0)


def test_project_real_sample_65_cube_is_decimated() -> None:
    volume = load_lut_volume_data(
        Path("samples/LUTs/3D/colorspace/aces2065_to_acescg_65.cube")
    )
    assert volume is not None

    projection = project_lut_volume(volume.values)

    assert projection.total_point_count == 65**3
    assert projection.projected_point_count <= DEFAULT_VOLUME_SAMPLE_LIMIT
    assert np.all(np.isfinite(projection.xy))
