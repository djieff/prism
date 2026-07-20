"""Tests for LUT interpolation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from prism.core.lut.interpolation import (
    evaluate_piecewise_linear,
    normalize_values_to_unit_from_points,
    sample_lut3d_trilinear,
)


def _axis_probe_cube() -> np.ndarray:
    cube = np.zeros((2, 2, 2, 3), dtype=np.float32)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cube[z, y, x] = [float(x), float(y), float(z)]
    return cube


def test_evaluate_piecewise_linear_clamps_to_endpoints() -> None:
    points = np.asarray([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]], dtype=np.float32)
    x_values = np.asarray([-0.25, 0.25, 0.75, 1.25], dtype=np.float32)

    result = evaluate_piecewise_linear(points, x_values)

    assert result.dtype == np.float32
    assert result == pytest.approx([0.0, 0.125, 0.625, 1.0])


def test_evaluate_piecewise_linear_rejects_unsorted_points() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]], dtype=np.float32)

    with pytest.raises(ValueError, match="sorted"):
        evaluate_piecewise_linear(points, np.asarray([0.5], dtype=np.float32))


def test_normalize_values_to_unit_from_points_clamps_results() -> None:
    points = np.asarray([[0.0, 0.2], [1.0, 0.8]], dtype=np.float32)
    values = np.asarray([0.1, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)

    result = normalize_values_to_unit_from_points(points, values)

    assert result == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_sample_lut3d_trilinear_uses_public_xyz_coordinates() -> None:
    cube = _axis_probe_cube()
    coordinates = np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32)

    result = sample_lut3d_trilinear(cube, coordinates)

    assert result.shape == (1, 3)
    assert result[0] == pytest.approx([0.25, 0.5, 0.75])


def test_sample_lut3d_trilinear_clamps_before_sampling() -> None:
    cube = _axis_probe_cube()
    coordinates = np.asarray([[-0.25, 0.5, 1.25]], dtype=np.float32)

    result = sample_lut3d_trilinear(cube, coordinates)

    assert result[0] == pytest.approx([0.0, 0.5, 1.0])


def test_sample_lut3d_trilinear_preserves_coordinate_leading_shape() -> None:
    cube = _axis_probe_cube()
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    result = sample_lut3d_trilinear(cube, coordinates)

    assert result.shape == (2, 2, 3)
    assert result[0, 0] == pytest.approx([0.0, 0.0, 0.0])
    assert result[0, 1] == pytest.approx([1.0, 0.0, 0.0])
    assert result[1, 0] == pytest.approx([0.0, 1.0, 0.0])
    assert result[1, 1] == pytest.approx([0.0, 0.0, 1.0])


def test_sample_lut3d_trilinear_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="cube"):
        sample_lut3d_trilinear(
            np.zeros((2, 2, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="coordinates"):
        sample_lut3d_trilinear(
            np.zeros((2, 2, 2, 3), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        )
