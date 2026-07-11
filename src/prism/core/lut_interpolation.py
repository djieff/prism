"""Interpolation helpers for LUT inspection and analysis."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def evaluate_piecewise_linear(points: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Evaluate ordered `(x, y)` points at `x_values` with endpoint clamping."""
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("Piecewise-linear points must be shaped (n, 2)")
    if point_array.shape[0] == 0:
        return np.asarray(x_values, dtype=np.float32).copy()

    x = point_array[:, 0]
    y = point_array[:, 1]
    if np.any(np.diff(x) < 0.0):
        raise ValueError("Piecewise-linear point x values must be sorted")

    return np.interp(
        np.asarray(x_values, dtype=np.float32),
        x,
        y,
        left=float(y[0]),
        right=float(y[-1]),
    ).astype(np.float32)


def normalize_values_to_unit_from_points(points: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Normalize values to `[0, 1]` using the first/last point output range."""
    point_array = np.asarray(points, dtype=np.float32)
    value_array = np.asarray(values, dtype=np.float32)
    if point_array.size == 0:
        clipped_empty = np.asarray(np.clip(value_array, 0.0, 1.0), dtype=np.float32)
        return clipped_empty
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("Normalization points must be shaped (n, 2)")

    out_min = float(point_array[0, 1])
    out_max = float(point_array[-1, 1])
    span = out_max - out_min
    if abs(span) < 1e-12:
        return np.zeros_like(value_array, dtype=np.float32)

    normalized = (value_array - out_min) / span
    clipped = np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)
    return clipped


def sample_lut3d_trilinear(cube: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Sample a `(z, y, x, c)` LUT cube at normalized `(x, y, z)` coordinates."""
    cube_array = np.asarray(cube, dtype=np.float32)
    if cube_array.ndim != 4:
        raise ValueError("3D LUT cube must be shaped (z, y, x, channels)")
    if min(cube_array.shape[:3]) <= 1:
        raise ValueError("3D LUT cube dimensions must all be greater than one")
    if cube_array.shape[3] <= 0:
        raise ValueError("3D LUT cube must contain at least one channel")

    coordinate_array = np.asarray(coordinates, dtype=np.float32)
    if coordinate_array.ndim < 1 or coordinate_array.shape[-1] != 3:
        raise ValueError("3D LUT coordinates must end with three normalized values")

    original_shape = coordinate_array.shape[:-1]
    flat_xyz = coordinate_array.reshape((-1, 3))
    clamped_xyz = np.clip(flat_xyz, 0.0, 1.0)

    z_size, y_size, x_size, _channels = cube_array.shape
    grid = (
        np.linspace(0.0, 1.0, z_size, dtype=np.float32),
        np.linspace(0.0, 1.0, y_size, dtype=np.float32),
        np.linspace(0.0, 1.0, x_size, dtype=np.float32),
    )
    # SciPy's grid order follows the stored cube axis order `(z, y, x)`.
    flat_zyx = clamped_xyz[:, [2, 1, 0]]
    interpolator = RegularGridInterpolator(
        grid,
        cube_array,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    sampled = np.asarray(interpolator(flat_zyx), dtype=np.float32)
    reshaped = np.asarray(sampled.reshape((*original_shape, cube_array.shape[3])), dtype=np.float32)
    return reshaped
