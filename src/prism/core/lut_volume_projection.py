"""Projection helpers for 3D LUT volume inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

VolumeProjectionMode = Literal["RGB isometric", "RG plane", "RB plane", "GB plane"]

DEFAULT_VOLUME_SAMPLE_LIMIT = 50_000


@dataclass(frozen=True)
class LutVolumePointCloud:
    """Flattened and optionally decimated LUT volume samples."""

    input_rgb: np.ndarray
    output_rgb: np.ndarray
    colors_rgb: np.ndarray
    sample_indices: np.ndarray
    total_point_count: int


@dataclass(frozen=True)
class LutVolumeProjection:
    """2D projection payload for a LUT volume renderer."""

    xy: np.ndarray
    colors_rgb: np.ndarray
    sample_indices: np.ndarray
    total_point_count: int
    projected_point_count: int
    mode: VolumeProjectionMode


def build_lut_volume_point_cloud(
    values: np.ndarray,
    *,
    sample_limit: int = DEFAULT_VOLUME_SAMPLE_LIMIT,
) -> LutVolumePointCloud:
    """Return deterministic input/output RGB point samples from a 3D LUT volume."""
    volume = _validated_volume_values(values)
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")

    size_z, size_y, size_x, _channels = volume.shape
    input_rgb = _input_lattice_coordinates(size_x, size_y, size_z)
    output_rgb = volume.reshape((-1, 3))
    total_count = int(output_rgb.shape[0])
    sample_indices = _deterministic_sample_indices(total_count, sample_limit)

    sampled_output = output_rgb[sample_indices]
    return LutVolumePointCloud(
        input_rgb=input_rgb[sample_indices],
        output_rgb=sampled_output,
        colors_rgb=np.asarray(np.clip(sampled_output, 0.0, 1.0), dtype=np.float32),
        sample_indices=sample_indices,
        total_point_count=total_count,
    )


def project_lut_volume_point_cloud(
    point_cloud: LutVolumePointCloud,
    *,
    mode: VolumeProjectionMode = "RGB isometric",
    use_output_positions: bool = True,
) -> LutVolumeProjection:
    """Project a LUT volume point cloud into normalized 2D coordinates."""
    positions = point_cloud.output_rgb if use_output_positions else point_cloud.input_rgb
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("point cloud positions must be shaped (n, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("point cloud positions must contain only finite values")

    clipped_positions = np.asarray(np.clip(positions, 0.0, 1.0), dtype=np.float32)
    if mode == "RGB isometric":
        xy = _project_isometric(clipped_positions)
    elif mode == "RG plane":
        xy = clipped_positions[:, [0, 1]]
    elif mode == "RB plane":
        xy = clipped_positions[:, [0, 2]]
    elif mode == "GB plane":
        xy = clipped_positions[:, [1, 2]]
    else:
        raise ValueError(f"Unsupported LUT volume projection mode: {mode!r}")

    return LutVolumeProjection(
        xy=np.asarray(np.clip(xy, 0.0, 1.0), dtype=np.float32),
        colors_rgb=point_cloud.colors_rgb,
        sample_indices=point_cloud.sample_indices,
        total_point_count=point_cloud.total_point_count,
        projected_point_count=int(xy.shape[0]),
        mode=mode,
    )


def project_lut_volume(
    values: np.ndarray,
    *,
    mode: VolumeProjectionMode = "RGB isometric",
    sample_limit: int = DEFAULT_VOLUME_SAMPLE_LIMIT,
    use_output_positions: bool = True,
) -> LutVolumeProjection:
    """Build and project a LUT volume point cloud in one call."""
    point_cloud = build_lut_volume_point_cloud(values, sample_limit=sample_limit)
    return project_lut_volume_point_cloud(
        point_cloud,
        mode=mode,
        use_output_positions=use_output_positions,
    )


def _validated_volume_values(values: np.ndarray) -> np.ndarray:
    volume = np.asarray(values, dtype=np.float32)
    if volume.ndim != 4 or volume.shape[3] != 3:
        raise ValueError("LUT volume values must be shaped (z, y, x, 3)")
    if min(volume.shape[:3]) <= 1:
        raise ValueError("LUT volume dimensions must all be greater than one")
    if not np.all(np.isfinite(volume)):
        raise ValueError("LUT volume values must contain only finite values")
    return volume


def _input_lattice_coordinates(size_x: int, size_y: int, size_z: int) -> np.ndarray:
    x_values = np.linspace(0.0, 1.0, size_x, dtype=np.float32)
    y_values = np.linspace(0.0, 1.0, size_y, dtype=np.float32)
    z_values = np.linspace(0.0, 1.0, size_z, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z_values, y_values, x_values, indexing="ij")
    return np.stack((xx, yy, zz), axis=3).reshape((-1, 3))


def _deterministic_sample_indices(total_count: int, sample_limit: int) -> np.ndarray:
    if total_count <= sample_limit:
        return np.arange(total_count, dtype=np.int64)
    stride = int(np.ceil(total_count / float(sample_limit)))
    return np.arange(0, total_count, stride, dtype=np.int64)[:sample_limit]


def _project_isometric(positions: np.ndarray) -> np.ndarray:
    centered = positions - np.float32(0.5)
    basis_x = np.asarray([1.0, -1.0, 0.0], dtype=np.float32) / np.float32(np.sqrt(2.0))
    basis_y = np.asarray([-1.0, -1.0, 2.0], dtype=np.float32) / np.float32(np.sqrt(6.0))
    raw = np.column_stack((centered @ basis_x, centered @ basis_y))

    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    centered_corners = corners - np.float32(0.5)
    projected_corners = np.column_stack(
        (centered_corners @ basis_x, centered_corners @ basis_y)
    )
    min_xy = np.min(projected_corners, axis=0)
    max_xy = np.max(projected_corners, axis=0)
    span = np.maximum(max_xy - min_xy, np.float32(1e-8))
    return np.asarray((raw - min_xy) / span, dtype=np.float32)
