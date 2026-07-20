"""Tests for LUT volume camera math."""

from __future__ import annotations

import numpy as np
import pytest

from prism.core.lut.volume_camera import (
    MAX_CAMERA_DISTANCE,
    MAX_PITCH_DEGREES,
    MAX_VIEW_SCALE,
    MIN_CAMERA_DISTANCE,
    MIN_PITCH_DEGREES,
    MIN_VIEW_SCALE,
    LutVolumeCamera,
    dolly_lut_volume_camera,
    lut_volume_orthographic_projection_matrix,
    lut_volume_view_matrix,
    lut_volume_view_projection_matrix,
    orbit_lut_volume_camera,
    pan_lut_volume_camera,
    reset_lut_volume_camera,
    zoom_lut_volume_camera,
)


def _project_to_view_xy(camera: LutVolumeCamera, rgb: tuple[float, float, float]) -> np.ndarray:
    point = np.asarray([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
    return (lut_volume_view_matrix(camera) @ point)[:2]


def test_reset_camera_presets_map_plane_axes_to_expected_screen_axes() -> None:
    center = (0.5, 0.5, 0.5)

    rg = reset_lut_volume_camera("RG plane")
    rg_center = _project_to_view_xy(rg, center)
    assert _project_to_view_xy(rg, (1.0, 0.5, 0.5))[0] > rg_center[0]
    assert _project_to_view_xy(rg, (0.5, 1.0, 0.5))[1] > rg_center[1]

    rb = reset_lut_volume_camera("RB plane")
    rb_center = _project_to_view_xy(rb, center)
    assert _project_to_view_xy(rb, (1.0, 0.5, 0.5))[0] > rb_center[0]
    assert _project_to_view_xy(rb, (0.5, 0.5, 1.0))[1] > rb_center[1]

    gb = reset_lut_volume_camera("GB plane")
    gb_center = _project_to_view_xy(gb, center)
    assert _project_to_view_xy(gb, (0.5, 1.0, 0.5))[0] > gb_center[0]
    assert _project_to_view_xy(gb, (0.5, 0.5, 1.0))[1] > gb_center[1]


def test_reset_camera_isometric_returns_finite_matrices() -> None:
    camera = reset_lut_volume_camera("RGB isometric")

    view = lut_volume_view_matrix(camera)
    projection = lut_volume_orthographic_projection_matrix(camera, aspect_ratio=16.0 / 9.0)
    view_projection = lut_volume_view_projection_matrix(camera, aspect_ratio=1.0)

    assert view.shape == (4, 4)
    assert projection.shape == (4, 4)
    assert view_projection.shape == (4, 4)
    assert view.dtype == np.float32
    assert projection.dtype == np.float32
    assert view_projection.dtype == np.float32
    assert np.all(np.isfinite(view))
    assert np.all(np.isfinite(projection))
    assert np.all(np.isfinite(view_projection))


def test_orbit_camera_normalizes_yaw_and_clamps_pitch() -> None:
    camera = reset_lut_volume_camera("RGB isometric")

    updated = orbit_lut_volume_camera(
        camera,
        delta_yaw_degrees=400.0,
        delta_pitch_degrees=200.0,
    )

    assert -180.0 <= updated.yaw_degrees < 180.0
    assert updated.pitch_degrees == MAX_PITCH_DEGREES

    updated = orbit_lut_volume_camera(
        camera,
        delta_yaw_degrees=-400.0,
        delta_pitch_degrees=-300.0,
    )

    assert -180.0 <= updated.yaw_degrees < 180.0
    assert updated.pitch_degrees == MIN_PITCH_DEGREES


def test_pan_camera_updates_target_without_mutating_original() -> None:
    camera = reset_lut_volume_camera("RG plane")

    updated = pan_lut_volume_camera(camera, delta_x=0.1, delta_y=-0.2)

    assert camera.pan_x == 0.0
    assert camera.pan_y == 0.0
    assert camera.pan_z == 0.0
    assert updated.pan_x == pytest.approx(0.1)
    assert updated.pan_y == pytest.approx(-0.2)
    assert updated.pan_z == pytest.approx(0.0)
    np.testing.assert_allclose(updated.target_rgb, [0.6, 0.3, 0.5])


@pytest.mark.parametrize(
    ("mode", "expected_horizontal", "expected_vertical"),
    [
        ("RG plane", (0.1, 0.0, 0.0), (0.0, 0.2, 0.0)),
        ("RB plane", (0.1, 0.0, 0.0), (0.0, 0.0, 0.2)),
        ("GB plane", (0.0, 0.1, 0.0), (0.0, 0.0, 0.2)),
    ],
)
def test_pan_camera_follows_current_plane_screen_axes(
    mode,
    expected_horizontal: tuple[float, float, float],
    expected_vertical: tuple[float, float, float],
) -> None:
    camera = reset_lut_volume_camera(mode)

    horizontal = pan_lut_volume_camera(camera, delta_x=0.1, delta_y=0.0)
    vertical = pan_lut_volume_camera(camera, delta_x=0.0, delta_y=0.2)

    np.testing.assert_allclose(
        [horizontal.pan_x, horizontal.pan_y, horizontal.pan_z],
        expected_horizontal,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [vertical.pan_x, vertical.pan_y, vertical.pan_z],
        expected_vertical,
        atol=1e-6,
    )


def test_zoom_and_dolly_clamp_ranges() -> None:
    camera = reset_lut_volume_camera("RGB isometric")

    zoomed_in = zoom_lut_volume_camera(camera, scale_factor=0.001)
    zoomed_out = zoom_lut_volume_camera(camera, scale_factor=1000.0)
    near = dolly_lut_volume_camera(camera, distance_factor=0.001)
    far = dolly_lut_volume_camera(camera, distance_factor=1000.0)

    assert zoomed_in.view_scale == MIN_VIEW_SCALE
    assert zoomed_out.view_scale == MAX_VIEW_SCALE
    assert near.distance == MIN_CAMERA_DISTANCE
    assert far.distance == MAX_CAMERA_DISTANCE


def test_camera_rejects_invalid_zoom_dolly_and_aspect_values() -> None:
    camera = reset_lut_volume_camera("RGB isometric")

    with pytest.raises(ValueError, match="scale_factor"):
        zoom_lut_volume_camera(camera, scale_factor=0.0)

    with pytest.raises(ValueError, match="distance_factor"):
        dolly_lut_volume_camera(camera, distance_factor=0.0)

    with pytest.raises(ValueError, match="aspect_ratio"):
        lut_volume_orthographic_projection_matrix(camera, aspect_ratio=0.0)


def test_reset_camera_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="preset"):
        reset_lut_volume_camera("bad mode")  # type: ignore[arg-type]
