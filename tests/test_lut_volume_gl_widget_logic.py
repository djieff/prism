"""Logic tests for LUT OpenGL volume widget state."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QVector4D
from PySide6.QtWidgets import QApplication

from prism.core.lut_volume_camera import lut_volume_view_projection_matrix
from prism.io.lut_loader import LutVolumeData
from prism.ui import lut_volume_gl_widget as gl_widget_module
from prism.ui.lut_volume_gl_widget import LutVolumeGlWidget


@pytest.fixture(scope="module", autouse=True)
def _qapplication() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _volume_data() -> LutVolumeData:
    values = np.zeros((2, 2, 2, 3), dtype=np.float32)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                values[z, y, x] = [x, y, z]
    return LutVolumeData(
        path=Path("identity.cube"),
        format="cube",
        size_x=2,
        size_y=2,
        size_z=2,
        values=values,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def _compressed_volume_data() -> LutVolumeData:
    values = np.zeros((2, 2, 2, 3), dtype=np.float32)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                values[z, y, x] = [
                    0.45 + (x * 0.1),
                    0.45 + (y * 0.1),
                    0.45 + (z * 0.1),
                ]
    return LutVolumeData(
        path=Path("compressed.cube"),
        format="cube",
        size_x=2,
        size_y=2,
        size_z=2,
        values=values,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def _payload_ndc_xy(widget: LutVolumeGlWidget) -> np.ndarray:
    assert widget._payload is not None
    widget._apply_pending_camera_fit()
    aspect = widget.width() / max(float(widget.height()), 1.0)
    matrix = lut_volume_view_projection_matrix(widget._camera, aspect_ratio=aspect)
    positions = widget._payload.positions_rgb
    homogeneous_positions = np.concatenate(
        (
            positions,
            np.ones((positions.shape[0], 1), dtype=np.float32),
        ),
        axis=1,
    )
    projected = homogeneous_positions @ matrix.T
    return projected[:, :2] / projected[:, 3:4]


def test_gl_widget_reports_empty_and_loaded_status() -> None:
    widget = LutVolumeGlWidget()

    assert widget.status_text() == "OpenGL volume view requires a 3D LUT."

    widget.set_volume_data(_volume_data())

    assert (
        widget.status_text()
        == "OpenGL volume: 2x2x2 | shown: 8/8 | density: Medium | position: Output cloud"
    )


def test_gl_widget_rebuilds_payload_for_density_and_position() -> None:
    widget = LutVolumeGlWidget()
    widget.set_volume_data(_volume_data())

    assert widget._payload is not None
    assert widget._payload.density == "Medium"
    assert widget._payload.use_output_positions is True

    widget.set_density("Low")
    assert widget._payload is not None
    assert widget._payload.density == "Low"

    widget.set_use_output_positions(False)
    assert widget._payload is not None
    assert widget._payload.use_output_positions is False
    assert "position: Source RGB lattice" in widget.status_text()


def test_gl_widget_projection_mode_resets_camera() -> None:
    widget = LutVolumeGlWidget()
    original_camera = widget._camera

    widget.set_projection_mode("GB plane")
    widget._apply_pending_camera_fit()

    assert widget._projection_mode == "GB plane"
    assert widget._camera != original_camera
    assert widget._camera.up_rgb == (0.0, 0.0, 1.0)


def test_gl_widget_auto_fits_loaded_volume_to_view(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 640)
    monkeypatch.setattr(widget, "height", lambda: 360)

    widget.set_volume_data(_volume_data())

    ndc_xy = _payload_ndc_xy(widget)
    assert float(np.max(np.abs(ndc_xy))) <= 1.0


def test_gl_widget_auto_fit_does_not_over_zoom_compressed_output(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 640)
    monkeypatch.setattr(widget, "height", lambda: 360)

    widget.set_volume_data(_compressed_volume_data())
    widget._apply_pending_camera_fit()

    assert widget._camera.view_scale >= gl_widget_module.DEFAULT_VIEW_SCALE


def test_gl_widget_auto_fit_includes_visible_rgb_axes(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 640)
    monkeypatch.setattr(widget, "height", lambda: 360)

    widget.set_volume_data(_compressed_volume_data())
    widget._apply_pending_camera_fit()

    fit_positions = widget._fit_positions_rgb()
    axis_positions = np.asarray(
        gl_widget_module.AXIS_POSITIONS_RGB,
        dtype=np.float32,
    ).reshape((-1, 3))

    assert fit_positions.shape[0] == widget._payload.positions_rgb.shape[0] + axis_positions.shape[0]
    for axis_position in axis_positions:
        assert np.any(np.all(np.isclose(fit_positions, axis_position), axis=1))


def test_gl_widget_isometric_fit_keeps_visible_axes_below_top_target(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 1280)
    monkeypatch.setattr(widget, "height", lambda: 720)

    widget.set_volume_data(_volume_data())
    widget._apply_pending_camera_fit()

    fit_positions = widget._fit_positions_rgb()
    aspect = widget.width() / max(float(widget.height()), 1.0)
    matrix = lut_volume_view_projection_matrix(widget._camera, aspect_ratio=aspect)
    homogeneous_positions = np.concatenate(
        (
            fit_positions,
            np.ones((fit_positions.shape[0], 1), dtype=np.float32),
        ),
        axis=1,
    )
    projected = homogeneous_positions @ matrix.T
    ndc_y = projected[:, 1] / projected[:, 3]

    assert float(np.max(ndc_y)) <= gl_widget_module.FIT_TARGET_TOP_NDC


def test_gl_widget_pending_fit_uses_latest_viewport_size(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 100)
    monkeypatch.setattr(widget, "height", lambda: 100)

    widget.set_volume_data(_volume_data())

    monkeypatch.setattr(widget, "width", lambda: 1280)
    monkeypatch.setattr(widget, "height", lambda: 360)
    ndc_xy = _payload_ndc_xy(widget)

    assert float(np.max(np.abs(ndc_xy))) <= 1.0


def test_gl_widget_reset_view_refits_current_volume_after_orbit(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 640)
    monkeypatch.setattr(widget, "height", lambda: 360)
    widget.set_volume_data(_volume_data())
    widget._orbit_by_pixels(120.0, 60.0)

    widget.reset_view()

    ndc_xy = _payload_ndc_xy(widget)
    assert float(np.max(np.abs(ndc_xy))) <= 1.0


def test_gl_widget_rgb_axes_visibility_updates_without_rebuilding_payload(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    widget.set_volume_data(_volume_data())
    widget._apply_pending_camera_fit()
    payload = widget._payload
    camera = widget._camera
    update_calls: list[None] = []
    monkeypatch.setattr(widget, "update", lambda: update_calls.append(None))

    widget.set_show_rgb_axes(False)

    assert widget._show_rgb_axes is False
    assert widget._payload is payload
    assert widget._camera == camera
    assert widget._needs_camera_fit is False
    assert update_calls == [None]


def test_gl_widget_interaction_helpers_update_camera(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 200)
    monkeypatch.setattr(widget, "height", lambda: 100)
    update_calls: list[None] = []
    monkeypatch.setattr(widget, "update", lambda: update_calls.append(None))
    original_camera = widget._camera

    widget._orbit_by_pixels(10.0, -5.0)

    assert widget._camera.yaw_degrees < original_camera.yaw_degrees
    assert widget._camera.pitch_degrees != original_camera.pitch_degrees

    orbit_camera = widget._camera
    widget._pan_by_pixels(20.0, -10.0)

    assert widget._camera.pan_x != orbit_camera.pan_x
    assert widget._camera.pan_y != orbit_camera.pan_y

    pan_camera = widget._camera
    widget._zoom_by_steps(1.0)

    assert widget._camera.view_scale < pan_camera.view_scale
    assert update_calls == [None, None, None]


def test_gl_widget_left_drag_orbits_only_in_isometric_mode() -> None:
    widget = LutVolumeGlWidget()
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10.0, 20.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    widget.mousePressEvent(event)

    assert widget._active_mouse_action == "orbit"

    widget.set_projection_mode("GB plane")
    widget.mousePressEvent(event)

    assert widget._active_mouse_action == "pan"


def test_gl_widget_mvp_matrix_matches_core_matrix(monkeypatch) -> None:
    widget = LutVolumeGlWidget()
    monkeypatch.setattr(widget, "width", lambda: 100)
    monkeypatch.setattr(widget, "height", lambda: 100)
    expected = lut_volume_view_projection_matrix(widget._camera, aspect_ratio=1.0)
    point = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    mapped = widget._mvp_matrix().map(QVector4D(float(point[0]), float(point[1]), float(point[2]), float(point[3])))

    np.testing.assert_allclose(
        [mapped.x(), mapped.y(), mapped.z(), mapped.w()],
        expected @ point,
        rtol=1e-6,
        atol=1e-6,
    )


def test_gl_widget_reset_view_restores_current_projection_camera() -> None:
    widget = LutVolumeGlWidget()
    widget.set_projection_mode("GB plane")
    widget._orbit_by_pixels(30.0, 10.0)

    widget.reset_view()

    assert widget._projection_mode == "GB plane"
    assert widget._camera.yaw_degrees == 90.0
    assert widget._camera.pitch_degrees == 0.0
    assert widget._camera.up_rgb == (0.0, 0.0, 1.0)


def test_gl_widget_style_setters_validate_ranges() -> None:
    widget = LutVolumeGlWidget()

    widget.set_point_size(3.0)
    widget.set_opacity(0.5)

    assert widget._point_size == 3.0
    assert widget._opacity == 0.5

    with pytest.raises(ValueError, match="point_size"):
        widget.set_point_size(0.0)

    with pytest.raises(ValueError, match="opacity"):
        widget.set_opacity(1.5)


def test_gl_widget_records_payload_errors(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise ValueError("bad volume")

    monkeypatch.setattr(gl_widget_module, "build_lut_volume_render_payload", _raise)
    widget = LutVolumeGlWidget()

    widget.set_volume_data(_volume_data())

    assert widget._payload is None
    assert widget.status_text() == "OpenGL volume unavailable: bad volume"
