"""OpenGL volume widget for interactive LUT inspection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMatrix4x4, QMouseEvent, QOpenGLFunctions, QWheelEvent
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from prism.core.lut_volume_camera import (
    DEFAULT_VIEW_SCALE,
    MAX_VIEW_SCALE,
    MIN_VIEW_SCALE,
    LutVolumeCamera,
    lut_volume_view_matrix,
    lut_volume_view_projection_matrix,
    orbit_lut_volume_camera,
    pan_lut_volume_camera,
    reset_lut_volume_camera,
    zoom_lut_volume_camera,
)
from prism.core.lut_volume_projection import (
    LutVolumeRenderPayload,
    VolumeDensityPreset,
    VolumeProjectionMode,
    build_lut_volume_render_payload,
)
from prism.io.lut_loader import LutVolumeData

GL_POINTS = 0x0000
GL_LINES = 0x0001
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_FLOAT = 0x1406
GL_PROGRAM_POINT_SIZE = 0x8642
ORBIT_DEGREES_PER_PIXEL = 0.35
PAN_VIEW_SCALE_FACTOR = 1.0
WHEEL_ZOOM_BASE = 0.9
FIT_VIEW_MARGIN = 1.15
FIT_TARGET_TOP_NDC = 0.88
FIT_BOTTOM_LIMIT_NDC = -0.95
AXIS_POSITIONS_RGB = (
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)
AXIS_COLORS_RGB = (
    1.0,
    0.15,
    0.15,
    1.0,
    0.15,
    0.15,
    0.15,
    1.0,
    0.15,
    0.15,
    1.0,
    0.15,
    0.2,
    0.45,
    1.0,
    0.2,
    0.45,
    1.0,
)

VERTEX_SHADER = """
attribute vec3 position;
attribute vec3 color;
uniform mat4 mvp;
uniform float point_size;
varying vec3 v_color;

void main() {
    v_color = color;
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = point_size;
}
"""

FRAGMENT_SHADER = """
varying vec3 v_color;
uniform float opacity;

void main() {
    gl_FragColor = vec4(v_color, opacity);
}
"""


class LutVolumeGlWidget(QOpenGLWidget):
    """Render a 3D LUT point cloud with OpenGL."""

    initialization_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._volume_data: LutVolumeData | None = None
        self._payload: LutVolumeRenderPayload | None = None
        self._projection_mode: VolumeProjectionMode = "RGB isometric"
        self._density: VolumeDensityPreset = "Medium"
        self._use_output_positions = True
        self._show_neutral_axis = True
        self._show_rgb_axes = True
        self._point_size = 2.0
        self._opacity = 0.85
        self._camera = reset_lut_volume_camera(self._projection_mode)
        self._last_mouse_pos: QPointF | None = None
        self._active_mouse_action: str | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._vertex_array = QOpenGLVertexArrayObject(self)
        self._position_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._color_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._axis_position_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._axis_color_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._gl_ready = False
        self._needs_upload = False
        self._needs_camera_fit = False
        self._camera_user_modified = False
        self._error_text: str | None = None
        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_volume_data(self, data: LutVolumeData | None) -> None:
        """Set the 3D LUT volume data to render."""
        self._volume_data = data
        self._rebuild_payload()
        self._request_camera_fit()
        self.update()

    def set_projection_mode(self, mode: VolumeProjectionMode) -> None:
        """Set the camera preset used by the OpenGL volume view."""
        if self._projection_mode == mode:
            return
        self._projection_mode = mode
        self._request_camera_fit()
        self.update()

    def set_density(self, density: VolumeDensityPreset) -> None:
        """Set the volume render density preset."""
        if self._density == density:
            return
        self._density = density
        self._rebuild_payload()
        self._request_camera_fit()
        self.update()

    def set_use_output_positions(self, enabled: bool) -> None:
        """Choose whether points are positioned by output RGB or source RGB lattice."""
        if self._use_output_positions == enabled:
            return
        self._use_output_positions = enabled
        self._rebuild_payload()
        self._request_camera_fit()
        self.update()

    def set_show_neutral_axis(self, enabled: bool) -> None:
        """Choose whether neutral-axis samples should be highlighted."""
        if self._show_neutral_axis == enabled:
            return
        self._show_neutral_axis = enabled
        self.update()

    def set_show_rgb_axes(self, enabled: bool) -> None:
        """Choose whether RGB orientation axes are displayed."""
        if self._show_rgb_axes == enabled:
            return
        self._show_rgb_axes = enabled
        self.update()

    def set_point_size(self, point_size: float) -> None:
        """Set OpenGL point size."""
        if point_size <= 0.0:
            raise ValueError("point_size must be positive")
        self._point_size = point_size
        self.update()

    def set_opacity(self, opacity: float) -> None:
        """Set OpenGL point opacity in the inclusive [0, 1] range."""
        if opacity < 0.0 or opacity > 1.0:
            raise ValueError("opacity must be in [0, 1]")
        self._opacity = opacity
        self.update()

    def reset_view(self) -> None:
        """Reset camera to the current projection preset."""
        self._request_camera_fit()
        self._apply_pending_camera_fit()
        self.update()

    def status_text(self) -> str:
        """Return a concise human-readable OpenGL render status."""
        if self._error_text is not None:
            return self._error_text
        if self._volume_data is None or self._payload is None:
            return "OpenGL volume view requires a 3D LUT."
        return (
            f"OpenGL volume: {self._volume_data.size_x}x{self._volume_data.size_y}x{self._volume_data.size_z} "
            f"| shown: {self._payload.rendered_point_count}/{self._payload.total_point_count} "
            f"| density: {self._payload.density} "
            f"| position: {self._position_label()}"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start orbit or pan interaction."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.position()
            self._active_mouse_action = (
                "pan"
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                or self._projection_mode != "RGB isometric"
                else "orbit"
            )
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_mouse_pos = event.position()
            self._active_mouse_action = "pan"
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Apply active mouse interaction."""
        if self._last_mouse_pos is None or self._active_mouse_action is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._last_mouse_pos
        self._last_mouse_pos = event.position()
        if self._active_mouse_action == "orbit":
            self._orbit_by_pixels(float(delta.x()), float(delta.y()))
        elif self._active_mouse_action == "pan":
            self._pan_by_pixels(float(delta.x()), float(delta.y()))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End active mouse interaction."""
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._last_mouse_pos = None
            self._active_mouse_action = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Reset view on double click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom the view with the mouse wheel."""
        steps = float(event.angleDelta().y()) / 120.0
        if steps == 0.0:
            super().wheelEvent(event)
            return
        self._zoom_by_steps(steps)
        event.accept()

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        try:
            functions = self.context().functions()
            functions.initializeOpenGLFunctions()
            functions.glEnable(GL_DEPTH_TEST)
            functions.glEnable(GL_BLEND)
            functions.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            functions.glEnable(GL_PROGRAM_POINT_SIZE)
            self._program = self._create_program()
            self._vertex_array.create()
            self._position_buffer.create()
            self._color_buffer.create()
            self._axis_position_buffer.create()
            self._axis_color_buffer.create()
            self._upload_axis_buffers()
            self._gl_ready = True
            self._upload_payload_if_needed()
        except Exception as exc:
            self._program = None
            self._gl_ready = False
            self._error_text = f"OpenGL unavailable: {exc}"
            self.initialization_failed.emit(str(exc))

    def resizeGL(self, width: int, height: int) -> None:
        """Resize OpenGL viewport."""
        self.context().functions().glViewport(0, 0, width, height)
        if not self._camera_user_modified:
            self._needs_camera_fit = True

    def paintGL(self) -> None:
        """Paint the OpenGL point cloud."""
        functions = self.context().functions()
        functions.glClearColor(0.08, 0.08, 0.08, 1.0)
        functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self._gl_ready or self._payload is None or self._program is None:
            return
        self._upload_payload_if_needed()
        if self._payload.rendered_point_count <= 0:
            return
        self._apply_pending_camera_fit()

        self._program.bind()
        self._program.setUniformValue(self._program.uniformLocation(b"mvp"), self._mvp_matrix())
        opacity_location = self._program.uniformLocation(b"opacity")
        if opacity_location >= 0 and hasattr(functions, "glUniform1f"):
            functions.glUniform1f(opacity_location, float(self._opacity))
        point_size_location = self._program.uniformLocation(b"point_size")
        if point_size_location >= 0 and hasattr(functions, "glUniform1f"):
            functions.glUniform1f(point_size_location, float(self._point_size))

        if self._show_rgb_axes:
            self._draw_arrays(
                functions=functions,
                mode=GL_LINES,
                count=6,
                position_buffer=self._axis_position_buffer,
                color_buffer=self._axis_color_buffer,
            )

        self._draw_arrays(
            functions=functions,
            mode=GL_POINTS,
            count=self._payload.rendered_point_count,
            position_buffer=self._position_buffer,
            color_buffer=self._color_buffer,
        )
        self._program.release()

    def _draw_arrays(
        self,
        *,
        functions: QOpenGLFunctions,
        mode: int,
        count: int,
        position_buffer: QOpenGLBuffer,
        color_buffer: QOpenGLBuffer,
    ) -> None:
        if self._program is None:
            return
        position_location = self._program.attributeLocation(b"position")
        color_location = self._program.attributeLocation(b"color")
        self._vertex_array.bind()
        position_buffer.bind()
        self._program.enableAttributeArray(position_location)
        self._program.setAttributeBuffer(position_location, GL_FLOAT, 0, 3)
        color_buffer.bind()
        self._program.enableAttributeArray(color_location)
        self._program.setAttributeBuffer(color_location, GL_FLOAT, 0, 3)
        functions.glDrawArrays(mode, 0, count)
        self._program.disableAttributeArray(position_location)
        self._program.disableAttributeArray(color_location)
        color_buffer.release()
        position_buffer.release()
        self._vertex_array.release()

    def _rebuild_payload(self) -> None:
        self._payload = None
        self._error_text = None
        self._needs_upload = True
        if self._volume_data is None:
            return
        try:
            self._payload = build_lut_volume_render_payload(
                self._volume_data.values,
                density=self._density,
                use_output_positions=self._use_output_positions,
            )
        except ValueError as exc:
            self._error_text = f"OpenGL volume unavailable: {exc}"

    def _upload_payload_if_needed(self) -> None:
        if not self._gl_ready or not self._needs_upload or self._payload is None:
            return
        self._position_buffer.bind()
        self._position_buffer.allocate(
            self._payload.positions_rgb.tobytes(),
            int(self._payload.positions_rgb.nbytes),
        )
        self._position_buffer.release()
        self._color_buffer.bind()
        self._color_buffer.allocate(
            self._payload.colors_rgb.tobytes(),
            int(self._payload.colors_rgb.nbytes),
        )
        self._color_buffer.release()
        self._needs_upload = False

    def _upload_axis_buffers(self) -> None:
        axis_positions = bytes(np.asarray(AXIS_POSITIONS_RGB, dtype=np.float32))
        axis_colors = bytes(np.asarray(AXIS_COLORS_RGB, dtype=np.float32))
        self._axis_position_buffer.bind()
        self._axis_position_buffer.allocate(axis_positions, len(axis_positions))
        self._axis_position_buffer.release()
        self._axis_color_buffer.bind()
        self._axis_color_buffer.allocate(axis_colors, len(axis_colors))
        self._axis_color_buffer.release()

    def _create_program(self) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram(self)
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER):
            raise RuntimeError(program.log())
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            FRAGMENT_SHADER,
        ):
            raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        return program

    def _mvp_matrix(self) -> QMatrix4x4:
        aspect = self.width() / max(float(self.height()), 1.0)
        matrix = lut_volume_view_projection_matrix(self._camera, aspect_ratio=aspect)
        return QMatrix4x4(*[float(value) for value in matrix.reshape(-1)])

    def _position_label(self) -> str:
        return "Output cloud" if self._use_output_positions else "Source RGB lattice"

    def _request_camera_fit(self) -> None:
        self._needs_camera_fit = True
        self._camera_user_modified = False

    def _apply_pending_camera_fit(self) -> None:
        if not self._needs_camera_fit:
            return
        self._reset_camera_to_fit_payload()
        self._needs_camera_fit = False

    def _reset_camera_to_fit_payload(self) -> None:
        camera = reset_lut_volume_camera(self._projection_mode)
        if self._payload is None or self._payload.rendered_point_count <= 0:
            self._camera = camera
            return

        positions = self._fit_positions_rgb()
        bounds_min = positions.min(axis=0)
        bounds_max = positions.max(axis=0)
        center = (bounds_min + bounds_max) * np.float32(0.5)
        camera = replace(
            camera,
            pan_x=float(center[0] - np.float32(0.5)),
            pan_y=float(center[1] - np.float32(0.5)),
            pan_z=float(center[2] - np.float32(0.5)),
        )

        view = lut_volume_view_matrix(camera)
        homogeneous_positions = np.concatenate(
            (
                positions,
                np.ones((positions.shape[0], 1), dtype=np.float32),
            ),
            axis=1,
        )
        view_positions = homogeneous_positions @ view.T
        view_min = view_positions[:, :2].min(axis=0)
        view_max = view_positions[:, :2].max(axis=0)
        view_extent = view_max - view_min
        aspect = self.width() / max(float(self.height()), 1.0)
        required_height = max(float(view_extent[1]), float(view_extent[0]) / aspect)
        fitted_scale = float(
            np.clip(
                max(required_height * FIT_VIEW_MARGIN, DEFAULT_VIEW_SCALE),
                MIN_VIEW_SCALE,
                MAX_VIEW_SCALE,
            )
        )
        fitted_camera = replace(camera, view_scale=fitted_scale)
        if (
            self._projection_mode == "RGB isometric"
            and self._show_rgb_axes
            and self._use_output_positions
        ):
            fitted_camera = self._fit_camera_with_top_headroom(
                fitted_camera,
                positions,
                aspect=aspect,
            )
        self._camera = fitted_camera

    def _fit_camera_with_top_headroom(
        self,
        camera: LutVolumeCamera,
        positions: np.ndarray,
        *,
        aspect: float,
    ) -> LutVolumeCamera:
        ndc_y = self._project_positions_ndc_y(camera, positions, aspect=aspect)
        top = float(np.max(ndc_y))
        bottom = float(np.min(ndc_y))
        desired_shift_ndc = max(top - FIT_TARGET_TOP_NDC, 0.0)
        available_shift_ndc = max(bottom - FIT_BOTTOM_LIMIT_NDC, 0.0)
        shift_ndc = min(desired_shift_ndc, available_shift_ndc)
        if shift_ndc <= 0.0:
            return camera
        return pan_lut_volume_camera(
            camera,
            delta_x=0.0,
            delta_y=(camera.view_scale * shift_ndc) * 0.5,
        )

    def _project_positions_ndc_y(
        self,
        camera: LutVolumeCamera,
        positions: np.ndarray,
        *,
        aspect: float,
    ) -> np.ndarray:
        matrix = lut_volume_view_projection_matrix(camera, aspect_ratio=aspect)
        homogeneous_positions = np.concatenate(
            (
                positions,
                np.ones((positions.shape[0], 1), dtype=np.float32),
            ),
            axis=1,
        )
        projected = homogeneous_positions @ matrix.T
        return projected[:, 1] / projected[:, 3]

    def _fit_positions_rgb(self) -> np.ndarray:
        if self._payload is None:
            return np.empty((0, 3), dtype=np.float32)
        positions = np.asarray(self._payload.positions_rgb, dtype=np.float32)
        if not self._show_rgb_axes:
            return positions
        axis_positions = np.asarray(AXIS_POSITIONS_RGB, dtype=np.float32).reshape((-1, 3))
        return np.concatenate((positions, axis_positions), axis=0)

    def _orbit_by_pixels(self, delta_x: float, delta_y: float) -> None:
        self._needs_camera_fit = False
        self._camera_user_modified = True
        self._camera = orbit_lut_volume_camera(
            self._camera,
            delta_yaw_degrees=-(delta_x * ORBIT_DEGREES_PER_PIXEL),
            delta_pitch_degrees=delta_y * ORBIT_DEGREES_PER_PIXEL,
        )
        self.update()

    def _pan_by_pixels(self, delta_x: float, delta_y: float) -> None:
        self._needs_camera_fit = False
        self._camera_user_modified = True
        extent = max(min(self.width(), self.height()), 1)
        scale = (self._camera.view_scale * PAN_VIEW_SCALE_FACTOR) / float(extent)
        self._camera = pan_lut_volume_camera(
            self._camera,
            delta_x=-(delta_x * scale),
            delta_y=delta_y * scale,
        )
        self.update()

    def _zoom_by_steps(self, steps: float) -> None:
        self._needs_camera_fit = False
        self._camera_user_modified = True
        self._camera = zoom_lut_volume_camera(
            self._camera,
            scale_factor=WHEEL_ZOOM_BASE**steps,
        )
        self.update()
