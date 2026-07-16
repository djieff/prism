"""Camera math for interactive 3D LUT volume viewing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from prism.core.lut.volume_projection import VolumeProjectionMode

FloatArray = NDArray[np.float32]

DEFAULT_CAMERA_DISTANCE = 3.0
DEFAULT_VIEW_SCALE = 1.35
MIN_CAMERA_DISTANCE = 0.25
MAX_CAMERA_DISTANCE = 20.0
MIN_VIEW_SCALE = 0.1
MAX_VIEW_SCALE = 10.0
MIN_PITCH_DEGREES = -89.0
MAX_PITCH_DEGREES = 89.0
RGB_CUBE_CENTER = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
WORLD_UP_RGB = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
BLUE_UP_RGB = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
WORLD_UP_TUPLE: tuple[float, float, float] = (0.0, 1.0, 0.0)
BLUE_UP_TUPLE: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class LutVolumeCamera:
    """UI-agnostic camera state for an interactive RGB volume view."""

    yaw_degrees: float
    pitch_degrees: float
    distance: float = DEFAULT_CAMERA_DISTANCE
    pan_x: float = 0.0
    pan_y: float = 0.0
    pan_z: float = 0.0
    view_scale: float = DEFAULT_VIEW_SCALE
    up_rgb: tuple[float, float, float] = (0.0, 1.0, 0.0)

    @property
    def target_rgb(self) -> FloatArray:
        """Return the current RGB-space look target including pan offsets."""
        return np.asarray(
            [
                RGB_CUBE_CENTER[0] + np.float32(self.pan_x),
                RGB_CUBE_CENTER[1] + np.float32(self.pan_y),
                RGB_CUBE_CENTER[2] + np.float32(self.pan_z),
            ],
            dtype=np.float32,
        )


def reset_lut_volume_camera(mode: VolumeProjectionMode = "RGB isometric") -> LutVolumeCamera:
    """Return the default camera preset for a volume projection mode."""
    if mode == "RGB isometric":
        return LutVolumeCamera(
            yaw_degrees=45.0,
            pitch_degrees=35.26438968,
            up_rgb=WORLD_UP_TUPLE,
        )
    if mode == "RG plane":
        return LutVolumeCamera(
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            up_rgb=WORLD_UP_TUPLE,
        )
    if mode == "RB plane":
        return LutVolumeCamera(
            yaw_degrees=0.0,
            pitch_degrees=-90.0,
            up_rgb=BLUE_UP_TUPLE,
        )
    if mode == "GB plane":
        return LutVolumeCamera(
            yaw_degrees=90.0,
            pitch_degrees=0.0,
            up_rgb=BLUE_UP_TUPLE,
        )
    raise ValueError(f"Unsupported LUT volume camera preset: {mode!r}")


def orbit_lut_volume_camera(
    camera: LutVolumeCamera,
    *,
    delta_yaw_degrees: float,
    delta_pitch_degrees: float,
) -> LutVolumeCamera:
    """Return camera state after orbiting around its current target."""
    return replace(
        camera,
        yaw_degrees=_normalize_degrees(camera.yaw_degrees + delta_yaw_degrees),
        pitch_degrees=float(
            np.clip(
                camera.pitch_degrees + delta_pitch_degrees,
                MIN_PITCH_DEGREES,
                MAX_PITCH_DEGREES,
            )
        ),
    )


def pan_lut_volume_camera(
    camera: LutVolumeCamera,
    *,
    delta_x: float,
    delta_y: float,
) -> LutVolumeCamera:
    """Return camera state after applying a normalized screen-space pan."""
    right_rgb, up_rgb = _camera_screen_basis(camera)
    offset = (right_rgb * np.float32(delta_x)) + (up_rgb * np.float32(delta_y))
    return replace(
        camera,
        pan_x=camera.pan_x + float(offset[0]),
        pan_y=camera.pan_y + float(offset[1]),
        pan_z=camera.pan_z + float(offset[2]),
    )


def zoom_lut_volume_camera(camera: LutVolumeCamera, *, scale_factor: float) -> LutVolumeCamera:
    """Return camera state after scaling the orthographic view size."""
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    return replace(
        camera,
        view_scale=float(np.clip(camera.view_scale * scale_factor, MIN_VIEW_SCALE, MAX_VIEW_SCALE)),
    )


def dolly_lut_volume_camera(camera: LutVolumeCamera, *, distance_factor: float) -> LutVolumeCamera:
    """Return camera state after scaling camera distance from its target."""
    if distance_factor <= 0.0:
        raise ValueError("distance_factor must be positive")
    return replace(
        camera,
        distance=float(
            np.clip(camera.distance * distance_factor, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE)
        ),
    )


def lut_volume_view_matrix(camera: LutVolumeCamera) -> FloatArray:
    """Return a right-handed view matrix for the camera."""
    eye = camera.target_rgb + (_camera_direction(camera) * np.float32(camera.distance))
    return _look_at(
        eye=eye,
        target=camera.target_rgb,
        up=np.asarray(camera.up_rgb, dtype=np.float32),
    )


def lut_volume_orthographic_projection_matrix(
    camera: LutVolumeCamera,
    *,
    aspect_ratio: float,
) -> FloatArray:
    """Return an orthographic projection matrix for the camera view scale."""
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")
    half_height = np.float32(camera.view_scale * 0.5)
    half_width = np.float32(half_height * np.float32(aspect_ratio))
    near = np.float32(0.01)
    far = np.float32(MAX_CAMERA_DISTANCE + 5.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = np.float32(1.0) / half_width
    matrix[1, 1] = np.float32(1.0) / half_height
    matrix[2, 2] = np.float32(-2.0) / (far - near)
    matrix[2, 3] = -(far + near) / (far - near)
    matrix[3, 3] = np.float32(1.0)
    return matrix


def lut_volume_view_projection_matrix(
    camera: LutVolumeCamera,
    *,
    aspect_ratio: float,
) -> FloatArray:
    """Return projection * view matrix for OpenGL-style rendering."""
    return np.asarray(
        lut_volume_orthographic_projection_matrix(camera, aspect_ratio=aspect_ratio)
        @ lut_volume_view_matrix(camera),
        dtype=np.float32,
    )


def _camera_direction(camera: LutVolumeCamera) -> FloatArray:
    yaw = np.deg2rad(np.float32(camera.yaw_degrees))
    pitch = np.deg2rad(np.float32(camera.pitch_degrees))
    cos_pitch = np.cos(pitch)
    return _normalized(
        np.asarray(
            [
                cos_pitch * np.sin(yaw),
                np.sin(pitch),
                cos_pitch * np.cos(yaw),
            ],
            dtype=np.float32,
        )
    )


def _camera_screen_basis(camera: LutVolumeCamera) -> tuple[FloatArray, FloatArray]:
    forward = -_camera_direction(camera)
    right = _normalized(np.cross(forward, np.asarray(camera.up_rgb, dtype=np.float32)))
    corrected_up = np.cross(right, forward)
    return right, np.asarray(corrected_up, dtype=np.float32)


def _look_at(*, eye: FloatArray, target: FloatArray, up: FloatArray) -> FloatArray:
    forward = _normalized(target - eye)
    right = _normalized(np.cross(forward, up))
    corrected_up = np.cross(right, forward)

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = corrected_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(corrected_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def _normalized(vector: FloatArray) -> FloatArray:
    norm = np.linalg.norm(vector)
    if norm <= np.float32(1e-8):
        raise ValueError("camera vector cannot be normalized")
    return np.asarray(vector / norm, dtype=np.float32)


def _normalize_degrees(value: float) -> float:
    return float(((value + 180.0) % 360.0) - 180.0)
