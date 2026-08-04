"""Vectorscope density plot widget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from prism.core.scopes.vectorscope import VectorscopeTrace, vectorscope_chroma_scale
from prism.core.scopes.waveform_science import (
    DEFAULT_WAVEFORM_SIGNAL_STANDARD,
    WaveformSignalStandard,
    waveform_y_prime_coefficients,
)

VectorscopeColorMode = Literal["Teal", "Normal", "Boosted"]


def _boosted_vectorscope_rgb(source_rgb: np.ndarray) -> np.ndarray:
    """Return brighter, more saturated source RGB for vectorscope rendering."""
    rgb = np.clip(np.asarray(source_rgb, dtype=np.float32), 0.0, 1.0)
    luma = (
        (rgb[:, :, 0:1] * np.float32(0.2126))
        + (rgb[:, :, 1:2] * np.float32(0.7152))
        + (rgb[:, :, 2:3] * np.float32(0.0722))
    )
    saturated = luma + ((rgb - luma) * np.float32(2.2))
    brightened = np.power(np.clip(saturated, 0.0, 1.0), np.float32(0.82))
    return np.ascontiguousarray(brightened, dtype=np.float32)


@dataclass(frozen=True)
class VectorscopeTarget:
    """Presentation target for one chroma reference label."""

    label: str
    x: float
    y: float


class VectorscopePlotWidget(QWidget):
    """Render vectorscope chroma density with a custom digital graticule."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._trace: VectorscopeTrace | None = None
        self._color_mode: VectorscopeColorMode = "Normal"
        self._drop_highlight = False
        self._heatmap: QImage | None = None
        self._heatmap_buffer: np.ndarray | None = None
        self._signal_standard: WaveformSignalStandard = DEFAULT_WAVEFORM_SIGNAL_STANDARD
        self._target_coefficients = self._coefficients_for_standard(self._signal_standard)
        self._target_plot_scale = vectorscope_chroma_scale(self._target_coefficients)
        self.setMinimumSize(280, 280)

    def set_trace(self, trace: VectorscopeTrace | None) -> None:
        """Update vectorscope trace and rebuild the cached heatmap image."""
        self._trace = trace
        self._heatmap = None
        self._heatmap_buffer = None
        if trace is not None:
            self._signal_standard = trace.signal_standard
            self._target_coefficients = trace.y_prime_coefficients
            self._target_plot_scale = trace.plot_scale
            self._rebuild_heatmap(trace)
        self.update()

    def set_color_mode(self, mode: VectorscopeColorMode) -> None:
        """Set trace color presentation mode."""
        if mode not in ("Teal", "Normal", "Boosted"):
            raise ValueError(f"Unsupported vectorscope color mode: {mode!r}")
        if self._color_mode == mode:
            return
        self._color_mode = mode
        self._heatmap = None
        self._heatmap_buffer = None
        if self._trace is not None:
            self._rebuild_heatmap(self._trace)
        self.update()

    def set_signal_standard(self, standard: WaveformSignalStandard) -> None:
        """Set the graticule target standard for empty and future trace states."""
        if self._signal_standard == standard:
            return
        self._signal_standard = standard
        self._target_coefficients = self._coefficients_for_standard(standard)
        self._target_plot_scale = vectorscope_chroma_scale(self._target_coefficients)
        self.update()

    def set_drop_target_highlight(self, highlighted: bool) -> None:
        """Set drop-target highlight state for drag/drop feedback."""
        if self._drop_highlight == highlighted:
            return
        self._drop_highlight = highlighted
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(18, 18, 18))

        scope_rect = self._scope_rect(self.rect())
        if scope_rect.width() <= 0 or scope_rect.height() <= 0:
            return

        if self._heatmap is not None:
            painter.drawImage(scope_rect, self._heatmap)

        self._draw_graticule(painter, scope_rect)

    def _rebuild_heatmap(self, trace: VectorscopeTrace) -> None:
        density = np.asarray(trace.density, dtype=np.float32)
        if density.ndim != 2:
            raise ValueError("Expected vectorscope density shape (H, W)")
        if not np.all(np.isfinite(density)):
            raise ValueError("Vectorscope density must contain only finite values")
        if np.any(density < 0.0):
            raise ValueError("Vectorscope density must be non-negative")

        maximum = float(np.max(density))
        if maximum > 0.0:
            density = density * np.float32(1.0 / maximum)
        filtered_color_density = None
        if maximum > 0.0:
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(density, sigma=1.1, mode="constant", output=np.float32)
            if self._color_mode in ("Normal", "Boosted"):
                color_density = self._validated_color_density(trace.color_density, density.shape)
                filtered_color_density = np.zeros_like(color_density)
                for channel in range(3):
                    filtered_color_density[:, :, channel] = gaussian_filter(
                        color_density[:, :, channel],
                        sigma=1.1,
                        mode="constant",
                        output=np.float32,
                    )
            filtered_maximum = float(np.max(density))
            if filtered_maximum > 0.0:
                density = density * np.float32(1.0 / filtered_maximum)
                if filtered_color_density is not None:
                    filtered_color_density = filtered_color_density * np.float32(1.0 / filtered_maximum)
        elif self._color_mode in ("Normal", "Boosted"):
            filtered_color_density = np.zeros((*density.shape, 3), dtype=np.float32)

        linear_density = np.clip(density, 0.0, 1.0)
        display_density = np.power(linear_density, 0.45)

        if self._color_mode in ("Normal", "Boosted"):
            if filtered_color_density is None:
                filtered_color_density = self._validated_color_density(trace.color_density, linear_density.shape)
            source_rgb = np.divide(
                filtered_color_density,
                np.maximum(linear_density[:, :, None], np.float32(1e-6)),
                out=np.zeros_like(filtered_color_density),
                where=linear_density[:, :, None] > np.float32(1e-6),
            )
            if self._color_mode == "Boosted":
                source_rgb = _boosted_vectorscope_rgb(source_rgb)
            else:
                source_rgb = np.clip(source_rgb, 0.0, 1.0)
            rgb = source_rgb * display_density[:, :, None]
        else:
            rgb = np.zeros((*display_density.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = display_density * np.float32(0.25)
            rgb[:, :, 1] = display_density * np.float32(0.95)
            rgb[:, :, 2] = display_density * np.float32(0.72)

        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        self._heatmap_buffer = np.ascontiguousarray(rgb)
        h, w = self._heatmap_buffer.shape[:2]
        self._heatmap = QImage(
            self._heatmap_buffer.data,
            w,
            h,
            int(self._heatmap_buffer.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

    def _validated_color_density(self, color_density: np.ndarray, density_shape: tuple[int, int]) -> np.ndarray:
        rgb = np.asarray(color_density, dtype=np.float32)
        if rgb.shape != (*density_shape, 3):
            raise ValueError("Expected vectorscope color density shape (H, W, 3)")
        if not np.all(np.isfinite(rgb)):
            raise ValueError("Vectorscope color density must contain only finite values")
        if np.any(rgb < 0.0):
            raise ValueError("Vectorscope color density must be non-negative")
        return rgb

    def _draw_graticule(self, painter: QPainter, scope_rect: QRectF) -> None:
        center = scope_rect.center()
        radius = min(scope_rect.width(), scope_rect.height()) * 0.5

        grid_pen = QPen(QColor(62, 78, 78), 1)
        axis_pen = QPen(QColor(96, 130, 130), 1)
        border = QColor(72, 174, 255) if self._drop_highlight else QColor(88, 104, 104)
        border_pen = QPen(border, 2 if self._drop_highlight else 1)

        painter.setPen(grid_pen)
        for fraction in (0.25, 0.5, 0.75):
            ring_radius = radius * fraction
            ring = QRectF(
                center.x() - ring_radius,
                center.y() - ring_radius,
                ring_radius * 2.0,
                ring_radius * 2.0,
            )
            painter.drawEllipse(ring)

        for degrees in range(0, 360, 30):
            angle = math.radians(float(degrees))
            end = QPointF(
                center.x() + math.cos(angle) * radius,
                center.y() - math.sin(angle) * radius,
            )
            painter.drawLine(center, end)

        self._draw_outer_ticks(painter, scope_rect)

        painter.setPen(axis_pen)
        painter.drawLine(QPointF(scope_rect.left(), center.y()), QPointF(scope_rect.right(), center.y()))
        painter.drawLine(QPointF(center.x(), scope_rect.top()), QPointF(center.x(), scope_rect.bottom()))

        painter.setPen(border_pen)
        painter.drawEllipse(scope_rect)

        self._draw_targets(painter, scope_rect)

    def _draw_outer_ticks(self, painter: QPainter, scope_rect: QRectF) -> None:
        center = scope_rect.center()
        radius = min(scope_rect.width(), scope_rect.height()) * 0.5
        painter.setPen(QPen(QColor(82, 96, 96), 1))
        for degrees in range(0, 360, 5):
            angle = math.radians(float(degrees))
            tick_length = 10.0 if degrees % 30 == 0 else 5.0
            outer = QPointF(
                center.x() + math.cos(angle) * radius,
                center.y() - math.sin(angle) * radius,
            )
            inner = QPointF(
                center.x() + math.cos(angle) * (radius - tick_length),
                center.y() - math.sin(angle) * (radius - tick_length),
            )
            painter.drawLine(inner, outer)

    def _draw_targets(self, painter: QPainter, scope_rect: QRectF) -> None:
        painter.setPen(QPen(QColor(205, 210, 190), 1))
        for target in self._targets_for_current_standard():
            point = self._scope_point(scope_rect, target.x, target.y)
            self._draw_target_box(painter, point)
            label_rect = QRectF(point.x() - 14.0, point.y() - 28.0, 28.0, 16.0)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, target.label)

    def _draw_target_box(self, painter: QPainter, center: QPointF) -> None:
        size = 20.0
        corner = 6.0
        left = center.x() - (size * 0.5)
        right = center.x() + (size * 0.5)
        top = center.y() - (size * 0.5)
        bottom = center.y() + (size * 0.5)

        painter.drawLine(QPointF(left, top + corner), QPointF(left, top))
        painter.drawLine(QPointF(left, top), QPointF(left + corner, top))
        painter.drawLine(QPointF(right - corner, top), QPointF(right, top))
        painter.drawLine(QPointF(right, top), QPointF(right, top + corner))
        painter.drawLine(QPointF(right, bottom - corner), QPointF(right, bottom))
        painter.drawLine(QPointF(right, bottom), QPointF(right - corner, bottom))
        painter.drawLine(QPointF(left + corner, bottom), QPointF(left, bottom))
        painter.drawLine(QPointF(left, bottom), QPointF(left, bottom - corner))

    def _targets_for_trace(self, trace: VectorscopeTrace) -> tuple[VectorscopeTarget, ...]:
        return self._targets_from_coefficients(trace.y_prime_coefficients, trace.plot_scale)

    def _targets_for_current_standard(self) -> tuple[VectorscopeTarget, ...]:
        return self._targets_from_coefficients(
            self._target_coefficients,
            self._target_plot_scale,
        )

    def _targets_from_coefficients(
        self,
        coefficients: tuple[float, float, float],
        plot_scale: float,
    ) -> tuple[VectorscopeTarget, ...]:
        return tuple(
            VectorscopeTarget(
                label,
                *(value / plot_scale for value in self._chroma_for_rgb(rgb, coefficients)),
            )
            for label, rgb in (
                ("R", (1.0, 0.0, 0.0)),
                ("Y", (1.0, 1.0, 0.0)),
                ("G", (0.0, 1.0, 0.0)),
                ("Cy", (0.0, 1.0, 1.0)),
                ("B", (0.0, 0.0, 1.0)),
                ("Mg", (1.0, 0.0, 1.0)),
            )
        )

    def _coefficients_for_standard(
        self,
        standard: WaveformSignalStandard,
    ) -> tuple[float, float, float]:
        coefficients = waveform_y_prime_coefficients(standard)
        return (float(coefficients[0]), float(coefficients[1]), float(coefficients[2]))

    def _chroma_for_rgb(
        self,
        rgb: tuple[float, float, float],
        coefficients: tuple[float, float, float],
    ) -> tuple[float, float]:
        r, g, b = rgb
        kr, kg, kb = coefficients
        y_prime = (kr * r) + (kg * g) + (kb * b)
        cb = 0.5 * ((b - y_prime) / (1.0 - kb))
        cr = 0.5 * ((r - y_prime) / (1.0 - kr))
        return cb * 2.0, cr * 2.0

    def _scope_rect(self, bounds: QRect) -> QRectF:
        margin = 24.0
        side = float(max(min(bounds.width(), bounds.height()) - int(margin * 2.0), 0))
        left = bounds.left() + ((bounds.width() - side) * 0.5)
        top = bounds.top() + ((bounds.height() - side) * 0.5)
        return QRectF(left, top, side, side)

    def _scope_point(self, scope_rect: QRectF, x: float, y: float) -> QPointF:
        center = scope_rect.center()
        radius = min(scope_rect.width(), scope_rect.height()) * 0.5
        return QPointF(
            center.x() + (float(x) * radius),
            center.y() - (float(y) * radius),
        )
