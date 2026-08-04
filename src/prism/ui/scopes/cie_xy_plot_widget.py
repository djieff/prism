"""CIE 1931 xy chromaticity plot widget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from prism.core.scopes.cie_xy import (
    CIE_XY_X_RANGE,
    CIE_XY_Y_RANGE,
    CieGamutOverlay,
    CieWhitePoint,
    CieXyTrace,
    get_cie_rgb_space,
    get_cie_whitepoint,
    spectral_locus_xy,
    unique_cie_gamut_overlay_names,
)

OVERLAY_LEGEND_WIDTH = 132.0
OVERLAY_LEGEND_ROW_HEIGHT = 18.0
OVERLAY_LEGEND_MARGIN = 12.0
OVERLAY_LEGEND_ROW_GAP = 5.0
WHITEPOINT_COLOR = QColor(230, 230, 210)
TraceColorMode = Literal["normal", "boosted"]


@dataclass(frozen=True)
class CieXyPlotPoint:
    """Pixel-space position for one xy coordinate."""

    x: float
    y: float


@dataclass(frozen=True)
class CieXyOverlayLabel:
    """Label placement for one gamut overlay."""

    text: str
    point: CieXyPlotPoint
    color: QColor


def closed_spectral_locus_polygon(locus_xy: np.ndarray) -> np.ndarray:
    """Return spectral locus closed by the line of purples."""
    locus = np.asarray(locus_xy, dtype=np.float32)
    if locus.ndim != 2 or locus.shape[1] != 2:
        raise ValueError("Expected spectral locus shape (N, 2)")
    if locus.shape[0] < 3:
        raise ValueError("Expected at least three spectral locus points")
    if not np.all(np.isfinite(locus)):
        raise ValueError("Spectral locus must contain only finite values")
    if np.array_equal(locus[0], locus[-1]):
        return np.ascontiguousarray(locus, dtype=np.float32)
    return np.ascontiguousarray(np.vstack((locus, locus[0])), dtype=np.float32)


def xy_in_polygon(x_value: float, y_value: float, polygon_xy: np.ndarray) -> bool:
    """Return whether an xy coordinate is inside a closed polygon."""
    polygon = np.asarray(polygon_xy, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Expected polygon shape (N, 2)")
    if polygon.shape[0] < 4:
        raise ValueError("Expected at least four closed polygon points")

    x = float(x_value)
    y = float(y_value)
    inside = False
    previous_x = float(polygon[-1, 0])
    previous_y = float(polygon[-1, 1])
    for current_x_raw, current_y_raw in polygon:
        current_x = float(current_x_raw)
        current_y = float(current_y_raw)
        crosses_y = (current_y > y) != (previous_y > y)
        if crosses_y:
            x_intersection = ((previous_x - current_x) * (y - current_y) / (previous_y - current_y)) + current_x
            if x < x_intersection:
                inside = not inside
        previous_x = current_x
        previous_y = current_y
    return inside


def boosted_trace_rgb(source_rgb: np.ndarray) -> np.ndarray:
    """Return brighter, more saturated source RGB for scope trace rendering."""
    rgb = np.clip(np.asarray(source_rgb, dtype=np.float32), 0.0, 1.0)
    luma = (
        (rgb[:, :, 0:1] * np.float32(0.2126))
        + (rgb[:, :, 1:2] * np.float32(0.7152))
        + (rgb[:, :, 2:3] * np.float32(0.0722))
    )
    saturated = luma + ((rgb - luma) * np.float32(2.2))
    brightened = np.power(np.clip(saturated, 0.0, 1.0), np.float32(0.82))
    return np.ascontiguousarray(brightened, dtype=np.float32)


class CieXyPlotWidget(QWidget):
    """Render a CIE 1931 xy graticule, gamut overlays, and image density."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._trace: CieXyTrace | None = None
        self._drop_highlight = False
        self._density_image: QImage | None = None
        self._density_image_buffer: np.ndarray | None = None
        self._overlay_names: tuple[str, ...] = ()
        self._whitepoint: CieWhitePoint | None = get_cie_whitepoint("D65")
        self._trace_color_mode: TraceColorMode = "boosted"
        self._spectral_locus = spectral_locus_xy()
        self._horseshoe_polygon = closed_spectral_locus_polygon(self._spectral_locus)
        self.setMinimumSize(320, 320)

    def set_trace(self, trace: CieXyTrace | None) -> None:
        """Update CIE xy trace and rebuild the cached density image."""
        self._trace = trace
        self._density_image = None
        self._density_image_buffer = None
        if trace is not None:
            self._rebuild_density_image(trace)
        self.update()

    def set_gamut_overlay_names(self, names: tuple[str | None, ...]) -> None:
        """Set global gamut overlay selections, de-duplicating at render time."""
        overlay_names = unique_cie_gamut_overlay_names(names)
        if self._overlay_names == overlay_names:
            return
        self._overlay_names = overlay_names
        self.update()

    def set_drop_target_highlight(self, highlighted: bool) -> None:
        """Set drop-target highlight state for drag/drop feedback."""
        if self._drop_highlight == highlighted:
            return
        self._drop_highlight = highlighted
        self.update()

    def set_whitepoint(self, whitepoint: CieWhitePoint | None) -> None:
        """Set the reference white point marker, or hide it when absent."""
        if self._whitepoint == whitepoint:
            return
        self._whitepoint = whitepoint
        self.update()

    def set_trace_color_mode(self, mode: TraceColorMode) -> None:
        """Set display-only source color rendering mode for the density trace."""
        if mode not in ("normal", "boosted"):
            raise ValueError(f"Unsupported CIE xy trace color mode: {mode!r}")
        if self._trace_color_mode == mode:
            return
        self._trace_color_mode = mode
        self._density_image = None
        self._density_image_buffer = None
        if self._trace is not None:
            self._rebuild_density_image(self._trace)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Segoe UI", 8))
        painter.fillRect(self.rect(), QColor(18, 18, 18))

        plot_rect = self._plot_rect(self.rect())
        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        if self._density_image is not None:
            painter.drawImage(plot_rect, self._density_image)
        self._draw_grid(painter, plot_rect)
        self._draw_spectral_locus(painter, plot_rect)
        self._draw_gamut_overlays(painter, plot_rect)
        if self._whitepoint is not None:
            self._draw_whitepoint(painter, plot_rect, self._whitepoint)
            self._draw_whitepoint_label(painter, plot_rect, self._whitepoint)
        self._draw_border(painter, plot_rect)

        if self._trace is None:
            painter.setPen(QPen(QColor(170, 170, 170), 1))
            painter.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, "No CIE xy data")

    def _rebuild_density_image(self, trace: CieXyTrace) -> None:
        density = np.asarray(trace.density, dtype=np.float32)
        if density.ndim != 2:
            raise ValueError("Expected CIE xy density shape (H, W)")
        if not np.all(np.isfinite(density)):
            raise ValueError("CIE xy density must contain only finite values")
        if np.any(density < 0.0):
            raise ValueError("CIE xy density must be non-negative")

        color_density = np.asarray(trace.color_density, dtype=np.float32)
        if color_density.shape != (*density.shape, 3):
            raise ValueError("Expected CIE xy color density shape (H, W, 3)")
        if not np.all(np.isfinite(color_density)):
            raise ValueError("CIE xy color density must contain only finite values")
        if np.any(color_density < 0.0):
            raise ValueError("CIE xy color density must be non-negative")

        maximum = float(np.max(density))
        if maximum > 0.0:
            density = density * np.float32(1.0 / maximum)
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(density, sigma=1.0, mode="constant", output=np.float32)
            filtered_color_density = np.zeros_like(color_density)
            for channel in range(3):
                filtered_color_density[:, :, channel] = gaussian_filter(
                    color_density[:, :, channel],
                    sigma=1.0,
                    mode="constant",
                    output=np.float32,
                )
            filtered_maximum = float(np.max(density))
            if filtered_maximum > 0.0:
                density = density * np.float32(1.0 / filtered_maximum)
                filtered_color_density = filtered_color_density * np.float32(1.0 / filtered_maximum)
        else:
            filtered_color_density = np.zeros_like(color_density)
        linear_density = np.clip(density, 0.0, 1.0)
        display_density = np.power(linear_density, 0.34)

        source_rgb = np.divide(
            filtered_color_density,
            np.maximum(linear_density[:, :, None], np.float32(1e-6)),
            out=np.zeros_like(filtered_color_density),
            where=linear_density[:, :, None] > np.float32(1e-6),
        )
        if self._trace_color_mode == "boosted":
            source_rgb = boosted_trace_rgb(source_rgb)
        else:
            source_rgb = np.clip(source_rgb, 0.0, 1.0)

        rgba = np.zeros((*display_density.shape, 4), dtype=np.uint8)
        alpha = np.clip(display_density * 255.0, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 0:3] = np.clip(source_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 3] = alpha

        self._density_image_buffer = np.ascontiguousarray(rgba)
        h, w = self._density_image_buffer.shape[:2]
        self._density_image = QImage(
            self._density_image_buffer.data,
            w,
            h,
            int(self._density_image_buffer.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()

    def _draw_grid(self, painter: QPainter, plot_rect: QRectF) -> None:
        grid_pen = QPen(QColor(8, 8, 8, 95), 1)
        axis_pen = QPen(QColor(12, 12, 12, 155), 1)
        label_pen = QPen(QColor(184, 190, 190), 1)

        painter.setPen(grid_pen)
        for x in np.arange(CIE_XY_X_RANGE[0], CIE_XY_X_RANGE[1] + 0.001, 0.1):
            point = self._plot_point(plot_rect, float(x), CIE_XY_Y_RANGE[0])
            painter.drawLine(QPointF(point.x, plot_rect.top()), QPointF(point.x, plot_rect.bottom()))
        for y in np.arange(CIE_XY_Y_RANGE[0], CIE_XY_Y_RANGE[1] + 0.001, 0.1):
            point = self._plot_point(plot_rect, CIE_XY_X_RANGE[0], float(y))
            painter.drawLine(QPointF(plot_rect.left(), point.y), QPointF(plot_rect.right(), point.y))

        painter.setPen(axis_pen)
        origin = self._plot_point(plot_rect, 0.0, 0.0)
        painter.drawLine(QPointF(origin.x, plot_rect.top()), QPointF(origin.x, plot_rect.bottom()))
        painter.drawLine(QPointF(plot_rect.left(), origin.y), QPointF(plot_rect.right(), origin.y))

        painter.setPen(label_pen)
        painter.drawText(int(plot_rect.right()) - 14, int(plot_rect.bottom()) + 20, "x")
        painter.drawText(int(plot_rect.left()) - 24, int(plot_rect.top()) + 12, "y")
        for x in (0.0, 0.2, 0.4, 0.6, 0.8):
            point = self._plot_point(plot_rect, x, 0.0)
            painter.drawText(int(point.x) - 10, int(plot_rect.bottom()) + 16, f"{x:.1f}")
        for y in (0.0, 0.3, 0.6, 0.9):
            point = self._plot_point(plot_rect, 0.0, y)
            painter.drawText(int(plot_rect.left()) - 30, int(point.y) + 4, f"{y:.1f}")

    def _draw_spectral_locus(self, painter: QPainter, plot_rect: QRectF) -> None:
        polygon = self._horseshoe_polygon
        if polygon.shape[0] < 4:
            return
        path = QPainterPath()
        first = self._plot_point(plot_rect, float(polygon[0, 0]), float(polygon[0, 1]))
        path.moveTo(first.x, first.y)
        for x_value, y_value in polygon[1:]:
            point = self._plot_point(plot_rect, float(x_value), float(y_value))
            path.lineTo(point.x, point.y)

        painter.setPen(QPen(QColor(210, 215, 205), 2))
        painter.drawPath(path)

    def _draw_gamut_overlays(self, painter: QPainter, plot_rect: QRectF) -> None:
        for index, overlay in enumerate(self._selected_overlays()):
            points = self._overlay_polygon(plot_rect, overlay)
            if len(points) != 4:
                continue
            painter.setPen(QPen(self._overlay_color(index), 2))
            for start, end in zip(points[:-1], points[1:], strict=True):
                painter.drawLine(QPointF(start.x, start.y), QPointF(end.x, end.y))
        self._draw_gamut_overlay_labels(painter, plot_rect)

    def _draw_gamut_overlay_labels(self, painter: QPainter, plot_rect: QRectF) -> None:
        for index, overlay in enumerate(self._selected_overlays()):
            label = self._overlay_label(plot_rect, overlay, index)
            label_rect = QRectF(
                label.point.x,
                label.point.y,
                OVERLAY_LEGEND_WIDTH,
                OVERLAY_LEGEND_ROW_HEIGHT,
            )

            painter.fillRect(label_rect, QColor(8, 8, 8, 165))
            painter.setPen(QPen(label.color, 1))
            painter.drawRect(label_rect)
            painter.drawText(label_rect.adjusted(4.0, 0.0, -4.0, 0.0), Qt.AlignmentFlag.AlignVCenter, label.text)

    def _draw_whitepoint(self, painter: QPainter, plot_rect: QRectF, whitepoint: CieWhitePoint) -> None:
        point = self._plot_point(plot_rect, whitepoint.xy[0], whitepoint.xy[1])
        painter.setPen(QPen(WHITEPOINT_COLOR, 1))
        painter.drawLine(QPointF(point.x - 5.0, point.y), QPointF(point.x + 5.0, point.y))
        painter.drawLine(QPointF(point.x, point.y - 5.0), QPointF(point.x, point.y + 5.0))

    def _draw_whitepoint_label(self, painter: QPainter, plot_rect: QRectF, whitepoint: CieWhitePoint) -> None:
        label = self._whitepoint_label(plot_rect, whitepoint)
        label_rect = QRectF(
            label.point.x,
            label.point.y,
            OVERLAY_LEGEND_WIDTH,
            OVERLAY_LEGEND_ROW_HEIGHT,
        )

        painter.fillRect(label_rect, QColor(8, 8, 8, 165))
        painter.setPen(QPen(label.color, 1))
        painter.drawRect(label_rect)
        painter.drawText(
            label_rect.adjusted(4.0, 0.0, -4.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter,
            label.text,
        )

    def _draw_border(self, painter: QPainter, plot_rect: QRectF) -> None:
        border = QColor(72, 174, 255) if self._drop_highlight else QColor(92, 102, 106)
        painter.setPen(QPen(border, 2 if self._drop_highlight else 1))
        painter.drawRect(plot_rect)

    def _selected_overlays(self) -> tuple[CieGamutOverlay, ...]:
        overlays: list[CieGamutOverlay] = []
        for name in self._overlay_names:
            rgb_space = get_cie_rgb_space(name)
            overlays.append(
                CieGamutOverlay(
                    name=rgb_space.name,
                    primaries_xy=rgb_space.primaries_xy,
                    whitepoint_xy=rgb_space.whitepoint_xy,
                )
            )
        return tuple(overlays)

    def _overlay_polygon(self, plot_rect: QRectF, overlay: CieGamutOverlay) -> tuple[CieXyPlotPoint, ...]:
        points = tuple(self._plot_point(plot_rect, x_value, y_value) for x_value, y_value in overlay.primaries_xy)
        if not points:
            return ()
        return (*points, points[0])

    def _overlay_label(
        self,
        plot_rect: QRectF,
        overlay: CieGamutOverlay,
        index: int,
    ) -> CieXyOverlayLabel:
        point = CieXyPlotPoint(
            float(plot_rect.right()) - OVERLAY_LEGEND_MARGIN - OVERLAY_LEGEND_WIDTH,
            float(plot_rect.top())
            + OVERLAY_LEGEND_MARGIN
            + (float(index) * (OVERLAY_LEGEND_ROW_HEIGHT + OVERLAY_LEGEND_ROW_GAP)),
        )
        return CieXyOverlayLabel(
            text=self._overlay_label_text(overlay.name),
            point=point,
            color=self._overlay_color(index),
        )

    def _whitepoint_label(self, plot_rect: QRectF, whitepoint: CieWhitePoint) -> CieXyOverlayLabel:
        index = len(self._selected_overlays())
        point = CieXyPlotPoint(
            float(plot_rect.right()) - OVERLAY_LEGEND_MARGIN - OVERLAY_LEGEND_WIDTH,
            float(plot_rect.top())
            + OVERLAY_LEGEND_MARGIN
            + (float(index) * (OVERLAY_LEGEND_ROW_HEIGHT + OVERLAY_LEGEND_ROW_GAP)),
        )
        return CieXyOverlayLabel(
            text=f"White: {whitepoint.name}",
            point=point,
            color=WHITEPOINT_COLOR,
        )

    def _overlay_color(self, index: int) -> QColor:
        colors = (
            QColor(255, 88, 72),
            QColor(60, 220, 92),
            QColor(96, 148, 255),
        )
        return colors[index % len(colors)]

    def _overlay_label_text(self, name: str) -> str:
        labels = {
            "ITU-R BT.709": "BT.709",
            "ITU-R BT.2020": "BT.2020",
            "Adobe RGB (1998)": "Adobe RGB",
        }
        return labels.get(name, name)

    def _plot_rect(self, bounds: QRect) -> QRectF:
        left_margin = 42.0
        top_margin = 18.0
        right_margin = 16.0
        bottom_margin = 34.0
        available_width = max(float(bounds.width()) - left_margin - right_margin, 0.0)
        available_height = max(float(bounds.height()) - top_margin - bottom_margin, 0.0)
        side = min(available_width, available_height)
        left = float(bounds.left()) + left_margin + ((available_width - side) * 0.5)
        top = float(bounds.top()) + top_margin + ((available_height - side) * 0.5)
        return QRectF(
            left,
            top,
            side,
            side,
        )

    def _plot_point(self, plot_rect: QRectF, x_value: float, y_value: float) -> CieXyPlotPoint:
        x_min, x_max = CIE_XY_X_RANGE
        y_min, y_max = CIE_XY_Y_RANGE
        x_fraction = (float(x_value) - x_min) / (x_max - x_min)
        y_fraction = (float(y_value) - y_min) / (y_max - y_min)
        return CieXyPlotPoint(
            x=float(plot_rect.left()) + (x_fraction * float(plot_rect.width())),
            y=float(plot_rect.bottom()) - (y_fraction * float(plot_rect.height())),
        )
