"""Volume projection widget for LUT inspection."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from prism.core.lut_volume_projection import (
    DEFAULT_VOLUME_SAMPLE_LIMIT,
    LutVolumeProjection,
    VolumeProjectionMode,
    project_lut_volume,
    select_neutral_axis_sample_mask,
)
from prism.io.lut_loader import LutVolumeData


class LutVolumeWidget(QWidget):
    """Render a projected 3D LUT volume as a 2D RGB cube point cloud."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._volume_data: LutVolumeData | None = None
        self._projection: LutVolumeProjection | None = None
        self._projection_mode: VolumeProjectionMode = "RGB isometric"
        self._sample_limit = DEFAULT_VOLUME_SAMPLE_LIMIT
        self._use_output_positions = True
        self._show_neutral_axis = True
        self._error_text: str | None = None
        self.setMinimumSize(480, 320)

    def set_volume_data(self, data: LutVolumeData | None) -> None:
        """Set the 3D LUT volume data to render."""
        self._volume_data = data
        self._rebuild_projection()
        self.update()

    def set_projection_mode(self, mode: VolumeProjectionMode) -> None:
        """Set the RGB-space projection preset."""
        if self._projection_mode == mode:
            return
        self._projection_mode = mode
        self._rebuild_projection()
        self.update()

    def set_sample_limit(self, sample_limit: int) -> None:
        """Set the maximum point count used for display projection."""
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        if self._sample_limit == sample_limit:
            return
        self._sample_limit = sample_limit
        self._rebuild_projection()
        self.update()

    def set_use_output_positions(self, enabled: bool) -> None:
        """Choose whether points are positioned by output RGB or source RGB lattice."""
        if self._use_output_positions == enabled:
            return
        self._use_output_positions = enabled
        self._rebuild_projection()
        self.update()

    def set_show_neutral_axis(self, enabled: bool) -> None:
        """Choose whether neutral-axis samples are highlighted over the point cloud."""
        if self._show_neutral_axis == enabled:
            return
        self._show_neutral_axis = enabled
        self.update()

    def status_text(self) -> str:
        """Return a concise human-readable projection status."""
        if self._error_text is not None:
            return self._error_text
        if self._volume_data is None or self._projection is None:
            return "Volume view requires a 3D LUT."
        return (
            f"Volume: {self._volume_data.size_x}x{self._volume_data.size_y}x{self._volume_data.size_z} "
            f"| shown: {self._projection.projected_point_count}/{self._projection.total_point_count} "
            f"| projection: {self._projection.mode} "
            f"| position: {self._position_label()}"
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        plot_rect = self._plot_rect()
        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        self._draw_frame(painter, plot_rect)
        if self._projection is None:
            painter.setPen(QPen(QColor(170, 170, 170), 1))
            painter.drawText(plot_rect.toRect(), Qt.AlignmentFlag.AlignCenter, self.status_text())
            return

        self._draw_points(painter, plot_rect, self._projection)
        self._draw_neutral_axis(painter, plot_rect, self._projection)
        self._draw_status(painter, plot_rect)

    def _rebuild_projection(self) -> None:
        self._projection = None
        self._error_text = None
        if self._volume_data is None:
            return
        try:
            self._projection = project_lut_volume(
                self._volume_data.values,
                mode=self._projection_mode,
                sample_limit=self._sample_limit,
                use_output_positions=self._use_output_positions,
            )
        except ValueError as exc:
            self._error_text = f"Volume unavailable: {exc}"

    def _position_label(self) -> str:
        return "Output cloud" if self._use_output_positions else "Source RGB lattice"

    def _draw_frame(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor(65, 65, 65), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(45, 45, 45), 1))
        steps = 4
        for idx in range(1, steps):
            t = idx / float(steps)
            x = rect.left() + (rect.width() * t)
            y = rect.top() + (rect.height() * t)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawText(int(rect.left()), int(rect.bottom()) + 18, "RGB cube projection")
        painter.restore()

        axis_labels = self._axis_labels()
        if axis_labels is not None:
            y_label, x_label = axis_labels
            painter.setPen(QPen(QColor(230, 230, 230), 1))
            painter.drawText(QPointF(rect.left() - 24.0, rect.top() + 8.0), y_label)
            painter.drawText(QPointF(rect.right() + 10.0, rect.bottom() + 2.0), x_label)

    def _draw_points(
        self,
        painter: QPainter,
        rect: QRectF,
        projection: LutVolumeProjection,
    ) -> None:
        xy = projection.xy
        colors = projection.colors_rgb
        if xy.size == 0:
            return

        point_radius = 2.0 if projection.projected_point_count <= 10_000 else 1.0
        for idx in range(xy.shape[0]):
            px = rect.left() + (rect.width() * float(xy[idx, 0]))
            py = rect.bottom() - (rect.height() * float(xy[idx, 1]))
            color = colors[idx]
            painter.setPen(
                QPen(
                    QColor(
                        int(float(color[0]) * 255.0),
                        int(float(color[1]) * 255.0),
                        int(float(color[2]) * 255.0),
                        190,
                    ),
                    point_radius,
                )
            )
            painter.drawPoint(QPointF(px, py))

    def _draw_neutral_axis(
        self,
        painter: QPainter,
        rect: QRectF,
        projection: LutVolumeProjection,
    ) -> None:
        if not self._show_neutral_axis or self._volume_data is None:
            return
        neutral_mask = select_neutral_axis_sample_mask(
            projection.sample_indices,
            size_x=self._volume_data.size_x,
            size_y=self._volume_data.size_y,
            size_z=self._volume_data.size_z,
        )
        if not bool(neutral_mask.any()):
            return

        painter.save()
        painter.setPen(QPen(QColor(255, 255, 255, 235), 1.6))
        for xy in projection.xy[neutral_mask]:
            px = rect.left() + (rect.width() * float(xy[0]))
            py = rect.bottom() - (rect.height() * float(xy[1]))
            painter.drawEllipse(QPointF(px, py), 4.0, 4.0)
        painter.restore()

    def _draw_status(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(160, 220, 160), 1))
        painter.drawText(int(rect.left()), int(rect.top()) - 8, self.status_text())

    def _plot_rect(self) -> QRectF:
        left_margin = 48.0
        right_margin = 18.0
        top_margin = 30.0
        bottom_margin = 34.0
        available_w = max(self.width() - left_margin - right_margin, 50.0)
        available_h = max(self.height() - top_margin - bottom_margin, 50.0)
        side = min(available_w, available_h)
        x = left_margin + ((available_w - side) * 0.5)
        y = top_margin + ((available_h - side) * 0.5)
        return QRectF(
            x,
            y,
            side,
            side,
        )

    def _axis_labels(self) -> tuple[str, str] | None:
        if self._projection_mode == "RG plane":
            return ("G", "R")
        if self._projection_mode == "RB plane":
            return ("B", "R")
        if self._projection_mode == "GB plane":
            return ("B", "G")
        return None
