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
        """Choose whether points are positioned by output RGB or input lattice RGB."""
        if self._use_output_positions == enabled:
            return
        self._use_output_positions = enabled
        self._rebuild_projection()
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
            f"| mode: {self._projection.mode}"
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
