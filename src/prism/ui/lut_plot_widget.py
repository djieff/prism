"""Plot widget for LUT inspection curves."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from prism.io.lut_loader import LutPlotData


class LutPlotWidget(QWidget):
    """Render LUT transfer curves on a scalable X/Y diagram."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot_data: LutPlotData | None = None
        self.setMinimumSize(480, 320)

    def set_plot_data(self, data: LutPlotData | None) -> None:
        self._plot_data = data
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        plot_rect = self._square_plot_rect()
        self._draw_grid_and_axes(painter, plot_rect)

        if self._plot_data is None:
            painter.setPen(QPen(QColor(170, 170, 170), 1))
            painter.drawText(plot_rect.toRect(), Qt.AlignCenter, "Drop a LUT file to plot curves")
            return

        self._draw_curves(painter, plot_rect, self._plot_data)

    def _draw_grid_and_axes(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        steps = 10
        for idx in range(1, steps):
            t = idx / float(steps)
            x = rect.left() + (rect.width() * t)
            y = rect.top() + (rect.height() * t)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        painter.setPen(QPen(QColor(180, 180, 180), 1))
        for idx in range(0, steps + 1):
            value = idx / float(steps)
            x = rect.left() + (rect.width() * value)
            y = rect.bottom() - (rect.height() * value)
            label = f"{value:.1f}"
            painter.drawText(int(x) - 10, int(rect.bottom()) + 18, label)
            painter.drawText(int(rect.left()) - 34, int(y) + 4, label)
        painter.restore()

    def _draw_curves(self, painter: QPainter, rect: QRectF, data: LutPlotData) -> None:
        x = data.x_values
        y = data.y_values
        if x.size == 0 or y.size == 0:
            return

        x_min = float(data.domain_min)
        x_max = float(data.domain_max)
        if abs(x_max - x_min) < 1e-8:
            x_min = 0.0
            x_max = 1.0

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if abs(y_max - y_min) < 1e-8:
            y_min -= 0.5
            y_max += 0.5

        channels = min(y.shape[1], 3)
        if channels == 1:
            colors = [QColor(240, 240, 240)]
        else:
            colors = [QColor(255, 90, 90), QColor(87, 217, 87), QColor(90, 160, 255)]
        for channel in range(channels):
            path = QPainterPath()
            for idx in range(x.shape[0]):
                px = self._map_x(float(x[idx]), x_min, x_max, rect)
                py = self._map_y(float(y[idx, channel]), y_min, y_max, rect)
                if idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.setPen(QPen(colors[channel], 2))
            painter.drawPath(path)

    def _map_x(self, value: float, x_min: float, x_max: float, rect: QRectF) -> float:
        t = (value - x_min) / max(x_max - x_min, 1e-8)
        return rect.left() + (rect.width() * max(0.0, min(1.0, t)))

    def _map_y(self, value: float, y_min: float, y_max: float, rect: QRectF) -> float:
        t = (value - y_min) / max(y_max - y_min, 1e-8)
        return rect.bottom() - (rect.height() * max(0.0, min(1.0, t)))

    def _square_plot_rect(self) -> QRectF:
        left_margin = 54.0
        right_margin = 20.0
        top_margin = 18.0
        bottom_margin = 34.0
        available_w = max(self.width() - left_margin - right_margin, 50.0)
        available_h = max(self.height() - top_margin - bottom_margin, 50.0)
        side = min(available_w, available_h)
        x = left_margin + ((available_w - side) * 0.5)
        y = top_margin + ((available_h - side) * 0.5)
        return QRectF(x, y, side, side)
