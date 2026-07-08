"""Waveform density plot widget."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from prism.core.scope_waveform import WaveformTrace
from prism.core.scope_waveform_science import (
    WAVEFORM_DENSITY_FILTER_SIGMA,
    WaveformDensityChannels,
    prepare_waveform_densities_for_render,
)

SignalMode = Literal["R", "G", "B", "RGB Parade", "RGB Overlay", "Y'"]


class WaveformPlotWidget(QWidget):
    """Render RGB waveform density as an additive heatmap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._trace: WaveformTrace | None = None
        self._signal_mode: SignalMode = "RGB Overlay"
        self._drop_highlight = False
        self._heatmap: QImage | None = None
        self._heatmap_buffer: np.ndarray | None = None
        self._prepared_densities: WaveformDensityChannels | None = None
        self.setMinimumSize(280, 220)

    def set_trace(self, trace: WaveformTrace | None) -> None:
        """Update waveform trace and rebuild the cached heatmap image."""
        self._trace = trace
        self._heatmap = None
        self._heatmap_buffer = None
        self._prepared_densities = None
        if trace is not None:
            self._rebuild_heatmap(trace, self._signal_mode)
        self.update()

    def set_signal_mode(self, mode: SignalMode) -> None:
        """Set signal display mode and refresh rendered heatmap."""
        if self._signal_mode == mode:
            return
        self._signal_mode = mode
        self._heatmap = None
        self._heatmap_buffer = None
        if self._trace is not None:
            self._rebuild_heatmap(self._trace, mode)
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
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        plot_rect = self.rect().adjusted(36, 14, -12, -28)
        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        if self._heatmap is None:
            border = QColor(72, 174, 255) if self._drop_highlight else QColor(65, 65, 65)
            painter.setPen(QPen(border, 2 if self._drop_highlight else 1))
            painter.drawRect(plot_rect)
            self._draw_axis_labels(painter, plot_rect)
            painter.setPen(QPen(QColor(170, 170, 170), 1))
            painter.drawText(plot_rect, Qt.AlignCenter, "No waveform data")
            return

        painter.drawImage(plot_rect, self._heatmap)
        border = QColor(72, 174, 255) if self._drop_highlight else QColor(65, 65, 65)
        painter.setPen(QPen(border, 2 if self._drop_highlight else 1))
        painter.drawRect(plot_rect)
        self._draw_axis_labels(painter, plot_rect)

    def _draw_axis_labels(self, painter: QPainter, plot_rect: QRectF) -> None:
        steps = 10
        painter.setPen(QPen(QColor(75, 75, 75), 1))
        for step in range(1, steps):
            t = step / float(steps)
            y = plot_rect.bottom() - (plot_rect.height() * t)
            painter.drawLine(int(plot_rect.left()), int(y), int(plot_rect.right()), int(y))

        painter.setPen(QPen(QColor(180, 180, 180), 1))
        for step in range(0, steps + 1):
            value = step * 10
            t = step / float(steps)
            y = plot_rect.bottom() - (plot_rect.height() * t)
            painter.drawText(int(plot_rect.left()) - 30, int(y) + 4, f"{value}")

        painter.drawText(int(plot_rect.left()), int(plot_rect.bottom()) + 18, "X")

    def _rebuild_heatmap(self, trace: WaveformTrace, mode: SignalMode) -> None:
        r, g, b, y_prime = self._prepared_density_channels(trace)

        if mode == "R":
            rgb = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=2)
        elif mode == "G":
            rgb = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=2)
        elif mode == "B":
            rgb = np.stack([np.zeros_like(b), np.zeros_like(b), b], axis=2)
        elif mode == "Y'":
            rgb = np.stack([y_prime, y_prime, y_prime], axis=2)
        elif mode == "RGB Parade":
            rgb = self._build_rgb_parade_image(r, g, b)
        else:
            rgb = np.stack([r, g, b], axis=2)

        rgb = np.power(rgb, 0.5)
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

    def _prepared_density_channels(self, trace: WaveformTrace) -> WaveformDensityChannels:
        """Return cached filtered density copies for the current trace."""
        if self._prepared_densities is None:
            self._prepared_densities = prepare_waveform_densities_for_render(
                (
                    trace.density_r,
                    trace.density_g,
                    trace.density_b,
                    trace.density_luma,
                ),
                sigma=WAVEFORM_DENSITY_FILTER_SIGMA,
            )
        return self._prepared_densities

    def _build_rgb_parade_image(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        h, w = r.shape
        seg_width = max(w // 3, 1)
        out = np.zeros((h, seg_width * 3, 3), dtype=np.float32)

        def _resample_x(src: np.ndarray) -> np.ndarray:
            x_idx = np.linspace(0, src.shape[1] - 1, seg_width, dtype=np.int32)
            return src[:, x_idx]

        rr = _resample_x(r)
        gg = _resample_x(g)
        bb = _resample_x(b)

        out[:, 0:seg_width, 0] = rr
        out[:, seg_width : seg_width * 2, 1] = gg
        out[:, seg_width * 2 : seg_width * 3, 2] = bb
        return out
