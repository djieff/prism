"""Modeless waveform monitor window."""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from prism.core.scope_waveform import WaveformMode, build_waveform_view_data
from prism.core.scope_waveform_science import (
    DEFAULT_WAVEFORM_SIGNAL_STANDARD,
    SUPPORTED_WAVEFORM_SIGNAL_STANDARDS,
    WaveformSignalStandard,
)
from prism.core.viewer_state import ViewerSide
from prism.ui.waveform_plot_widget import SignalMode, WaveformPlotWidget


class WaveformWindow(QWidget):
    """Waveform monitor for A, B, or A|B side-by-side scope display."""

    def __init__(
        self,
        parent: QWidget | None = None,
        on_drop_file: Callable[[str, ViewerSide], None] | None = None,
        on_source_mode_changed: Callable[[WaveformMode], None] | None = None,
    ) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Waveform Monitor")
        self.resize(900, 520)
        self.setAcceptDrops(True)

        self._buffer_a: np.ndarray | None = None
        self._buffer_b: np.ndarray | None = None
        self._mode: WaveformMode = "A"
        self._signal_mode: SignalMode = "RGB Overlay"
        self._signal_standard: WaveformSignalStandard = DEFAULT_WAVEFORM_SIGNAL_STANDARD
        self._on_drop_file = on_drop_file
        self._on_source_mode_changed = on_source_mode_changed
        self._unsupported_main_mode: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode", self))
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem("Full (A)", "A")
        self._mode_combo.addItem("Full (B)", "B")
        self._mode_combo.addItem("Split", "A|B")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)
        controls.addWidget(QLabel("Signal", self))
        self._signal_combo = QComboBox(self)
        self._signal_combo.addItem("R", "R")
        self._signal_combo.addItem("G", "G")
        self._signal_combo.addItem("B", "B")
        self._signal_combo.addItem("RGB Parade", "RGB Parade")
        self._signal_combo.addItem("RGB Overlay", "RGB Overlay")
        self._signal_combo.addItem("Y'", "Y'")
        self._signal_combo.setCurrentText("RGB Overlay")
        self._signal_combo.currentIndexChanged.connect(self._on_signal_mode_changed)
        controls.addWidget(self._signal_combo)
        controls.addWidget(QLabel("Y' Standard", self))
        self._signal_standard_combo = QComboBox(self)
        for standard in SUPPORTED_WAVEFORM_SIGNAL_STANDARDS:
            self._signal_standard_combo.addItem(standard.removeprefix("ITU-R "), standard)
        self._signal_standard_combo.currentIndexChanged.connect(
            self._on_signal_standard_changed
        )
        controls.addWidget(self._signal_standard_combo)
        controls.addStretch(1)
        root.addLayout(controls)

        self._plots_row = QHBoxLayout()
        self._plots_row.setSpacing(8)
        self._plot_a = WaveformPlotWidget(self)
        self._plot_b = WaveformPlotWidget(self)
        self._plots_row.addWidget(self._plot_a, 1)
        self._plots_row.addWidget(self._plot_b, 1)
        root.addLayout(self._plots_row, 1)

        self._status_label = QLabel("Waiting for image data", self)
        self._status_label.setStyleSheet("color: #b0b0b0;")
        root.addWidget(self._status_label)

        self._refresh_view_data()

    def current_mode(self) -> WaveformMode:
        """Return currently selected waveform mode."""
        return self._mode

    def current_signal_standard(self) -> WaveformSignalStandard:
        """Return the selected encoded Y' signal standard."""
        return self._signal_standard

    def set_source_mode(self, mode: WaveformMode) -> None:
        """Set waveform source mode from external state without emitting callbacks."""
        if self._mode == mode:
            return
        index = self._mode_combo.findData(mode)
        if index < 0:
            return
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)
        self._mode = mode
        self._refresh_view_data()

    def set_processed_buffers(
        self, buffer_a: np.ndarray | None, buffer_b: np.ndarray | None
    ) -> None:
        """Update source buffers and rebuild waveform data for current mode."""
        self._buffer_a = buffer_a
        self._buffer_b = buffer_b
        self._refresh_view_data()

    def set_unsupported_main_mode(self, mode: str | None) -> None:
        """Set/clear unsupported main-view mode override for waveform display."""
        self._unsupported_main_mode = mode
        self._refresh_view_data()

    def _on_mode_changed(self) -> None:
        mode = cast(WaveformMode, self._mode_combo.currentData())
        self._mode = mode
        if self._on_source_mode_changed is not None:
            self._on_source_mode_changed(mode)
        self._refresh_view_data()

    def _on_signal_mode_changed(self) -> None:
        mode = cast(SignalMode, self._signal_combo.currentData())
        self._signal_mode = mode
        self._plot_a.set_signal_mode(mode)
        self._plot_b.set_signal_mode(mode)
        self._refresh_view_data()

    def _on_signal_standard_changed(self) -> None:
        standard = cast(
            WaveformSignalStandard,
            self._signal_standard_combo.currentData(),
        )
        self._signal_standard = standard
        self._refresh_view_data()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if urls and any(url.isLocalFile() for url in urls):
            event.acceptProposedAction()
            return
        self._set_drop_target_highlight(None)
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        urls = event.mimeData().urls()
        if not urls or not any(url.isLocalFile() for url in urls):
            self._set_drop_target_highlight(None)
            event.ignore()
            return
        side = self._target_side_for_window_pos(event.position())
        self._set_drop_target_highlight(side)
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._set_drop_target_highlight(None)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = [url for url in event.mimeData().urls() if url.isLocalFile()]
        if not urls:
            self._set_drop_target_highlight(None)
            event.ignore()
            return
        side = self._target_side_for_window_pos(event.position())
        self._set_drop_target_highlight(None)
        if side is None:
            self._status_label.setText("Drop target unavailable for current mode")
            self._status_label.setStyleSheet("color: #ffcf8f;")
            event.ignore()
            return
        if self._on_drop_file is not None:
            for url in urls:
                self._on_drop_file(url.toLocalFile(), side)
        event.acceptProposedAction()

    def _refresh_view_data(self) -> None:
        if self._unsupported_main_mode is not None:
            self._plot_a.show()
            self._plot_b.show()
            self._plot_a.set_trace(None)
            self._plot_b.set_trace(None)
            self._status_label.setText(
                f"Unsupported mode: {self._unsupported_main_mode}. Use Full (A), Full (B), or Split."
            )
            self._status_label.setStyleSheet("color: #ffcf8f;")
            return

        try:
            view_data = build_waveform_view_data(
                self._mode,
                self._buffer_a,
                self._buffer_b,
                signal_standard=self._signal_standard,
            )
        except ValueError as exc:
            self._plot_a.set_trace(None)
            self._plot_b.set_trace(None)
            self._status_label.setText(f"Waveform unavailable: {exc}")
            self._status_label.setStyleSheet("color: #ff8f8f;")
            return

        standard_label = self._signal_standard.removeprefix("ITU-R ")
        self._status_label.setText(f"{view_data.status} | Y' standard: {standard_label}")
        self._status_label.setStyleSheet("color: #8fdf8f;" if "Waveform:" in view_data.status else "color: #b0b0b0;")

        if self._mode == "A":
            self._plot_a.show()
            self._plot_b.hide()
            self._plot_a.set_trace(view_data.trace_a)
            self._plot_b.set_trace(None)
            return
        if self._mode == "B":
            self._plot_a.hide()
            self._plot_b.show()
            self._plot_a.set_trace(None)
            self._plot_b.set_trace(view_data.trace_b)
            return

        self._plot_a.show()
        self._plot_b.show()
        self._plot_a.set_trace(view_data.trace_a)
        self._plot_b.set_trace(view_data.trace_b)

    def _target_side_for_window_pos(self, window_pos: QPointF) -> ViewerSide | None:
        if self._mode == "A":
            return "left"
        if self._mode == "B":
            return "right"
        point = window_pos.toPoint()
        if self._plot_a.geometry().contains(point):
            return "left"
        if self._plot_b.geometry().contains(point):
            return "right"
        return None

    def _set_drop_target_highlight(self, side: ViewerSide | None) -> None:
        self._plot_a.set_drop_target_highlight(side == "left")
        self._plot_b.set_drop_target_highlight(side == "right")
