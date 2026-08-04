"""Modeless CIE xy chromaticity monitor window."""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from prism.core.scopes.cie_xy import (
    DEFAULT_CIE_RGB_SPACE,
    CieXyMode,
    available_cie_rgb_space_names,
    available_cie_whitepoint_names,
    build_cie_xy_view_data,
    get_cie_whitepoint,
)
from prism.core.viewer_state import ViewerSide
from prism.ui.scopes.cie_xy_plot_widget import CieXyPlotWidget, TraceColorMode


class CieXyWindow(QWidget):
    """CIE xy monitor for A, B, or A|B side-by-side chromaticity display."""

    def __init__(
        self,
        parent: QWidget | None = None,
        on_drop_file: Callable[[str, ViewerSide], None] | None = None,
        on_source_mode_changed: Callable[[CieXyMode], None] | None = None,
    ) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("CIE xy Chromaticity")
        self.resize(980, 620)
        self.setAcceptDrops(True)

        self._buffer_a: np.ndarray | None = None
        self._buffer_b: np.ndarray | None = None
        self._mode: CieXyMode = "A"
        self._overlay_names: tuple[str | None, str | None, str | None] = (
            "sRGB",
            "ACEScg",
            "ARRI Wide Gamut 4",
        )
        self._whitepoint_name: str | None = "D65"
        self._trace_color_mode: TraceColorMode = "normal"
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

        self._gamut_combos: list[QComboBox] = []
        for label, default_name in zip(
            ("Gamut 1", "Gamut 2", "Gamut 3"),
            self._overlay_names,
            strict=True,
        ):
            controls.addWidget(QLabel(label, self))
            combo = QComboBox(self)
            combo.addItem("None", None)
            for name in available_cie_rgb_space_names():
                combo.addItem(name, name)
            combo.setCurrentIndex(combo.findData(default_name))
            combo.currentIndexChanged.connect(self._on_gamut_overlay_changed)
            controls.addWidget(combo)
            self._gamut_combos.append(combo)

        controls.addWidget(QLabel("White Point", self))
        self._whitepoint_combo = QComboBox(self)
        self._whitepoint_combo.addItem("None", None)
        for name in available_cie_whitepoint_names():
            self._whitepoint_combo.addItem(name, name)
        self._whitepoint_combo.setCurrentIndex(self._whitepoint_combo.findData(self._whitepoint_name))
        self._whitepoint_combo.currentIndexChanged.connect(
            self._on_whitepoint_changed
        )
        controls.addWidget(self._whitepoint_combo)

        controls.addWidget(QLabel("Trace Color", self))
        self._trace_color_combo = QComboBox(self)
        self._trace_color_combo.addItem("Normal", "normal")
        self._trace_color_combo.addItem("Boosted", "boosted")
        self._trace_color_combo.setCurrentIndex(
            self._trace_color_combo.findData(self._trace_color_mode)
        )
        self._trace_color_combo.currentIndexChanged.connect(self._on_trace_color_changed)
        controls.addWidget(self._trace_color_combo)

        controls.addStretch(1)
        root.addLayout(controls)

        self._plots_row = QHBoxLayout()
        self._plots_row.setSpacing(8)
        self._plot_a = CieXyPlotWidget(self)
        self._plot_b = CieXyPlotWidget(self)
        self._plots_row.addWidget(self._plot_a, 1)
        self._plots_row.addWidget(self._plot_b, 1)
        root.addLayout(self._plots_row, 1)

        self._status_label = QLabel("Waiting for image data", self)
        self._status_label.setStyleSheet("color: #b0b0b0;")
        root.addWidget(self._status_label)

        self._apply_overlay_names_to_plots()
        self._apply_whitepoint_to_plots()
        self._apply_trace_color_mode_to_plots()
        self._refresh_view_data()

    def current_mode(self) -> CieXyMode:
        """Return currently selected CIE xy mode."""
        return self._mode

    def current_gamut_overlay_names(self) -> tuple[str | None, str | None, str | None]:
        """Return the three global gamut overlay combo selections."""
        return self._overlay_names

    def set_source_mode(self, mode: CieXyMode) -> None:
        """Set CIE xy source mode from external state without emitting callbacks."""
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
        """Update source buffers and rebuild CIE xy data for current mode."""
        self._buffer_a = buffer_a
        self._buffer_b = buffer_b
        self._refresh_view_data()

    def set_unsupported_main_mode(self, mode: str | None) -> None:
        """Set/clear unsupported main-view mode override for CIE xy display."""
        self._unsupported_main_mode = mode
        self._refresh_view_data()

    def _on_mode_changed(self) -> None:
        mode = cast(CieXyMode, self._mode_combo.currentData())
        self._mode = mode
        if self._on_source_mode_changed is not None:
            self._on_source_mode_changed(mode)
        self._refresh_view_data()

    def _on_gamut_overlay_changed(self) -> None:
        self._overlay_names = cast(
            tuple[str | None, str | None, str | None],
            tuple(combo.currentData() for combo in self._gamut_combos),
        )
        self._apply_overlay_names_to_plots()

    def _on_whitepoint_changed(self) -> None:
        self._whitepoint_name = cast(str | None, self._whitepoint_combo.currentData())
        self._apply_whitepoint_to_plots()

    def _on_trace_color_changed(self) -> None:
        self._trace_color_mode = cast(TraceColorMode, self._trace_color_combo.currentData())
        self._apply_trace_color_mode_to_plots()

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
            view_data = build_cie_xy_view_data(
                self._mode,
                self._buffer_a,
                self._buffer_b,
                rgb_space_name=DEFAULT_CIE_RGB_SPACE,
            )
        except ValueError as exc:
            self._plot_a.set_trace(None)
            self._plot_b.set_trace(None)
            self._status_label.setText(f"CIE xy unavailable: {exc}")
            self._status_label.setStyleSheet("color: #ff8f8f;")
            return

        self._status_label.setText(view_data.status)
        self._status_label.setStyleSheet(
            "color: #8fdf8f;" if "CIE xy:" in view_data.status else "color: #b0b0b0;"
        )

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

    def _apply_overlay_names_to_plots(self) -> None:
        self._plot_a.set_gamut_overlay_names(self._overlay_names)
        self._plot_b.set_gamut_overlay_names(self._overlay_names)

    def _apply_whitepoint_to_plots(self) -> None:
        whitepoint = get_cie_whitepoint(self._whitepoint_name) if self._whitepoint_name else None
        self._plot_a.set_whitepoint(whitepoint)
        self._plot_b.set_whitepoint(whitepoint)

    def _apply_trace_color_mode_to_plots(self) -> None:
        self._plot_a.set_trace_color_mode(self._trace_color_mode)
        self._plot_b.set_trace_color_mode(self._trace_color_mode)

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
