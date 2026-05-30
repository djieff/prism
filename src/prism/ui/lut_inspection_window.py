"""Modeless LUT inspection window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from prism.io.lut_loader import LutLoadError, load_lut_plot_data
from prism.ui.lut_plot_widget import LutPlotWidget


class LutInspectionWindow(QWidget):
    """Modeless window that loads and plots LUT transfer curves."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("LUT Inspection")
        self.resize(760, 760)
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        self._browse_button = QPushButton("Open LUT...", self)
        self._browse_button.clicked.connect(self._browse_lut_file)
        header.addWidget(self._browse_button)

        self._file_label = QLabel("No LUT loaded", self)
        self._file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self._file_label, 1)
        root.addLayout(header)

        self._plot_widget = LutPlotWidget(self)
        root.addWidget(self._plot_widget, 1)

        self._info_panel = QPlainTextEdit(self)
        self._info_panel.setReadOnly(True)
        self._info_panel.setMaximumBlockCount(32)
        self._info_panel.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._info_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._info_panel.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._info_panel.setPlaceholderText("LUT info will appear here")
        self._set_info_panel_text("")
        root.addWidget(self._info_panel)

        self._status_label = QLabel(
            "Drop a .cube, .csp, .spi1d, .spi3d, .3dl, or .lut file here",
            self,
        )
        self._status_label.setStyleSheet("color: #b0b0b0;")
        root.addWidget(self._status_label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        path = Path(urls[0].toLocalFile())
        self._load_lut(path)
        event.acceptProposedAction()

    def _browse_lut_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LUT File",
            "",
            "LUT Files (*.cube *.csp *.spi1d *.spi3d *.3dl *.lut);;All Files (*)",
        )
        if not file_path:
            return
        self._load_lut(Path(file_path))

    def load_lut_path(self, path: Path) -> None:
        """Load a LUT file path into the inspector."""
        self._load_lut(path)

    def _load_lut(self, path: Path) -> None:
        try:
            data = load_lut_plot_data(path)
        except (OSError, ValueError, LutLoadError) as exc:
            self._plot_widget.set_plot_data(None)
            self._file_label.setText(str(path))
            self._set_info_panel_text("")
            self._status_label.setText(f"Failed to load LUT: {exc}")
            self._status_label.setStyleSheet("color: #ff8f8f;")
            return

        self._plot_widget.set_plot_data(data)
        self._file_label.setText(str(path))
        self._set_info_panel_text(self._build_info_text(data))
        if data.source_kind == "3d_neutral_axis":
            source_label = "3D neutral-axis projection"
        elif data.source_kind == "1d":
            source_label = "1D transfer curve"
        else:
            source_label = data.source_kind
        self._status_label.setText(
            f"Loaded {data.format.upper()} ({source_label}), channels: {data.channels}"
        )
        self._status_label.setStyleSheet("color: #8fdf8f;")

    def _build_info_text(self, data) -> str:
        y = data.y_values
        if y.size == 0:
            return "No LUT sample data."

        ch = min(max(int(data.channels), 1), y.shape[1])
        y_used = y[:, :ch]

        y_min = np.min(y_used, axis=0)
        y_max = np.max(y_used, axis=0)
        channel_names = ["R", "G", "B"] if ch > 1 else ["Y"]
        min_text = ", ".join(
            f"{channel_names[idx]}={float(y_min[idx]):.4f}" for idx in range(ch)
        )
        max_text = ", ".join(
            f"{channel_names[idx]}={float(y_max[idx]):.4f}" for idx in range(ch)
        )

        below_zero = bool(np.any(y_used < 0.0))
        above_one = bool(np.any(y_used > 1.0))
        clipped = "yes" if (below_zero or above_one) else "no"

        diffs = np.diff(y_used, axis=0)
        monotonic_channels: list[str] = []
        for idx in range(ch):
            monotonic_channels.append("yes" if bool(np.all(diffs[:, idx] >= -1e-6)) else "no")
        mono_text = ", ".join(f"{channel_names[idx]}={monotonic_channels[idx]}" for idx in range(ch))

        return "\n".join(
            [
                f"Format: {data.format.upper()}",
                f"Source: {data.source_kind}",
                f"Samples: {data.x_values.shape[0]}",
                f"Channels: {ch}",
                f"Domain: [{data.domain_min:.6f}, {data.domain_max:.6f}]",
                f"Output Min: {min_text}",
                f"Output Max: {max_text}",
                f"Out of [0,1]: {clipped}",
                f"Monotonic: {mono_text}",
            ]
            + (
                [f"CSP Shaper: {'yes' if data.csp_has_shaper else 'no'}"]
                if data.csp_has_shaper is not None
                else []
            )
        )

    def _set_info_panel_text(self, text: str) -> None:
        self._info_panel.setPlainText(text)
        self._resize_info_panel_to_content()

    def _resize_info_panel_to_content(self) -> None:
        doc = self._info_panel.document()
        font_metrics = self._info_panel.fontMetrics()
        line_height = max(font_metrics.lineSpacing(), 14)
        lines = max(int(doc.blockCount()), 1)
        margins = 12
        target_height = (line_height * lines) + margins
        self._info_panel.setFixedHeight(target_height)
