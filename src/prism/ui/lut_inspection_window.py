"""Modeless LUT inspection window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from prism.core.lut_analysis import LutAnalysisSummary, summarize_lut_samples
from prism.io.lut_loader import LutLoadError, LutPlotData, LutVolumeData, load_lut_inspection_data
from prism.ui.lut_plot_widget import LutPlotWidget
from prism.ui.lut_volume_widget import LutVolumeWidget


class LutInspectionWindow(QWidget):
    """Modeless window that loads and plots LUT transfer curves."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
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
        self._file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._file_label, 1)
        root.addLayout(header)

        self._view_tabs = QTabWidget(self)
        self._plot_widget = LutPlotWidget(self)
        self._volume_widget = LutVolumeWidget(self)
        self._view_tabs.addTab(self._plot_widget, "Curves")
        self._view_tabs.addTab(self._volume_widget, "Volume")
        root.addWidget(self._view_tabs, 1)

        self._info_panel = QPlainTextEdit(self)
        self._info_panel.setReadOnly(True)
        self._info_panel.setMaximumBlockCount(32)
        self._info_panel.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_panel.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
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
            data = load_lut_inspection_data(path)
        except (OSError, ValueError, LutLoadError) as exc:
            self._plot_widget.set_plot_data(None)
            self._volume_widget.set_volume_data(None)
            self._file_label.setText(str(path))
            self._set_info_panel_text("")
            self._status_label.setText(f"Failed to load LUT: {exc}")
            self._status_label.setStyleSheet("color: #ff8f8f;")
            return

        self._plot_widget.set_plot_data(data.plot)
        self._volume_widget.set_volume_data(data.volume)
        self._file_label.setText(str(path))
        self._set_info_panel_text(self._build_info_text(data.plot, data.volume))
        if data.plot.source_kind == "3d_neutral_axis":
            source_label = "3D neutral-axis projection"
        elif data.plot.source_kind == "1d":
            source_label = "1D transfer curve"
        else:
            source_label = data.plot.source_kind
        volume_label = (
            f", volume: {data.volume.size_x}x{data.volume.size_y}x{data.volume.size_z}"
            if data.volume is not None
            else ", volume: unavailable"
        )
        self._status_label.setText(
            f"Loaded {data.plot.format.upper()} ({source_label}), channels: {data.plot.channels}{volume_label}"
        )
        self._status_label.setStyleSheet("color: #8fdf8f;")

    def _build_info_text(self, data: LutPlotData, volume: LutVolumeData | None = None) -> str:
        summary = summarize_lut_samples(
            data.x_values,
            data.y_values,
            channels=data.channels,
        )
        if summary is None:
            return "No LUT sample data."

        min_text, max_text, clipped, mono_text = self._format_summary_values(summary)
        lines = [
            f"Format: {data.format.upper()}",
            f"Source: {data.source_kind}",
            f"Samples: {summary.sample_count}",
            f"Channels: {summary.channel_count}",
            f"Domain: [{data.domain_min:.6f}, {data.domain_max:.6f}]",
            f"Output Min: {min_text}",
            f"Output Max: {max_text}",
            f"Out of [0,1]: {clipped}",
            f"Monotonic: {mono_text}",
        ]
        if volume is not None:
            lines.extend(
                [
                    f"3D Size: {volume.size_x} x {volume.size_y} x {volume.size_z}",
                    f"Volume Samples: {volume.size_x * volume.size_y * volume.size_z}",
                ]
            )
        else:
            lines.append("Volume: unavailable")
        if data.csp_has_shaper is not None:
            lines.append(f"CSP Shaper: {'yes' if data.csp_has_shaper else 'no'}")
        return "\n".join(lines)

    def _format_summary_values(self, summary: LutAnalysisSummary) -> tuple[str, str, str, str]:
        channel_names = ["R", "G", "B"] if summary.channel_count > 1 else ["Y"]
        min_text = ", ".join(
            f"{channel_names[idx]}={summary.output_min[idx]:.4f}"
            for idx in range(summary.channel_count)
        )
        max_text = ", ".join(
            f"{channel_names[idx]}={summary.output_max[idx]:.4f}"
            for idx in range(summary.channel_count)
        )
        clipped = (
            "yes"
            if (summary.has_values_below_zero or summary.has_values_above_one)
            else "no"
        )
        mono_text = ", ".join(
            f"{channel_names[idx]}={'yes' if summary.monotonic_channels[idx] else 'no'}"
            for idx in range(summary.channel_count)
        )
        return min_text, max_text, clipped, mono_text

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
