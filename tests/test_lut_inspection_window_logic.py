"""Targeted logic tests for LUT inspection window load/status behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.io.lut.loader import LutInspectionData, LutLoadError, LutPlotData, LutVolumeData
from prism.ui.lut_inspector import window as lut_window_module
from prism.ui.lut_inspector.window import LutInspectionWindow


class _LabelStub:
    def __init__(self) -> None:
        self.text = ""
        self.style = ""

    def setText(self, value: str) -> None:
        self.text = value

    def setStyleSheet(self, value: str) -> None:
        self.style = value


class _PlotStub:
    def __init__(self) -> None:
        self.data = "unset"

    def set_plot_data(self, value) -> None:
        self.data = value


class _VolumeStub:
    def __init__(self) -> None:
        self.data = "unset"
        self.projection_modes: list[str] = []
        self.position_modes: list[bool] = []
        self.neutral_axis_modes: list[bool] = []
        self.rgb_axes_modes: list[bool] = []
        self.density_modes: list[str] = []
        self.reset_calls = 0

    def set_volume_data(self, value) -> None:
        self.data = value

    def set_projection_mode(self, value: str) -> None:
        self.projection_modes.append(value)

    def set_use_output_positions(self, value: bool) -> None:
        self.position_modes.append(value)

    def set_show_neutral_axis(self, value: bool) -> None:
        self.neutral_axis_modes.append(value)

    def set_show_rgb_axes(self, value: bool) -> None:
        self.rgb_axes_modes.append(value)

    def set_density(self, value: str) -> None:
        self.density_modes.append(value)

    def reset_view(self) -> None:
        self.reset_calls += 1


class _ComboStub:
    def __init__(self, data) -> None:
        self._data = data
        self.enabled = True

    def currentData(self):
        return self._data

    def setEnabled(self, value: bool) -> None:
        self.enabled = value

    def findData(self, data) -> int:
        if data == "qpainter":
            return 0
        if data == "opengl":
            return 1
        return -1

    def setCurrentIndex(self, index: int) -> None:
        if index == 0:
            self._data = "qpainter"
        elif index == 1:
            self._data = "opengl"


class _ButtonStub:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class _StackStub:
    def __init__(self) -> None:
        self.current_widget = None

    def setCurrentWidget(self, widget) -> None:
        self.current_widget = widget


class _CheckStub:
    def __init__(self, checked: bool) -> None:
        self._checked = checked
        self.enabled = True

    def isChecked(self) -> bool:
        return self._checked

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class _UrlStub:
    def __init__(self, local_file: str) -> None:
        self._local_file = local_file

    def isLocalFile(self) -> bool:
        return True

    def toLocalFile(self) -> str:
        return self._local_file


class _MimeStub:
    def __init__(self, urls: list[_UrlStub]) -> None:
        self._urls = urls

    def urls(self) -> list[_UrlStub]:
        return self._urls


class _DropEventStub:
    def __init__(self, urls: list[_UrlStub]) -> None:
        self._mime = _MimeStub(urls)
        self.accepted = False
        self.ignored = False

    def mimeData(self) -> _MimeStub:
        return self._mime

    def acceptProposedAction(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


def _make_window() -> LutInspectionWindow:
    window = LutInspectionWindow.__new__(LutInspectionWindow)
    window._plot_widget = _PlotStub()
    window._volume_painter_widget = _VolumeStub()
    window._volume_gl_widget = _VolumeStub()
    window._volume_widget = window._volume_painter_widget
    window._volume_stack = _StackStub()
    window._volume_renderer_combo = _ComboStub("opengl")
    window._volume_projection_combo = _ComboStub("RG plane")
    window._volume_position_combo = _ComboStub(False)
    window._volume_density_combo = _ComboStub("High")
    window._volume_neutral_axis_checkbox = _CheckStub(True)
    window._volume_rgb_axes_checkbox = _CheckStub(True)
    window._volume_reset_button = _ButtonStub()
    window._file_label = _LabelStub()
    window._status_label = _LabelStub()
    window._last_info_text = None
    window._current_volume_data = None

    def _capture_info(text: str) -> None:
        window._last_info_text = text

    window._set_info_panel_text = _capture_info
    return window


def _make_window_without_gl() -> LutInspectionWindow:
    window = _make_window()
    window._volume_gl_widget = None
    window._volume_renderer_combo = _ComboStub("qpainter")
    return window


def _sample_plot_data(path: Path) -> LutPlotData:
    x = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    y = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.5, 0.6], [1.0, 1.0, 1.0]], dtype=np.float32)
    return LutPlotData(
        path=path,
        format="csp",
        source_kind="3d_neutral_axis",
        channels=3,
        x_values=x,
        y_values=y,
        domain_min=0.0,
        domain_max=1.0,
    )


def _sample_volume_data(path: Path) -> LutVolumeData:
    values = np.zeros((2, 2, 2, 3), dtype=np.float32)
    return LutVolumeData(
        path=path,
        format="csp",
        size_x=2,
        size_y=2,
        size_z=2,
        values=values,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def test_load_lut_success_updates_plot_info_and_status(monkeypatch) -> None:
    window = _make_window_without_gl()
    path = Path("example.csp")
    plot = _sample_plot_data(path)
    volume = _sample_volume_data(path)
    data = LutInspectionData(plot=plot, volume=volume)
    monkeypatch.setattr(lut_window_module, "load_lut_inspection_data", lambda _path: data)

    window._load_lut(path)

    assert window._plot_widget.data is plot
    assert window._volume_painter_widget.data is volume
    assert window._volume_gl_widget is None
    assert window._current_volume_data is volume
    assert window._file_label.text == str(path)
    assert "Format: CSP" in window._last_info_text
    assert "3D Size: 2 x 2 x 2" in window._last_info_text
    assert window._status_label.text.startswith("Loaded CSP")
    assert "volume: 2x2x2" in window._status_label.text
    assert window._status_label.style == "color: #8fdf8f;"


def test_load_lut_success_with_1d_data_marks_volume_unavailable(monkeypatch) -> None:
    window = _make_window_without_gl()
    path = Path("example.spi1d")
    plot = LutPlotData(
        path=path,
        format="spi1d",
        source_kind="1d",
        channels=1,
        x_values=np.asarray([0.0, 1.0], dtype=np.float32),
        y_values=np.asarray([[0.0], [1.0]], dtype=np.float32),
        domain_min=0.0,
        domain_max=1.0,
    )
    data = LutInspectionData(plot=plot, volume=None)
    monkeypatch.setattr(lut_window_module, "load_lut_inspection_data", lambda _path: data)

    window._load_lut(path)

    assert window._plot_widget.data is plot
    assert window._volume_painter_widget.data is None
    assert window._volume_gl_widget is None
    assert window._current_volume_data is None
    assert "Volume: unavailable" in window._last_info_text
    assert "volume: unavailable" in window._status_label.text


def test_load_lut_error_clears_plot_and_sets_error_status(monkeypatch) -> None:
    window = _make_window_without_gl()
    path = Path("broken.cube")

    def _raise(_path: Path):
        raise LutLoadError("bad lut")

    monkeypatch.setattr(lut_window_module, "load_lut_inspection_data", _raise)

    window._load_lut(path)

    assert window._plot_widget.data is None
    assert window._volume_painter_widget.data is None
    assert window._volume_gl_widget is None
    assert window._current_volume_data is None
    assert window._file_label.text == str(path)
    assert window._last_info_text == ""
    assert window._status_label.text == "Failed to load LUT: bad lut"
    assert window._status_label.style == "color: #ff8f8f;"


def test_volume_projection_control_updates_volume_widget() -> None:
    window = _make_window()

    window._on_volume_projection_changed()

    assert window._volume_painter_widget.projection_modes == ["RG plane"]
    assert window._volume_gl_widget.projection_modes == ["RG plane"]


def test_volume_position_control_updates_volume_widget() -> None:
    window = _make_window()

    window._on_volume_position_changed()

    assert window._volume_painter_widget.position_modes == [False]
    assert window._volume_gl_widget.position_modes == [False]


def test_volume_neutral_axis_control_updates_volume_widget() -> None:
    window = _make_window()

    window._on_volume_neutral_axis_changed()

    assert window._volume_painter_widget.neutral_axis_modes == [True]
    assert window._volume_gl_widget.neutral_axis_modes == [True]


def test_volume_rgb_axes_control_updates_gl_widget_only() -> None:
    window = _make_window()

    window._on_volume_rgb_axes_changed()

    assert window._volume_gl_widget.rgb_axes_modes == [True]
    assert window._volume_painter_widget.rgb_axes_modes == []


def test_volume_renderer_control_switches_between_gl_and_painter() -> None:
    window = _make_window()

    window._on_volume_renderer_changed()

    assert window._volume_stack.current_widget is window._volume_gl_widget
    assert window._volume_density_combo.enabled is True
    assert window._volume_reset_button.enabled is True
    assert window._volume_rgb_axes_checkbox.enabled is True
    assert window._volume_neutral_axis_checkbox.enabled is False

    window._volume_renderer_combo = _ComboStub("qpainter")
    window._on_volume_renderer_changed()

    assert window._volume_stack.current_widget is window._volume_painter_widget
    assert window._volume_density_combo.enabled is False
    assert window._volume_reset_button.enabled is False
    assert window._volume_rgb_axes_checkbox.enabled is False
    assert window._volume_neutral_axis_checkbox.enabled is True


def test_volume_renderer_control_lazy_creates_gl_widget() -> None:
    window = _make_window_without_gl()
    gl_widget = _VolumeStub()
    volume = _sample_volume_data(Path("example.cube"))
    window._current_volume_data = volume
    window._volume_renderer_combo = _ComboStub("opengl")

    def _ensure_gl_widget():
        window._volume_gl_widget = gl_widget
        return gl_widget

    window._ensure_volume_gl_widget = _ensure_gl_widget

    window._on_volume_renderer_changed()

    assert window._volume_gl_widget is gl_widget
    assert window._volume_stack.current_widget is gl_widget
    assert gl_widget.data is volume
    assert window._volume_density_combo.enabled is True
    assert window._volume_reset_button.enabled is True
    assert window._volume_rgb_axes_checkbox.enabled is True
    assert window._volume_neutral_axis_checkbox.enabled is False


def test_volume_renderer_control_defaults_to_qpainter_without_gl() -> None:
    window = _make_window_without_gl()

    window._on_volume_renderer_changed()

    assert window._volume_gl_widget is None
    assert window._volume_stack.current_widget is window._volume_painter_widget
    assert window._volume_density_combo.enabled is False
    assert window._volume_reset_button.enabled is False
    assert window._volume_rgb_axes_checkbox.enabled is False
    assert window._volume_neutral_axis_checkbox.enabled is True


def test_set_volume_data_updates_lazy_gl_after_creation() -> None:
    window = _make_window_without_gl()
    volume = _sample_volume_data(Path("example.cube"))
    gl_widget = _VolumeStub()

    window._set_volume_data(volume)
    assert window._volume_painter_widget.data is volume
    assert window._volume_gl_widget is None

    window._volume_gl_widget = gl_widget
    window._set_volume_data(volume)

    assert window._volume_painter_widget.data is volume
    assert window._volume_gl_widget.data is volume


def test_volume_density_control_updates_gl_widget_only() -> None:
    window = _make_window()

    window._on_volume_density_changed()

    assert window._volume_gl_widget.density_modes == ["High"]
    assert window._volume_painter_widget.density_modes == []


def test_volume_reset_view_updates_gl_widget_only() -> None:
    window = _make_window()

    window._on_volume_reset_view()

    assert window._volume_gl_widget.reset_calls == 1
    assert window._volume_painter_widget.reset_calls == 0


def test_volume_gl_initialization_failure_switches_to_qpainter() -> None:
    window = _make_window()

    window._on_volume_gl_initialization_failed("no context")

    assert window._volume_renderer_combo.currentData() == "qpainter"
    assert "OpenGL unavailable; using QPainter renderer: no context" == window._status_label.text
    assert window._status_label.style == "color: #ffd27f;"


def test_load_entrypoints_delegate_to_common_loader_path(monkeypatch) -> None:
    window = _make_window()
    calls: list[Path] = []
    monkeypatch.setattr(window, "_load_lut", lambda path: calls.append(path))

    direct_path = Path("via_method.cube")
    window.load_lut_path(direct_path)

    drop_path = Path("via_drop.spi1d")
    event = _DropEventStub([_UrlStub(str(drop_path))])
    window.dropEvent(event)

    assert calls == [direct_path, drop_path]
    assert event.accepted is True
    assert event.ignored is False
