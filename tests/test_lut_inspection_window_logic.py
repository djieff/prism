"""Targeted logic tests for LUT inspection window load/status behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.io.lut_loader import LutInspectionData, LutLoadError, LutPlotData, LutVolumeData
from prism.ui import lut_inspection_window as lut_window_module
from prism.ui.lut_inspection_window import LutInspectionWindow


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

    def set_volume_data(self, value) -> None:
        self.data = value

    def set_projection_mode(self, value: str) -> None:
        self.projection_modes.append(value)

    def set_use_output_positions(self, value: bool) -> None:
        self.position_modes.append(value)

    def set_show_neutral_axis(self, value: bool) -> None:
        self.neutral_axis_modes.append(value)


class _ComboStub:
    def __init__(self, data) -> None:
        self._data = data

    def currentData(self):
        return self._data


class _CheckStub:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


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
    window._volume_widget = _VolumeStub()
    window._volume_projection_combo = _ComboStub("RG plane")
    window._volume_position_combo = _ComboStub(False)
    window._volume_neutral_axis_checkbox = _CheckStub(True)
    window._file_label = _LabelStub()
    window._status_label = _LabelStub()
    window._last_info_text = None

    def _capture_info(text: str) -> None:
        window._last_info_text = text

    window._set_info_panel_text = _capture_info
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
    window = _make_window()
    path = Path("example.csp")
    plot = _sample_plot_data(path)
    volume = _sample_volume_data(path)
    data = LutInspectionData(plot=plot, volume=volume)
    monkeypatch.setattr(lut_window_module, "load_lut_inspection_data", lambda _path: data)

    window._load_lut(path)

    assert window._plot_widget.data is plot
    assert window._volume_widget.data is volume
    assert window._file_label.text == str(path)
    assert "Format: CSP" in window._last_info_text
    assert "3D Size: 2 x 2 x 2" in window._last_info_text
    assert window._status_label.text.startswith("Loaded CSP")
    assert "volume: 2x2x2" in window._status_label.text
    assert window._status_label.style == "color: #8fdf8f;"


def test_load_lut_success_with_1d_data_marks_volume_unavailable(monkeypatch) -> None:
    window = _make_window()
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
    assert window._volume_widget.data is None
    assert "Volume: unavailable" in window._last_info_text
    assert "volume: unavailable" in window._status_label.text


def test_load_lut_error_clears_plot_and_sets_error_status(monkeypatch) -> None:
    window = _make_window()
    path = Path("broken.cube")

    def _raise(_path: Path):
        raise LutLoadError("bad lut")

    monkeypatch.setattr(lut_window_module, "load_lut_inspection_data", _raise)

    window._load_lut(path)

    assert window._plot_widget.data is None
    assert window._volume_widget.data is None
    assert window._file_label.text == str(path)
    assert window._last_info_text == ""
    assert window._status_label.text == "Failed to load LUT: bad lut"
    assert window._status_label.style == "color: #ff8f8f;"


def test_volume_projection_control_updates_volume_widget() -> None:
    window = _make_window()

    window._on_volume_projection_changed()

    assert window._volume_widget.projection_modes == ["RG plane"]


def test_volume_position_control_updates_volume_widget() -> None:
    window = _make_window()

    window._on_volume_position_changed()

    assert window._volume_widget.position_modes == [False]


def test_volume_neutral_axis_control_updates_volume_widget() -> None:
    window = _make_window()

    window._on_volume_neutral_axis_changed()

    assert window._volume_widget.neutral_axis_modes == [True]


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
