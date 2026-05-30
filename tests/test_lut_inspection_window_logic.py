"""Targeted logic tests for LUT inspection window load/status behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.io.lut_loader import LutLoadError, LutPlotData
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


def test_load_lut_success_updates_plot_info_and_status(monkeypatch) -> None:
    window = _make_window()
    path = Path("example.csp")
    data = _sample_plot_data(path)
    monkeypatch.setattr(lut_window_module, "load_lut_plot_data", lambda _path: data)

    window._load_lut(path)

    assert window._plot_widget.data is data
    assert window._file_label.text == str(path)
    assert "Format: CSP" in window._last_info_text
    assert window._status_label.text.startswith("Loaded CSP")
    assert window._status_label.style == "color: #8fdf8f;"


def test_load_lut_error_clears_plot_and_sets_error_status(monkeypatch) -> None:
    window = _make_window()
    path = Path("broken.cube")

    def _raise(_path: Path):
        raise LutLoadError("bad lut")

    monkeypatch.setattr(lut_window_module, "load_lut_plot_data", _raise)

    window._load_lut(path)

    assert window._plot_widget.data is None
    assert window._file_label.text == str(path)
    assert window._last_info_text == ""
    assert window._status_label.text == "Failed to load LUT: bad lut"
    assert window._status_label.style == "color: #ff8f8f;"


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

