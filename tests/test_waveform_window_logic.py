"""Targeted logic tests for waveform window mode/status behavior."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF

from prism.core.scope_waveform import WaveformTrace, WaveformViewData
from prism.core.scope_waveform_science import DEFAULT_WAVEFORM_SIGNAL_STANDARD
from prism.ui import waveform_window as waveform_window_module
from prism.ui.waveform_window import WaveformWindow


class _PlotStub:
    def __init__(self) -> None:
        self.trace = "unset"
        self.visible = True
        self.highlighted = False

    def set_trace(self, trace) -> None:
        self.trace = trace

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def set_drop_target_highlight(self, highlighted: bool) -> None:
        self.highlighted = highlighted

    def set_signal_mode(self, mode) -> None:
        self.signal_mode = mode


class _LabelStub:
    def __init__(self) -> None:
        self.text = ""
        self.style = ""

    def setText(self, text: str) -> None:
        self.text = text

    def setStyleSheet(self, style: str) -> None:
        self.style = style


class _ComboStub:
    def __init__(self, data: str) -> None:
        self._data = data

    def currentData(self) -> str:
        return self._data


def _trace() -> WaveformTrace:
    x = np.asarray([0.0, 1.0], dtype=np.float32)
    density = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    return WaveformTrace(
        x_values=x,
        density_r=density,
        density_g=density,
        density_b=density,
        density_luma=density,
        source_size=(2, 2),
        signal_standard="ITU-R BT.709",
        y_prime_coefficients=(0.2126, 0.7152, 0.0722),
    )


def _make_window(mode: str = "A") -> WaveformWindow:
    window = WaveformWindow.__new__(WaveformWindow)
    window._mode = mode
    window._buffer_a = None
    window._buffer_b = None
    window._signal_standard = DEFAULT_WAVEFORM_SIGNAL_STANDARD
    window._plot_a = _PlotStub()
    window._plot_b = _PlotStub()
    window._status_label = _LabelStub()
    window._on_drop_file = None
    window._unsupported_main_mode = None
    return window


def test_refresh_mode_a_shows_a_only(monkeypatch) -> None:
    window = _make_window("A")
    trace = _trace()
    monkeypatch.setattr(
        waveform_window_module,
        "build_waveform_view_data",
        lambda *_args, **_kwargs: WaveformViewData(mode="A", trace_a=trace, trace_b=None, status="Waveform: A"),
    )

    window._refresh_view_data()

    assert window._plot_a.visible is True
    assert window._plot_b.visible is False
    assert window._plot_a.trace is trace
    assert window._status_label.text == "Waveform: A | Y' standard: BT.709"


def test_refresh_mode_b_shows_b_only(monkeypatch) -> None:
    window = _make_window("B")
    trace = _trace()
    monkeypatch.setattr(
        waveform_window_module,
        "build_waveform_view_data",
        lambda *_args, **_kwargs: WaveformViewData(mode="B", trace_a=None, trace_b=trace, status="Waveform: B"),
    )

    window._refresh_view_data()

    assert window._plot_a.visible is False
    assert window._plot_b.visible is True
    assert window._plot_b.trace is trace
    assert window._status_label.text == "Waveform: B | Y' standard: BT.709"


def test_refresh_mode_ab_shows_both(monkeypatch) -> None:
    window = _make_window("A|B")
    trace_a = _trace()
    trace_b = _trace()
    monkeypatch.setattr(
        waveform_window_module,
        "build_waveform_view_data",
        lambda *_args, **_kwargs: WaveformViewData(
            mode="A|B",
            trace_a=trace_a,
            trace_b=trace_b,
            status="Waveform: A|B",
        ),
    )

    window._refresh_view_data()

    assert window._plot_a.visible is True
    assert window._plot_b.visible is True
    assert window._plot_a.trace is trace_a
    assert window._plot_b.trace is trace_b
    assert window._status_label.text == "Waveform: A|B | Y' standard: BT.709"


def test_signal_standard_change_rebuilds_current_traces(monkeypatch) -> None:
    window = _make_window("A")
    window._signal_standard_combo = _ComboStub("ITU-R BT.2020")
    trace = _trace()
    captured: dict[str, str] = {}

    def _build(*_args, **kwargs):
        captured["signal_standard"] = kwargs["signal_standard"]
        return WaveformViewData(mode="A", trace_a=trace, trace_b=None, status="Waveform: A")

    monkeypatch.setattr(waveform_window_module, "build_waveform_view_data", _build)

    window._on_signal_standard_changed()

    assert window.current_signal_standard() == "ITU-R BT.2020"
    assert captured == {"signal_standard": "ITU-R BT.2020"}
    assert window._plot_a.trace is trace
    assert window._status_label.text == "Waveform: A | Y' standard: BT.2020"


def test_target_side_for_window_pos_mode_a_and_b() -> None:
    window = _make_window("A")
    assert window._target_side_for_window_pos(QPointF(10.0, 10.0)) == "left"
    window._mode = "B"
    assert window._target_side_for_window_pos(QPointF(10.0, 10.0)) == "right"


def test_target_side_for_window_pos_mode_ab_by_pane_geometry() -> None:
    window = _make_window("A|B")

    class _GeomStub:
        def __init__(self, x0: int, y0: int, x1: int, y1: int) -> None:
            self._x0 = x0
            self._y0 = y0
            self._x1 = x1
            self._y1 = y1

        def contains(self, p) -> bool:
            return self._x0 <= p.x() <= self._x1 and self._y0 <= p.y() <= self._y1

    window._plot_a.geometry = lambda: _GeomStub(0, 0, 99, 99)
    window._plot_b.geometry = lambda: _GeomStub(100, 0, 199, 99)
    assert window._target_side_for_window_pos(QPointF(50.0, 40.0)) == "left"
    assert window._target_side_for_window_pos(QPointF(150.0, 40.0)) == "right"
    assert window._target_side_for_window_pos(QPointF(250.0, 40.0)) is None


def test_set_drop_target_highlight_updates_plot_stubs() -> None:
    window = _make_window("A|B")
    window._set_drop_target_highlight("left")
    assert window._plot_a.highlighted is True
    assert window._plot_b.highlighted is False
    window._set_drop_target_highlight("right")
    assert window._plot_a.highlighted is False
    assert window._plot_b.highlighted is True
    window._set_drop_target_highlight(None)
    assert window._plot_a.highlighted is False
    assert window._plot_b.highlighted is False


def test_refresh_unsupported_mode_shows_blank_and_status() -> None:
    window = _make_window("A|B")
    window._unsupported_main_mode = "Wipe"

    window._refresh_view_data()

    assert window._plot_a.visible is True
    assert window._plot_b.visible is True
    assert window._plot_a.trace is None
    assert window._plot_b.trace is None
    assert window._status_label.text.startswith("Unsupported mode: Wipe")
