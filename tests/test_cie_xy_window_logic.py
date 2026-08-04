"""Targeted logic tests for CIE xy window mode/status behavior."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF

from prism.core.scopes.cie_xy import CieWhitePoint, CieXyTrace, CieXyViewData
from prism.ui.scopes import cie_xy_window as cie_xy_window_module
from prism.ui.scopes.cie_xy_window import CieXyWindow


class _PlotStub:
    def __init__(self) -> None:
        self.trace = "unset"
        self.visible = True
        self.highlighted = False
        self.overlay_names: tuple[str | None, ...] = ()
        self.whitepoint = "unset"
        self.trace_color_mode = "unset"

    def set_trace(self, trace) -> None:
        self.trace = trace

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def set_drop_target_highlight(self, highlighted: bool) -> None:
        self.highlighted = highlighted

    def set_gamut_overlay_names(self, names: tuple[str | None, ...]) -> None:
        self.overlay_names = names

    def set_whitepoint(self, whitepoint: CieWhitePoint | None) -> None:
        self.whitepoint = whitepoint

    def set_trace_color_mode(self, mode: str) -> None:
        self.trace_color_mode = mode


class _LabelStub:
    def __init__(self) -> None:
        self.text = ""
        self.style = ""

    def setText(self, text: str) -> None:
        self.text = text

    def setStyleSheet(self, style: str) -> None:
        self.style = style


class _ComboStub:
    def __init__(self, data: str | None) -> None:
        self._data = data

    def currentData(self) -> str | None:
        return self._data


class _WhitePointComboStub:
    def __init__(self, data: str | None) -> None:
        self._data = data

    def currentData(self) -> str | None:
        return self._data


class _TraceColorComboStub:
    def __init__(self, data: str) -> None:
        self._data = data

    def currentData(self) -> str:
        return self._data


def _trace() -> CieXyTrace:
    density = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    return CieXyTrace(
        x_values=np.asarray([0.3127], dtype=np.float32),
        y_values=np.asarray([0.3290], dtype=np.float32),
        density=density,
        color_density=np.zeros((*density.shape, 3), dtype=np.float32),
        source_size=(2, 2),
        rgb_space_name="sRGB",
    )


def _make_window(mode: str = "A") -> CieXyWindow:
    window = CieXyWindow.__new__(CieXyWindow)
    window._mode = mode
    window._buffer_a = None
    window._buffer_b = None
    window._overlay_names = ("sRGB", "ACEScg", "ARRI Wide Gamut 4")
    window._whitepoint_name = "D65"
    window._trace_color_mode = "boosted"
    window._plot_a = _PlotStub()
    window._plot_b = _PlotStub()
    window._status_label = _LabelStub()
    window._on_drop_file = None
    window._unsupported_main_mode = None
    return window


def test_default_gamut_overlays_cover_display_aces_and_camera_spaces() -> None:
    window = _make_window("A")

    assert window.current_gamut_overlay_names() == (
        "sRGB",
        "ACEScg",
        "ARRI Wide Gamut 4",
    )


def test_refresh_mode_a_shows_a_only(monkeypatch) -> None:
    window = _make_window("A")
    trace = _trace()
    monkeypatch.setattr(
        cie_xy_window_module,
        "build_cie_xy_view_data",
        lambda *_args, **_kwargs: CieXyViewData(
            mode="A",
            trace_a=trace,
            trace_b=None,
            status="CIE xy: A",
        ),
    )

    window._refresh_view_data()

    assert window._plot_a.visible is True
    assert window._plot_b.visible is False
    assert window._plot_a.trace is trace
    assert window._status_label.text == "CIE xy: A"


def test_refresh_mode_b_shows_b_only(monkeypatch) -> None:
    window = _make_window("B")
    trace = _trace()
    monkeypatch.setattr(
        cie_xy_window_module,
        "build_cie_xy_view_data",
        lambda *_args, **_kwargs: CieXyViewData(
            mode="B",
            trace_a=None,
            trace_b=trace,
            status="CIE xy: B",
        ),
    )

    window._refresh_view_data()

    assert window._plot_a.visible is False
    assert window._plot_b.visible is True
    assert window._plot_b.trace is trace
    assert window._status_label.text == "CIE xy: B"


def test_refresh_mode_ab_shows_both(monkeypatch) -> None:
    window = _make_window("A|B")
    trace_a = _trace()
    trace_b = _trace()
    monkeypatch.setattr(
        cie_xy_window_module,
        "build_cie_xy_view_data",
        lambda *_args, **_kwargs: CieXyViewData(
            mode="A|B",
            trace_a=trace_a,
            trace_b=trace_b,
            status="CIE xy: A|B",
        ),
    )

    window._refresh_view_data()

    assert window._plot_a.visible is True
    assert window._plot_b.visible is True
    assert window._plot_a.trace is trace_a
    assert window._plot_b.trace is trace_b
    assert window._status_label.text == "CIE xy: A|B"


def test_refresh_view_data_uses_default_viewer_rgb_interpretation(monkeypatch) -> None:
    window = _make_window("A")
    trace = _trace()
    captured: dict[str, str] = {}

    def _build(*_args, **kwargs):
        captured["rgb_space_name"] = kwargs["rgb_space_name"]
        return CieXyViewData(
            mode="A",
            trace_a=trace,
            trace_b=None,
            status="CIE xy: A",
        )

    monkeypatch.setattr(cie_xy_window_module, "build_cie_xy_view_data", _build)

    window._refresh_view_data()

    assert captured == {"rgb_space_name": "sRGB"}
    assert window._plot_a.trace is trace
    assert window._status_label.text == "CIE xy: A"


def test_gamut_overlay_change_updates_both_plots() -> None:
    window = _make_window("A")
    window._gamut_combos = [
        _ComboStub("sRGB"),
        _ComboStub(None),
        _ComboStub("ITU-R BT.2020"),
    ]

    window._on_gamut_overlay_changed()

    assert window.current_gamut_overlay_names() == ("sRGB", None, "ITU-R BT.2020")
    assert window._plot_a.overlay_names == ("sRGB", None, "ITU-R BT.2020")
    assert window._plot_b.overlay_names == ("sRGB", None, "ITU-R BT.2020")


def test_whitepoint_change_updates_both_plots() -> None:
    window = _make_window("A")
    window._whitepoint_combo = _WhitePointComboStub("D60")

    window._on_whitepoint_changed()

    assert window._whitepoint_name == "D60"
    assert window._plot_a.whitepoint.name == "D60"
    assert window._plot_b.whitepoint.name == "D60"


def test_whitepoint_none_hides_marker_on_both_plots() -> None:
    window = _make_window("A")
    window._whitepoint_combo = _WhitePointComboStub(None)

    window._on_whitepoint_changed()

    assert window._whitepoint_name is None
    assert window._plot_a.whitepoint is None
    assert window._plot_b.whitepoint is None


def test_trace_color_change_updates_both_plots() -> None:
    window = _make_window("A")
    window._trace_color_combo = _TraceColorComboStub("normal")

    window._on_trace_color_changed()

    assert window._trace_color_mode == "normal"
    assert window._plot_a.trace_color_mode == "normal"
    assert window._plot_b.trace_color_mode == "normal"


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
