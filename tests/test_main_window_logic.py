"""Targeted logic tests for MainWindow non-visual behavior."""

from __future__ import annotations

import numpy as np

from prism.core.ui_tokens import CompareModeData
from prism.core.viewer_state import CompareState, CompareViewState
from prism.ui import main_window as main_window_module
from prism.ui.main_window import MainWindow


class _CompareViewStub:
    def __init__(self) -> None:
        self.active_side_calls: list[str] = []

    def set_active_side(self, side: str) -> None:
        self.active_side_calls.append(side)


def _make_window_for_logic() -> MainWindow:
    window = MainWindow.__new__(MainWindow)
    window._compare_state = CompareState()
    window._compare_view_state = CompareViewState()
    window._processed_display_buffers = {"left": None, "right": None}
    window._waveform_window = None
    window.compare_view = _CompareViewStub()
    window._update_active_side_ui = lambda: None
    return window


class _WaveformWindowStub:
    def __init__(self, _parent, on_drop_file=None, on_source_mode_changed=None) -> None:
        self.buffers: list[tuple[object, object]] = []
        self.attrs: list[tuple[object, bool]] = []
        self.show_calls = 0
        self.raise_calls = 0
        self.activate_calls = 0
        self.on_drop_file = on_drop_file
        self.on_source_mode_changed = on_source_mode_changed
        self.source_mode = None
        self.unsupported_main_mode = None
        self.destroyed = _SignalStub()

    def setAttribute(self, attr, enabled: bool) -> None:
        self.attrs.append((attr, enabled))

    def show(self) -> None:
        self.show_calls += 1

    def raise_(self) -> None:
        self.raise_calls += 1

    def activateWindow(self) -> None:
        self.activate_calls += 1

    def set_processed_buffers(self, a, b) -> None:
        self.buffers.append((a, b))

    def set_source_mode(self, mode) -> None:
        self.source_mode = mode

    def set_unsupported_main_mode(self, mode) -> None:
        self.unsupported_main_mode = mode


class _SignalStub:
    def connect(self, _fn) -> None:
        return


def test_compare_mode_data_from_state_maps_full_to_active_side_payload() -> None:
    window = _make_window_for_logic()

    window._compare_state.active_side = "left"
    assert window._compare_mode_data_from_state("full") == CompareModeData.FULL_LEFT.value

    window._compare_state.active_side = "right"
    assert window._compare_mode_data_from_state("full") == CompareModeData.FULL_RIGHT.value

    assert window._compare_mode_data_from_state("split") == "split"
    assert window._compare_mode_data_from_state("wipe") == "wipe"
    assert window._compare_mode_data_from_state("diff") == "diff"


def test_apply_compare_mode_data_rejects_unknown_value() -> None:
    window = _make_window_for_logic()
    assert window._apply_compare_mode_data("unknown_mode") is False


def test_apply_compare_mode_data_sets_full_left_state_and_updates_view() -> None:
    window = _make_window_for_logic()

    changed = window._apply_compare_mode_data(CompareModeData.FULL_LEFT.value)

    assert changed is True
    assert window._compare_view_state.mode == "full"
    assert window._compare_state.active_side == "left"
    assert window.compare_view.active_side_calls[-1] == "left"


def test_apply_compare_mode_data_sets_full_right_state_and_updates_view() -> None:
    window = _make_window_for_logic()

    changed = window._apply_compare_mode_data(CompareModeData.FULL_RIGHT.value)

    assert changed is True
    assert window._compare_view_state.mode == "full"
    assert window._compare_state.active_side == "right"
    assert window.compare_view.active_side_calls[-1] == "right"


def test_apply_compare_mode_data_sets_non_full_mode_without_side_swap() -> None:
    window = _make_window_for_logic()
    window._compare_state.active_side = "right"

    changed = window._apply_compare_mode_data(CompareModeData.WIPE.value)

    assert changed is True
    assert window._compare_view_state.mode == "wipe"
    assert window._compare_state.active_side == "right"


def test_choose_drop_target_side_prefers_active_when_empty() -> None:
    window = _make_window_for_logic()
    window._compare_state.active_side = "left"
    window._compare_state.left.loaded_image_data = None
    window._compare_state.right.loaded_image_data = object()

    assert window._choose_drop_target_side() == "left"


def test_choose_drop_target_side_uses_other_side_when_active_loaded() -> None:
    window = _make_window_for_logic()
    window._compare_state.active_side = "left"
    window._compare_state.left.loaded_image_data = object()
    window._compare_state.right.loaded_image_data = None

    assert window._choose_drop_target_side() == "right"


def test_choose_drop_target_side_falls_back_to_active_when_both_loaded() -> None:
    window = _make_window_for_logic()
    window._compare_state.active_side = "right"
    window._compare_state.left.loaded_image_data = object()
    window._compare_state.right.loaded_image_data = object()

    assert window._choose_drop_target_side() == "right"


def test_resolve_drop_target_side_none_delegates_to_choose() -> None:
    window = _make_window_for_logic()
    window._choose_drop_target_side = lambda: "right"

    assert window._resolve_drop_target_side(None) == "right"


def test_resolve_drop_target_side_prefers_requested_when_empty() -> None:
    window = _make_window_for_logic()
    window._compare_state.left.loaded_image_data = None
    window._compare_state.right.loaded_image_data = object()

    assert window._resolve_drop_target_side("left") == "left"


def test_resolve_drop_target_side_uses_other_when_requested_loaded() -> None:
    window = _make_window_for_logic()
    window._compare_state.left.loaded_image_data = object()
    window._compare_state.right.loaded_image_data = None

    assert window._resolve_drop_target_side("left") == "right"


def test_resolve_drop_target_side_returns_requested_when_both_loaded() -> None:
    window = _make_window_for_logic()
    window._compare_state.left.loaded_image_data = object()
    window._compare_state.right.loaded_image_data = object()

    assert window._resolve_drop_target_side("left") == "left"


def test_mode_requirement_hint_only_for_modes_requiring_both_images() -> None:
    window = _make_window_for_logic()
    window._compare_state.left.loaded_image_data = object()
    window._compare_state.right.loaded_image_data = None

    window._compare_view_state.mode = "split"
    assert window._mode_requirement_hint() == "Split compare needs A and B loaded"

    window._compare_view_state.mode = "wipe"
    assert window._mode_requirement_hint() == "Wipe compare needs A and B loaded"

    window._compare_view_state.mode = "diff"
    assert window._mode_requirement_hint() == "Diff needs A and B loaded"

    window._compare_view_state.mode = "full"
    assert window._mode_requirement_hint() is None


def test_mode_requirement_hint_none_when_both_loaded() -> None:
    window = _make_window_for_logic()
    window._compare_state.left.loaded_image_data = object()
    window._compare_state.right.loaded_image_data = object()
    window._compare_view_state.mode = "split"

    assert window._mode_requirement_hint() is None


def test_update_waveform_window_if_open_pushes_buffers() -> None:
    window = _make_window_for_logic()
    waveform = _WaveformWindowStub(window)
    buffer_a = np.zeros((2, 2, 3), dtype=np.float32)
    buffer_b = np.ones((2, 2, 3), dtype=np.float32)
    window._waveform_window = waveform
    window._processed_display_buffers = {"left": buffer_a, "right": buffer_b}

    window._update_waveform_window_if_open()

    assert waveform.buffers == [(buffer_a, buffer_b)]


def test_show_waveform_window_creates_then_reuses_window(monkeypatch) -> None:
    window = _make_window_for_logic()
    monkeypatch.setattr(main_window_module, "WaveformWindow", _WaveformWindowStub)

    window._show_waveform_window()
    first = window._waveform_window
    assert first is not None
    assert first.show_calls == 1
    assert first.raise_calls == 1
    assert first.activate_calls == 1

    window._show_waveform_window()
    assert window._waveform_window is first
    assert first.show_calls == 2
    assert first.raise_calls == 2
    assert first.activate_calls == 2


def test_handle_waveform_drop_file_delegates_to_main_drop_handler(monkeypatch) -> None:
    window = _make_window_for_logic()
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_handle_dropped_file",
        lambda file_path, requested_side=None: calls.append((file_path, requested_side)),
    )

    window._handle_waveform_drop_file("C:/tmp/test.exr", "left")

    assert calls == [("C:/tmp/test.exr", "left")]


def test_main_view_mode_to_waveform_mode_mapping() -> None:
    window = _make_window_for_logic()

    window._compare_view_state.mode = "split"
    assert window._main_view_mode_to_waveform_mode() == "A|B"

    window._compare_view_state.mode = "full"
    window._compare_state.active_side = "left"
    assert window._main_view_mode_to_waveform_mode() == "A"

    window._compare_state.active_side = "right"
    assert window._main_view_mode_to_waveform_mode() == "B"

    window._compare_view_state.mode = "wipe"
    assert window._main_view_mode_to_waveform_mode() is None


def test_sync_waveform_mode_from_main_if_open_updates_window_mode() -> None:
    window = _make_window_for_logic()
    waveform = _WaveformWindowStub(window)
    window._waveform_window = waveform
    window._compare_view_state.mode = "full"
    window._compare_state.active_side = "right"

    window._sync_waveform_mode_from_main_if_open()

    assert waveform.unsupported_main_mode is None
    assert waveform.source_mode == "B"


def test_sync_waveform_mode_marks_unsupported_for_wipe_and_diff() -> None:
    window = _make_window_for_logic()
    waveform = _WaveformWindowStub(window)
    window._waveform_window = waveform

    window._compare_view_state.mode = "wipe"
    window._sync_waveform_mode_from_main_if_open()
    assert waveform.unsupported_main_mode == "Wipe"

    window._compare_view_state.mode = "diff"
    window._sync_waveform_mode_from_main_if_open()
    assert waveform.unsupported_main_mode == "Diff"
