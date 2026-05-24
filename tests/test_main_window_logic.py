"""Targeted logic tests for MainWindow non-visual behavior."""

from __future__ import annotations

from prism.core.ui_tokens import CompareModeData
from prism.core.viewer_state import CompareState, CompareViewState
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
    window.compare_view = _CompareViewStub()
    window._update_active_side_ui = lambda: None
    return window


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
