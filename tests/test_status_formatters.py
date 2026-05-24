"""Regression tests for UI status/label formatting helpers."""

from __future__ import annotations

from pathlib import Path

from prism.core.source_models import ImageSequenceSource
from prism.core.viewer_state import PanelState
from prism.ui.status_formatters import (
    STATUS_LOADED,
    STATUS_NONE,
    STATUS_WAITING,
    panel_frame_suffix,
    persistent_config_label,
    persistent_status_label_for_panel,
    persistent_status_message,
    side_label,
)


def test_side_label_maps_viewer_sides() -> None:
    assert side_label("left") == "A"
    assert side_label("right") == "B"


def test_panel_frame_suffix_single_frame_is_empty() -> None:
    panel = PanelState(frame_count=1, current_frame_index=0)
    assert panel_frame_suffix(panel) == ""


def test_panel_frame_suffix_multi_frame_formats_display() -> None:
    panel = PanelState(frame_count=12, current_frame_index=4)
    assert panel_frame_suffix(panel) == " [frame 5/12]"


def test_persistent_config_label_uses_basename() -> None:
    assert persistent_config_label(None) == STATUS_NONE
    assert persistent_config_label("C:/show/configs/studio.ocio") == "studio.ocio"


def test_persistent_status_label_waiting_without_image() -> None:
    panel = PanelState()
    assert persistent_status_label_for_panel(panel) == STATUS_WAITING


def test_persistent_status_label_uses_sequence_frame_label() -> None:
    sequence = ImageSequenceSource(
        [
            Path("shot.beauty.1001.exr"),
            Path("shot.beauty.1002.exr"),
        ]
    )
    panel = PanelState(
        loaded_image_data=object(),  # marker only; formatter checks for non-None
        loaded_source=sequence,
        current_frame_index=1,
    )
    assert persistent_status_label_for_panel(panel) == "shot.beauty.1002.exr"


def test_persistent_status_label_uses_image_basename() -> None:
    panel = PanelState(
        loaded_image_data=object(),
        image_path="D:/renders/final_comp_v12.exr",
    )
    assert persistent_status_label_for_panel(panel) == "final_comp_v12.exr"


def test_persistent_status_label_falls_back_to_loaded() -> None:
    panel = PanelState(loaded_image_data=object())
    assert persistent_status_label_for_panel(panel) == STATUS_LOADED


def test_persistent_status_message_formats_expected_layout() -> None:
    left_panel = PanelState(loaded_image_data=object(), image_path="C:/a/left.exr")
    right_panel = PanelState()
    message = persistent_status_message(left_panel, right_panel, "C:/cfg/show.ocio")
    assert message == "A: left.exr | B: Waiting | Config: show.ocio"
