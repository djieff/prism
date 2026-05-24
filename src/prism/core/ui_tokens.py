"""Shared UI/control-flow token definitions.

These tokens centralize logic-critical string values used across UI/state
orchestration. Values remain string-compatible with existing state and widget
payloads.
"""

from __future__ import annotations

from enum import Enum

COMPARE_MODES_WITH_SIDE_CONTEXT = frozenset({"split", "wipe", "full"})
COMPARE_MODES_REQUIRING_BOTH_IMAGES = frozenset({"split", "wipe", "diff"})


class CompareModeData(str, Enum):
    """UI compare-mode payload values used by the mode combobox/hotkeys."""

    SPLIT = "split"
    WIPE = "wipe"
    FULL_LEFT = "full_left"
    FULL_RIGHT = "full_right"
    DIFF = "diff"


class ViewerChannel(str, Enum):
    """Viewer display channel modes for image preview."""

    RGB = "rgb"
    RED = "r"
    GREEN = "g"
    BLUE = "b"


class AlignmentAnchor(str, Enum):
    """Compare-space alignment anchors shared by MainWindow and CompareView."""

    VIEWPORT = "viewport_space"
    A_SPACE = "a_space"
    B_SPACE = "b_space"
