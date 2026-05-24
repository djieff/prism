"""Pure formatting helpers for main-window status and labels."""

from __future__ import annotations

from pathlib import Path

from prism.core.viewer_state import PanelState, ViewerSide

STATUS_WAITING = "Waiting"
STATUS_LOADED = "Loaded"
STATUS_NONE = "None"


def side_label(side: ViewerSide) -> str:
    """Return user-facing side label for a viewer side key."""
    return "A" if side == "left" else "B"


def panel_frame_suffix(panel: PanelState) -> str:
    """Return formatted frame-count suffix for a panel status message."""
    if panel.frame_count <= 1:
        return ""
    return f" [frame {panel.current_frame_index + 1}/{panel.frame_count}]"


def persistent_config_label(config_path: str | None) -> str:
    """Return persistent status text label for active OCIO config path."""
    if not config_path:
        return STATUS_NONE
    return Path(config_path).name


def persistent_status_label_for_panel(panel: PanelState) -> str:
    """Return persistent status text label for a panel image/source state."""
    if panel.loaded_image_data is None:
        return STATUS_WAITING

    source = panel.loaded_source
    if source is not None and getattr(source, "kind", None) == "sequence":
        frame_info = source.get_frame_info(panel.current_frame_index)
        if frame_info.label:
            return frame_info.label

    if panel.image_path:
        return Path(panel.image_path).name
    return STATUS_LOADED


def persistent_status_message(
    left_panel: PanelState,
    right_panel: PanelState,
    config_path: str | None,
) -> str:
    """Build persistent A/B/config status text for the window status bar."""
    left_label = persistent_status_label_for_panel(left_panel)
    right_label = persistent_status_label_for_panel(right_panel)
    config_label = persistent_config_label(config_path)
    return f"A: {left_label} | B: {right_label} | Config: {config_label}"
