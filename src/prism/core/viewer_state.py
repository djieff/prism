"""Viewer state models for single and compare workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from prism.core.source_models import ViewerSourceState

ViewerSide = Literal["left", "right"]
CompareMode = Literal["wipe", "split", "full", "diff"]
FitMode = Literal["fit", "manual"]
GeometryCanvasPolicy = Literal["native", "match_a", "match_b", "viewport"]
GeometryScalePolicy = Literal["fit", "fill", "stretch", "one_to_one"]
AlignmentAnchor = Literal["a_space", "b_space", "viewport_space"]


@dataclass
class PanelState:
    """State for one viewer panel."""

    image_path: str | None = None
    loaded_source: "ViewerSourceState | None" = None
    current_frame_index: int = 0
    frame_count: int = 0
    loaded_image_data: np.ndarray | None = None
    input_colorspace: str = ""
    output_colorspace: str = ""
    look: str | None = None
    bypass: bool = False
    canvas_policy: GeometryCanvasPolicy = "native"
    scale_policy: GeometryScalePolicy = "fit"
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class CompareState:
    """Top-level viewer state for left/right panels."""

    left: PanelState = field(default_factory=PanelState)
    right: PanelState = field(default_factory=PanelState)
    active_side: ViewerSide = "left"
    alignment_anchor: AlignmentAnchor = "viewport_space"

    def panel(self, side: ViewerSide) -> PanelState:
        """Return panel state for the requested side."""
        return self.left if side == "left" else self.right

    @property
    def active_panel(self) -> PanelState:
        """Return state for the active panel."""
        return self.panel(self.active_side)


@dataclass
class CompareViewState:
    """UI-facing display mode state for compare rendering."""

    mode: CompareMode = "full"
    wipe_position: float = 0.5
    wipe_angle: float = 90.0
    wipe_top_side: ViewerSide = "left"
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    fit_mode: FitMode = "fit"
