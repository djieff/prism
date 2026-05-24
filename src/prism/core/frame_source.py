"""Shared frame-source contracts for still/sequence/movie inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["still", "sequence", "movie"]


@dataclass(frozen=True)
class FrameInfo:
    """Frame metadata exposed to UI and orchestration layers."""

    index: int
    display_number: int | None = None
    label: str | None = None
