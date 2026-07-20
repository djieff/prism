"""Concrete still/sequence/movie source models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from PySide6.QtGui import QImage

from prism.core.frame_source import FrameInfo, SourceKind
from prism.io.image.loader import load_image
from prism.io.media.movie_loader import get_movie_frame_count, load_movie_frame
from prism.io.media.source_detect import detect_source_input

_SEQUENCE_FRAME_PATTERN = re.compile(r"^.+\.(?P<digits>\d+)\.[^.]+$")


@dataclass
class BaseSource:
    """Common source state shared by all source kinds."""

    kind: SourceKind
    display_name: str
    source_id: str = field(default_factory=lambda: uuid4().hex)
    current_index: int = 0

    @property
    def frame_count(self) -> int:
        """Return total frame count for the source."""
        raise NotImplementedError

    def set_current_index(self, index: int) -> int:
        """Clamp and set the current frame index.

        Args:
            index: Requested frame index.

        Returns:
            Clamped frame index stored on the source.
        """
        clamped = max(0, min(max(self.frame_count - 1, 0), index))
        self.current_index = clamped
        return clamped

    def get_frame_info(self, index: int | None = None) -> FrameInfo:
        """Return frame metadata for the given or current index.

        Args:
            index: Optional target index. When omitted, current index is used.

        Returns:
            Frame info with index, display number, and label.
        """
        frame_index = self.current_index if index is None else self.set_current_index(index)
        display_number = frame_index + 1
        return FrameInfo(
            index=frame_index,
            display_number=display_number,
            label=str(display_number),
        )

    def read_frame(self, index: int | None = None) -> QImage:
        """Read and return a frame image.

        Args:
            index: Optional frame index to resolve before reading.

        Returns:
            Decoded frame image.
        """
        raise NotImplementedError


@dataclass
class StillImageSource(BaseSource):
    """Single-image source represented as one clamped frame."""

    path: Path = field(default_factory=Path)

    def __init__(self, path: Path) -> None:
        super().__init__(kind="still", display_name=path.name)
        self.path = path

    @property
    def frame_count(self) -> int:
        """Return frame count for still-image sources."""
        return 1

    def read_frame(self, index: int | None = None) -> QImage:
        """Load still image data as the current frame.

        Args:
            index: Optional frame index (clamped to still-image bounds).

        Returns:
            Loaded still image.
        """
        if index is not None:
            self.set_current_index(index)
        return load_image(str(self.path))


@dataclass
class ImageSequenceSource(BaseSource):
    """Image sequence source with index-addressable frame paths."""

    frame_paths: list[Path] = field(default_factory=list)

    def __init__(self, frame_paths: list[Path]) -> None:
        if not frame_paths:
            raise ValueError("Sequence source requires at least one frame path.")
        super().__init__(kind="sequence", display_name=frame_paths[0].name)
        self.frame_paths = frame_paths

    @property
    def frame_count(self) -> int:
        """Return total number of sequence frames."""
        return len(self.frame_paths)

    def get_frame_info(self, index: int | None = None) -> FrameInfo:
        """Return sequence frame metadata for the given/current index.

        Args:
            index: Optional target index. When omitted, current index is used.

        Returns:
            Frame info with extracted display number and filename label.
        """
        frame_index = self.current_index if index is None else self.set_current_index(index)
        frame_path = self.frame_paths[frame_index]
        display_number = _extract_sequence_frame_number(frame_path, frame_index)
        return FrameInfo(
            index=frame_index,
            display_number=display_number,
            label=frame_path.name,
        )

    def read_frame(self, index: int | None = None) -> QImage:
        """Load one sequence frame image.

        Args:
            index: Optional frame index to resolve before loading.

        Returns:
            Loaded sequence frame image.
        """
        frame_index = self.current_index if index is None else self.set_current_index(index)
        return load_image(str(self.frame_paths[frame_index]))


@dataclass
class MovieFileSource(BaseSource):
    """Movie source boundary for manual frame access."""

    path: Path = field(default_factory=Path)
    _frame_count: int | None = None

    def __init__(self, path: Path) -> None:
        super().__init__(kind="movie", display_name=path.name)
        self.path = path
        self._frame_count = None

    @property
    def frame_count(self) -> int:
        """Return movie frame count, resolving lazily once."""
        if self._frame_count is None:
            self._frame_count = get_movie_frame_count(self.path)
        return self._frame_count

    def read_frame(self, index: int | None = None) -> QImage:
        """Load one movie frame image by index.

        Args:
            index: Optional frame index to resolve before loading.

        Returns:
            Loaded movie frame image.
        """
        frame_index = self.current_index if index is None else self.set_current_index(index)
        return load_movie_frame(self.path, frame_index)


ViewerSourceState = StillImageSource | ImageSequenceSource | MovieFileSource


def create_source_from_path(path: Path) -> ViewerSourceState:
    """Build a concrete source model from an input path.

    Args:
        path: Input file path to classify and wrap.

    Returns:
        Source model instance for still image, sequence, or movie input.
    """
    detection = detect_source_input(path)
    if detection.kind == "sequence":
        return ImageSequenceSource(list(detection.sequence_files))
    if detection.kind == "movie":
        return MovieFileSource(path)
    return StillImageSource(path)


def _extract_sequence_frame_number(path: Path, frame_index: int) -> int:
    match = _SEQUENCE_FRAME_PATTERN.match(path.name)
    if match is None:
        return frame_index + 1
    return int(match.group("digits"))
