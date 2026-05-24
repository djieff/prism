"""Source-type detection for still/sequence/movie inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prism.core.frame_source import SourceKind
from prism.io.sequence_loader import detect_sequence_files

_MOVIE_EXTENSIONS = {".mov", ".mp4", ".mxf", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class SourceInputDetection:
    """Detection result for a dropped input file."""

    kind: SourceKind
    sequence_files: tuple[Path, ...] = ()


def detect_source_input(path: Path) -> SourceInputDetection:
    """Detect whether an input path is still, sequence, or movie.

    Args:
        path: Candidate input file path.

    Returns:
        Detection result containing source kind and optional sequence members.
    """
    suffix = path.suffix.lower()
    if suffix in _MOVIE_EXTENSIONS:
        return SourceInputDetection(kind="movie")

    sequence_files = detect_sequence_files(path)
    if sequence_files:
        return SourceInputDetection(kind="sequence", sequence_files=tuple(sequence_files))

    return SourceInputDetection(kind="still")
