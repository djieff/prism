"""Helpers for deterministic image-sequence grouping."""

from __future__ import annotations

import re
from pathlib import Path

_FRAME_PATTERN = re.compile(
    r"^(?P<name>.+)\.(?P<frame_number>\d+)(?P<file_ext>\.[^.]+)$"
)


def detect_sequence_files(seed_file: Path) -> list[Path]:
    """Detect sequence companions for a seed frame file.

    Args:
        seed_file: Candidate frame file inside a numbered sequence.

    Returns:
        Ordered sequence file list when at least two matching frames exist,
        otherwise an empty list.
    """
    match = _FRAME_PATTERN.match(seed_file.name)
    if match is None:
        return []

    name = match.group("name")
    file_ext = match.group("file_ext")
    parent = seed_file.parent
    if not parent.exists():
        return []

    matched: list[Path] = []
    for candidate in parent.iterdir():
        if not candidate.is_file():
            continue
        candidate_match = _FRAME_PATTERN.match(candidate.name)
        if candidate_match is None:
            continue
        if (
            candidate_match.group("name") == name
            and candidate_match.group("file_ext") == file_ext
        ):
            matched.append(candidate)

    if len(matched) < 2:
        return []

    matched.sort(key=_sequence_sort_key)
    return matched


def _sequence_sort_key(path: Path) -> tuple[str, int, str]:
    match = _FRAME_PATTERN.match(path.name)
    if match is None:
        return (path.name, 0, path.name)
    return (
        match.group("name"),
        int(match.group("frame_number")),
        path.name,
    )
