"""Tests for deterministic image-sequence grouping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prism.io.media.sequence_loader import detect_sequence_files


class SequenceLoaderTests(unittest.TestCase):
    def test_detects_and_sorts_matching_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shotA.0002.exr").write_bytes(b"")
            (root / "shotA.0010.exr").write_bytes(b"")
            (root / "shotA.0001.exr").write_bytes(b"")
            (root / "shotB.0001.exr").write_bytes(b"")

            result = detect_sequence_files(root / "shotA.0002.exr")

            self.assertEqual(
                [path.name for path in result],
                ["shotA.0001.exr", "shotA.0002.exr", "shotA.0010.exr"],
            )

    def test_returns_empty_for_single_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "single.0001.exr"
            seed.write_bytes(b"")

            result = detect_sequence_files(seed)

            self.assertEqual(result, [])

    def test_returns_empty_when_name_has_no_frame_digits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "plate.exr"
            seed.write_bytes(b"")

            result = detect_sequence_files(seed)

            self.assertEqual(result, [])

    def test_requires_dot_before_frame_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "plate0001.exr"
            b = root / "plate0002.exr"
            a.write_bytes(b"")
            b.write_bytes(b"")

            result = detect_sequence_files(a)

            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
