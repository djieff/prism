"""Tests for still/sequence/movie input detection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prism.io.media.source_detect import detect_source_input


class SourceDetectTests(unittest.TestCase):
    def test_detects_movie_by_extension(self) -> None:
        detection = detect_source_input(Path("clip.MOV"))
        self.assertEqual(detection.kind, "movie")
        self.assertEqual(detection.sequence_files, ())

    def test_detects_sequence_when_group_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "render.0001.exr").write_bytes(b"")
            (root / "render.0002.exr").write_bytes(b"")

            detection = detect_source_input(root / "render.0002.exr")

            self.assertEqual(detection.kind, "sequence")
            self.assertEqual(
                [path.name for path in detection.sequence_files],
                ["render.0001.exr", "render.0002.exr"],
            )

    def test_detects_still_when_no_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            still = root / "plate.exr"
            still.write_bytes(b"")

            detection = detect_source_input(still)

            self.assertEqual(detection.kind, "still")
            self.assertEqual(detection.sequence_files, ())


if __name__ == "__main__":
    unittest.main()

