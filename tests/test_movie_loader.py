"""Tests for movie backend fallback behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QImage

from prism.io.media import movie_loader


class _StubBackend:
    def __init__(self, name: str, count_value: int = 0, raise_on_count: bool = False) -> None:
        self.name = name
        self._count_value = count_value
        self._raise_on_count = raise_on_count

    def get_frame_count(self, _: Path) -> int:
        if self._raise_on_count:
            raise RuntimeError("count failed")
        return self._count_value

    def load_frame(self, _: Path, __: int) -> QImage:
        return QImage(2, 2, QImage.Format.Format_RGB32)


class MovieLoaderTests(unittest.TestCase):
    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            movie_loader.get_movie_frame_count(Path("Z:/this/path/does/not/exist.mov"))

    def test_uses_first_working_backend_for_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            movie = Path(tmp) / "clip.mov"
            movie.write_bytes(b"dummy")
            backend = _StubBackend(name="stub-ok", count_value=12)
            with patch("prism.io.media.movie_loader._iter_backends", return_value=(backend,)):
                count = movie_loader.get_movie_frame_count(movie)
            self.assertEqual(count, 12)

    def test_falls_back_to_second_backend_for_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            movie = Path(tmp) / "clip.mov"
            movie.write_bytes(b"dummy")
            failing = _StubBackend(name="stub-fail", raise_on_count=True)
            success = _StubBackend(name="stub-ok", count_value=24)
            with patch(
                "prism.io.media.movie_loader._iter_backends",
                return_value=(failing, success),
            ):
                count = movie_loader.get_movie_frame_count(movie)
            self.assertEqual(count, 24)

    def test_reports_clear_error_when_no_backend_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            movie = Path(tmp) / "clip.mov"
            movie.write_bytes(b"dummy")
            with patch("prism.io.media.movie_loader._iter_backends", return_value=()):
                with self.assertRaises(RuntimeError) as raised:
                    movie_loader.load_movie_frame(movie, 0)
            self.assertIn("decoder backend not configured", str(raised.exception))
            self.assertIn("opencv-python", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

