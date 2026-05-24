"""Tests for frame service cache and stale-request behavior."""

from __future__ import annotations

import unittest

from PySide6.QtGui import QImage

from prism.core.frame_service import FrameService


class _DummySource:
    def __init__(self, source_id: str = "src", frame_count: int = 5) -> None:
        self.source_id = source_id
        self.current_index = 0
        self._frame_count = frame_count
        self.read_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def set_current_index(self, index: int) -> int:
        clamped = max(0, min(self._frame_count - 1, index))
        self.current_index = clamped
        return clamped

    def read_frame(self, index: int | None = None) -> QImage:
        if index is not None:
            self.set_current_index(index)
        self.read_count += 1
        return QImage(2, 2, QImage.Format.Format_RGB32)


class FrameServiceTests(unittest.TestCase):
    def test_marks_result_as_stale_for_old_token(self) -> None:
        source = _DummySource(source_id="a")
        service = FrameService(cache_size=8)
        old_token = service.next_request_token(source.source_id)
        service.next_request_token(source.source_id)

        result = service.get_frame(source, frame_index=0, token=old_token)

        self.assertTrue(result.stale)

    def test_marks_result_as_fresh_for_latest_token(self) -> None:
        source = _DummySource(source_id="a")
        service = FrameService(cache_size=8)
        latest_token = service.next_request_token(source.source_id)

        result = service.get_frame(source, frame_index=0, token=latest_token)

        self.assertFalse(result.stale)

    def test_reuses_cached_frame_without_extra_decode(self) -> None:
        source = _DummySource(source_id="cache")
        service = FrameService(cache_size=8)

        service.get_frame(source, frame_index=2)
        service.get_frame(source, frame_index=2)

        self.assertEqual(source.read_count, 1)


if __name__ == "__main__":
    unittest.main()

