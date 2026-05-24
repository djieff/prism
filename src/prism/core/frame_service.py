"""Frame decode orchestration with request-token staleness guards."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QImage

from prism.core.frame_cache import FrameCache
from prism.core.source_models import ViewerSourceState


@dataclass(frozen=True)
class FrameResult:
    """Decoded frame result with request staleness metadata."""

    image: QImage
    frame_index: int
    stale: bool = False


class FrameService:
    """Frame access service with cache and latest-request tracking."""

    def __init__(self, cache_size: int = 24) -> None:
        self._cache = FrameCache(max_items=cache_size)
        self._latest_request_by_source: dict[str, int] = {}
        self._request_counter = 0

    def next_request_token(self, source_id: str) -> int:
        """Create and register the next request token for a source.

        Args:
            source_id: Stable source identifier for stale-request tracking.

        Returns:
            Monotonic request token for the source.
        """
        self._request_counter += 1
        token = self._request_counter
        self._latest_request_by_source[source_id] = token
        return token

    def get_frame(
        self, source: ViewerSourceState, frame_index: int | None = None, token: int | None = None
    ) -> FrameResult:
        """Resolve a frame from cache or source and attach staleness metadata.

        Args:
            source: Source state used to read frame data.
            frame_index: Optional target frame index. When omitted, the source's
                current index is used.
            token: Optional request token used to flag stale responses.

        Returns:
            Decoded frame plus index and stale-request flag.
        """
        index = source.current_index if frame_index is None else source.set_current_index(frame_index)
        cached = self._cache.get(source.source_id, index)
        if cached is not None:
            return FrameResult(image=cached, frame_index=index, stale=self._is_stale(source.source_id, token))

        image = source.read_frame(index)
        self._cache.put(source.source_id, index, image)
        return FrameResult(image=image, frame_index=index, stale=self._is_stale(source.source_id, token))

    def _is_stale(self, source_id: str, token: int | None) -> bool:
        if token is None:
            return False
        latest = self._latest_request_by_source.get(source_id)
        return latest is not None and token != latest
