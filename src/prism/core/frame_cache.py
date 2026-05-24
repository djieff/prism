"""Small bounded cache for decoded source frames."""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtGui import QImage


class FrameCache:
    """LRU-like cache keyed by (source_id, frame_index)."""

    def __init__(self, max_items: int = 24) -> None:
        self._max_items = max(1, max_items)
        self._items: OrderedDict[tuple[str, int], QImage] = OrderedDict()

    def get(self, source_id: str, frame_index: int) -> QImage | None:
        """Fetch a cached frame image and refresh its LRU position.

        Args:
            source_id: Stable source identifier.
            frame_index: Frame index key for the source.

        Returns:
            Cached image when present, otherwise ``None``.
        """
        key = (source_id, frame_index)
        image = self._items.get(key)
        if image is None:
            return None
        self._items.move_to_end(key)
        return image

    def put(self, source_id: str, frame_index: int, image: QImage) -> None:
        """Store a frame image and evict oldest items above capacity.

        Args:
            source_id: Stable source identifier.
            frame_index: Frame index key for the source.
            image: Decoded frame image to cache.
        """
        key = (source_id, frame_index)
        self._items[key] = image
        self._items.move_to_end(key)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)

    def clear_source(self, source_id: str) -> None:
        """Remove all cached frames for a source.

        Args:
            source_id: Stable source identifier.
        """
        keys = [key for key in self._items if key[0] == source_id]
        for key in keys:
            self._items.pop(key, None)
