"""Movie frame loading with optional backend adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtGui import QImage

_UNAVAILABLE_MESSAGE = (
    "Movie frame support is unavailable in this environment (decoder backend not configured). "
    "Install 'opencv-python' or 'imageio[ffmpeg]'."
)


@dataclass(frozen=True)
class _MovieBackend:
    """Runtime adapter contract for movie decoder backend operations."""

    name: str
    get_frame_count: Callable[[Path], int]
    load_frame: Callable[[Path, int], QImage]


def get_movie_frame_count(path: Path) -> int:
    """Return movie frame count using the first available backend.

    Args:
        path: Movie file path.

    Returns:
        Total number of decodable frames.

    Raises:
        FileNotFoundError: If the movie file does not exist.
        RuntimeError: If no backend can open/read frame count.
    """
    _assert_movie_file_exists(path)
    errors: list[str] = []
    for backend in _iter_backends():
        try:
            count = int(backend.get_frame_count(path))
            if count <= 0:
                raise RuntimeError(f"invalid frame count: {count}")
            return count
        except Exception as exc:
            errors.append(f"{backend.name}: {exc}")
    raise RuntimeError(_format_backend_error(errors))


def load_movie_frame(path: Path, frame_index: int) -> QImage:
    """Decode one movie frame using the first available backend.

    Args:
        path: Movie file path.
        frame_index: Zero-based frame index to decode.

    Returns:
        Decoded frame as ``QImage``.

    Raises:
        FileNotFoundError: If the movie file does not exist.
        ValueError: If ``frame_index`` is negative.
        RuntimeError: If no backend can decode the requested frame.
    """
    _assert_movie_file_exists(path)
    if frame_index < 0:
        raise ValueError("Frame index must be >= 0.")

    errors: list[str] = []
    for backend in _iter_backends():
        try:
            return backend.load_frame(path, frame_index)
        except Exception as exc:
            errors.append(f"{backend.name}: {exc}")
    raise RuntimeError(_format_backend_error(errors))


def _assert_movie_file_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Movie file not found: {path}")


def _format_backend_error(errors: list[str]) -> str:
    if not errors:
        return _UNAVAILABLE_MESSAGE
    details = "; ".join(errors)
    return f"{_UNAVAILABLE_MESSAGE} Tried backends: {details}"


def _iter_backends() -> tuple[_MovieBackend, ...]:
    backends: list[_MovieBackend] = []

    cv2_module = _import_cv2()
    if cv2_module is not None:
        def _cv2_get_frame_count(path: Path, mod: Any = cv2_module) -> int:
            return _get_frame_count_cv2(mod, path)

        def _cv2_load_frame(path: Path, index: int, mod: Any = cv2_module) -> QImage:
            return _load_frame_cv2(mod, path, index)

        backends.append(
            _MovieBackend(
                name="cv2",
                get_frame_count=_cv2_get_frame_count,
                load_frame=_cv2_load_frame,
            )
        )

    imageio_module = _import_imageio_v3()
    if imageio_module is not None:
        def _imageio_get_frame_count(path: Path, mod: Any = imageio_module) -> int:
            return _get_frame_count_imageio(mod, path)

        def _imageio_load_frame(path: Path, index: int, mod: Any = imageio_module) -> QImage:
            return _load_frame_imageio(mod, path, index)

        backends.append(
            _MovieBackend(
                name="imageio.v3",
                get_frame_count=_imageio_get_frame_count,
                load_frame=_imageio_load_frame,
            )
        )

    return tuple(backends)


def _import_cv2() -> Any | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    return cv2


def _import_imageio_v3() -> Any | None:
    try:
        import imageio.v3 as iio  # type: ignore[import-not-found]
    except Exception:
        return None
    return iio


def _get_frame_count_cv2(cv2: Any, path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("failed to open movie file")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            return frame_count

        # Fallback for files/backends that don't report CAP_PROP_FRAME_COUNT.
        scanned = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            scanned += 1
        if scanned <= 0:
            raise RuntimeError("failed to determine frame count")
        return scanned
    finally:
        capture.release()


def _load_frame_cv2(cv2: Any, path: Path, frame_index: int) -> QImage:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("failed to open movie file")
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
            raise RuntimeError("failed to seek frame")
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise IndexError(f"frame {frame_index} unavailable")

        if frame_bgr.ndim == 2:
            frame_rgb = np.repeat(frame_bgr[:, :, None], 3, axis=2)
        elif frame_bgr.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return _numpy_rgb_to_qimage(frame_rgb)
    finally:
        capture.release()


def _get_frame_count_imageio(iio: Any, path: Path) -> int:
    props = iio.improps(path)
    n_images = getattr(props, "n_images", None)
    if isinstance(n_images, int) and n_images > 0:
        return n_images

    scanned = sum(1 for _ in iio.imiter(path))
    if scanned <= 0:
        raise RuntimeError("failed to determine frame count")
    return scanned


def _load_frame_imageio(iio: Any, path: Path, frame_index: int) -> QImage:
    frame = np.asarray(iio.imread(path, index=frame_index))
    if frame.size == 0:
        raise IndexError(f"frame {frame_index} unavailable")
    return _numpy_to_qimage(frame)


def _numpy_to_qimage(frame: np.ndarray) -> QImage:
    if frame.ndim == 2:
        rgb = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        rgb = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        rgb = frame[:, :, :3]
    else:
        raise ValueError(f"Unsupported movie frame shape: {frame.shape}")
    return _numpy_rgb_to_qimage(rgb)


def _numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    if rgb.dtype == np.uint8:
        image_u8 = rgb
    elif np.issubdtype(rgb.dtype, np.floating):
        max_value = float(np.max(rgb)) if rgb.size else 1.0
        if max_value <= 1.0:
            image_u8 = np.clip(rgb, 0.0, 1.0) * 255.0
        else:
            image_u8 = np.clip(rgb, 0.0, 255.0)
        image_u8 = image_u8.astype(np.uint8)
    else:
        image_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    contiguous = np.ascontiguousarray(image_u8)
    height, width, _ = contiguous.shape
    qimage = QImage(
        contiguous.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()
