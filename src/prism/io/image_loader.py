"""Still image loading helpers"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QImageReader


def load_image(image_path: str) -> QImage:
    """Load a still image file into a displayable ``QImage``.

    Args:
        image_path: Filesystem path to an image file.

    Returns:
        Loaded image converted by Qt or OpenImageIO fallback.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()

    if not image.isNull():
        return image

    # Qt often lacks EXR/DPX codec support in default builds.
    if path.suffix.lower() in {".exr", ".dpx"}:
        fallback_image = _load_with_oiio(path)
        if fallback_image is not None:
            return fallback_image
        raise ValueError(
            f"Failed to load image '{image_path}': EXR/DPX support is unavailable. "
            "Install OpenImageIO or use a Qt build with EXR/DPX codecs."
        )

    error_text = reader.errorString() or "unsupported or invalid image format"
    raise ValueError(f"Failed to load image '{image_path}': {error_text}")


def _load_with_oiio(path: Path) -> QImage | None:
    """Load EXR/DPX through OpenImageIO fallback.

    Args:
        path: Image file path.

    Returns:
        Converted RGB ``QImage`` when OpenImageIO can decode the file,
        otherwise ``None``.
    """
    try:
        import OpenImageIO as oiio
    except ImportError:
        return None

    in_file = oiio.ImageInput.open(str(path))
    if in_file is None:
        return None

    try:
        spec = in_file.spec()
        pixels = in_file.read_image(oiio.FLOAT)
    finally:
        in_file.close()

    if pixels is None:
        return None

    image_data = np.asarray(pixels, dtype=np.float32).reshape(
        spec.height, spec.width, spec.nchannels
    )

    if image_data.shape[2] == 1:
        image_data = np.repeat(image_data, 3, axis=2)
    elif image_data.shape[2] > 3:
        image_data = image_data[:, :, :3]

    image_data = np.clip(image_data, 0.0, 1.0)
    rgb8 = (image_data * 255.0).astype(np.uint8)

    qimage = QImage(
        rgb8.data,
        spec.width,
        spec.height,
        spec.width * 3,
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()


def qimage_to_float_rgb(image: QImage) -> np.ndarray:
    """Convert ``QImage`` to normalized float32 RGB buffer.

    Args:
        image: Input image to convert.

    Returns:
        Float32 array shaped ``(height, width, 3)`` with values in ``[0.0, 1.0]``.
    """
    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()

    buffer = rgb_image.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8, count=bytes_per_line * height)
    array = array.reshape(height, bytes_per_line)
    rgb = array[:, : width * 3].reshape(height, width, 3)

    return rgb.astype(np.float32) / 255.0


def float_rgb_to_qimage(image_rgb: np.ndarray) -> QImage:
    """Convert normalized float32 RGB buffer to ``QImage``.

    Args:
        image_rgb: Float array shaped ``(height, width, 3)`` with values in
            ``[0.0, 1.0]``.

    Returns:
        Converted ``QImage`` in RGB888 format.

    Raises:
        ValueError: If input shape is not ``(height, width, 3)``.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image buffer shape (height, width, 3).")

    image_clamped = np.clip(image_rgb, 0.0, 1.0)
    image_u8 = (image_clamped * 255.0).astype(np.uint8)
    height, width, _ = image_u8.shape

    qimage = QImage(
        image_u8.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()
