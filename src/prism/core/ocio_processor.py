"""OCIO processing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_ocio_processor(
    config: Any,
    input_colorspace: str,
    output_colorspace: str,
    look: str | None = None,
    context_values: dict[str, str] | None = None,
) -> Any:
    """Build an OCIO processor for the selected transform route.

    Args:
        config: Loaded OCIO config object.
        input_colorspace: Source colorspace name.
        output_colorspace: Destination colorspace name.
        look: Optional OCIO look name to apply between source and destination.
        context_values: Optional OCIO context variable overrides.

    Returns:
        OCIO processor object configured for the requested transform.

    Raises:
        ValueError: If required colorspace names are missing or processor build
            fails in the OCIO backend.
    """
    if not input_colorspace or not output_colorspace:
        raise ValueError("Input and output colorspaces are required.")

    try:
        context = None
        if context_values:
            context = config.getCurrentContext()
            for name, value in context_values.items():
                context.setStringVar(name, value)

        if not look:
            if context is None:
                return config.getProcessor(input_colorspace, output_colorspace)
            return config.getProcessor(context, input_colorspace, output_colorspace)

        # Build: input colorspace -> look -> output colorspace.
        import PyOpenColorIO as ocio

        look_transform = ocio.LookTransform()
        look_transform.setSrc(input_colorspace)
        look_transform.setDst(output_colorspace)
        look_transform.setLooks(look)
        if context is None:
            return config.getProcessor(look_transform)
        return config.getProcessor(context, look_transform, ocio.TRANSFORM_DIR_FORWARD)
    except Exception as exc:  # pragma: no cover - library-level exception types vary
        look_label = f" with look '{look}'" if look else ""
        raise ValueError(
            f"Failed to build OCIO processor for '{input_colorspace}' -> '{output_colorspace}'{look_label}."
        ) from exc


def apply_ocio_transform(image_rgb: np.ndarray, processor: Any) -> np.ndarray:
    """Apply a prepared OCIO processor to an RGB image buffer.

    Args:
        image_rgb: RGB image array shaped ``(height, width, 3)``.
        processor: OCIO processor object returned by ``build_ocio_processor``.

    Returns:
        Transformed RGB float32 image array with the same shape as input.

    Raises:
        ValueError: If the input shape is invalid or OCIO application fails.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image buffer shape (height, width, 3).")

    output = np.ascontiguousarray(image_rgb, dtype=np.float32).copy()
    cpu_processor = processor.getDefaultCPUProcessor()

    try:
        cpu_processor.applyRGB(output.reshape(-1, 3))
    except Exception as exc:  # pragma: no cover - library-level exception types vary
        raise ValueError("Failed to apply OCIO transform to image buffer.") from exc

    return output
