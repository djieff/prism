"""Shared view math for zoom/pan coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MIN_ZOOM = 0.1
MAX_ZOOM = 32.0
GeometryCanvasPolicy = Literal["native", "match_a", "match_b", "viewport"]
GeometryScalePolicy = Literal["fit", "fill", "stretch", "one_to_one"]


@dataclass(frozen=True)
class Size2D:
    """Width/height pair in pixels."""

    width: float
    height: float


@dataclass(frozen=True)
class Point2D:
    """X/Y pair in pixels."""

    x: float
    y: float


@dataclass(frozen=True)
class Rect2D:
    """Rectangle in viewport coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """Return rectangle right edge coordinate."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Return rectangle bottom edge coordinate."""
        return self.y + self.height


def clamp_zoom(zoom: float, minimum: float = MIN_ZOOM, maximum: float = MAX_ZOOM) -> float:
    """Clamp zoom value to configured bounds.

    Args:
        zoom: Requested zoom factor.
        minimum: Lower clamp bound.
        maximum: Upper clamp bound.

    Returns:
        Clamped zoom factor.
    """
    return max(minimum, min(maximum, zoom))


def fit_scale(image: Size2D, viewport: Size2D) -> float:
    """Compute fit scale that preserves image aspect ratio.

    Args:
        image: Source image size.
        viewport: Target viewport size.

    Returns:
        Uniform scale that fits the full image inside viewport.
    """
    image_w = max(image.width, 1.0)
    image_h = max(image.height, 1.0)
    view_w = max(viewport.width, 1.0)
    view_h = max(viewport.height, 1.0)
    return min(view_w / image_w, view_h / image_h)


def image_rect_in_viewport(
    image: Size2D, viewport: Size2D, zoom: float, pan: Point2D
) -> Rect2D:
    """Compute image draw rectangle in viewport space.

    Args:
        image: Source image size.
        viewport: Viewport size.
        zoom: Zoom multiplier applied after fit scale.
        pan: View-space pan offset.

    Returns:
        Image rectangle in viewport coordinates.
    """
    base_scale = fit_scale(image, viewport)
    draw_scale = base_scale * clamp_zoom(zoom)
    draw_w = max(image.width, 1.0) * draw_scale
    draw_h = max(image.height, 1.0) * draw_scale
    draw_x = ((viewport.width - draw_w) * 0.5) + pan.x
    draw_y = ((viewport.height - draw_h) * 0.5) + pan.y
    return Rect2D(draw_x, draw_y, draw_w, draw_h)


def clamp_pan(image: Size2D, viewport: Size2D, zoom: float, pan: Point2D) -> Point2D:
    """Clamp pan so image stays within center-bound movement limits.

    Args:
        image: Source image size.
        viewport: Viewport size.
        zoom: Active zoom multiplier.
        pan: Requested pan offset.

    Returns:
        Clamped pan offset.
    """
    rect = image_rect_in_viewport(image, viewport, zoom, Point2D(0.0, 0.0))

    max_x = max(0.0, (rect.width - viewport.width) * 0.5)
    max_y = max(0.0, (rect.height - viewport.height) * 0.5)
    clamped_x = max(-max_x, min(max_x, pan.x))
    clamped_y = max(-max_y, min(max_y, pan.y))
    return Point2D(clamped_x, clamped_y)


def view_to_image_point(point: Point2D, image_rect: Rect2D, image: Size2D) -> Point2D | None:
    """Map a viewport point into image pixel coordinate space.

    Args:
        point: Viewport-space point.
        image_rect: Image draw rectangle in viewport space.
        image: Source image size.

    Returns:
        Image-space point when inside image bounds, otherwise ``None``.
    """
    if image_rect.width <= 0 or image_rect.height <= 0:
        return None
    if point.x < image_rect.x or point.x > image_rect.right:
        return None
    if point.y < image_rect.y or point.y > image_rect.bottom:
        return None

    u = (point.x - image_rect.x) / image_rect.width
    v = (point.y - image_rect.y) / image_rect.height
    image_x = u * max(image.width - 1.0, 0.0)
    image_y = v * max(image.height - 1.0, 0.0)
    return Point2D(image_x, image_y)


def resolve_canvas_size(
    policy: GeometryCanvasPolicy,
    source: Size2D,
    size_a: Size2D,
    size_b: Size2D,
    viewport: Size2D,
) -> Size2D:
    """Resolve canvas size according to geometry policy.

    Args:
        policy: Canvas sizing policy.
        source: Current source size.
        size_a: Side A source size.
        size_b: Side B source size.
        viewport: Viewport size.

    Returns:
        Canvas size used for geometry mapping.
    """
    if policy == "match_a":
        return Size2D(max(size_a.width, 1.0), max(size_a.height, 1.0))
    if policy == "match_b":
        return Size2D(max(size_b.width, 1.0), max(size_b.height, 1.0))
    if policy == "viewport":
        return Size2D(max(viewport.width, 1.0), max(viewport.height, 1.0))
    return Size2D(max(source.width, 1.0), max(source.height, 1.0))


def source_rect_in_canvas(
    source: Size2D, canvas: Size2D, scale_policy: GeometryScalePolicy
) -> Rect2D:
    """Compute centered source rectangle in canvas space.

    Args:
        source: Source image size.
        canvas: Target canvas size.
        scale_policy: Scaling policy for fitting/filling/stretching.

    Returns:
        Source rectangle in canvas coordinates.
    """
    src_w = max(source.width, 1.0)
    src_h = max(source.height, 1.0)
    canvas_w = max(canvas.width, 1.0)
    canvas_h = max(canvas.height, 1.0)

    if scale_policy == "stretch":
        scale_x = canvas_w / src_w
        scale_y = canvas_h / src_h
    elif scale_policy == "one_to_one":
        scale_x = 1.0
        scale_y = 1.0
    elif scale_policy == "fill":
        uniform = max(canvas_w / src_w, canvas_h / src_h)
        scale_x = uniform
        scale_y = uniform
    else:
        uniform = min(canvas_w / src_w, canvas_h / src_h)
        scale_x = uniform
        scale_y = uniform

    draw_w = src_w * scale_x
    draw_h = src_h * scale_y
    draw_x = (canvas_w - draw_w) * 0.5
    draw_y = (canvas_h - draw_h) * 0.5
    return Rect2D(draw_x, draw_y, draw_w, draw_h)


def apply_offset(rect: Rect2D, offset: Point2D) -> Rect2D:
    """Translate a rectangle by offset in canvas space.

    Args:
        rect: Rectangle to translate.
        offset: Translation offset.

    Returns:
        Translated rectangle.
    """
    return Rect2D(rect.x + offset.x, rect.y + offset.y, rect.width, rect.height)


def canvas_to_source_point(
    point: Point2D, source_rect: Rect2D, source: Size2D
) -> Point2D | None:
    """Map canvas-space point into source pixel coordinates.

    Args:
        point: Canvas-space point.
        source_rect: Source rectangle mapped in canvas space.
        source: Source image size.

    Returns:
        Source-space point when inside mapped rect, otherwise ``None``.
    """
    if source_rect.width <= 0 or source_rect.height <= 0:
        return None
    if point.x < source_rect.x or point.x > source_rect.right:
        return None
    if point.y < source_rect.y or point.y > source_rect.bottom:
        return None

    u = (point.x - source_rect.x) / source_rect.width
    v = (point.y - source_rect.y) / source_rect.height
    image_x = u * max(source.width - 1.0, 0.0)
    image_y = v * max(source.height - 1.0, 0.0)
    return Point2D(image_x, image_y)
