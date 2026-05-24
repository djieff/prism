"""Compare viewport widget for wipe and full-image modes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from prism.core.ui_tokens import AlignmentAnchor
from prism.core.viewer_state import CompareMode, ViewerSide
from prism.ui.view_math import (
    Point2D,
    Rect2D,
    Size2D,
    apply_offset,
    clamp_zoom,
    image_rect_in_viewport,
    resolve_canvas_size,
    source_rect_in_canvas,
    view_to_image_point,
)


@dataclass(frozen=True)
class _SideGeometry:
    """Per-side geometry policies and offsets for compare rendering."""

    canvas_policy: str = "native"
    scale_policy: str = "fit"
    offset_x: int = 0
    offset_y: int = 0


class CompareView(QWidget):
    """Render A/B image comparison with wipe and full display modes."""

    wipe_changed = Signal(float)
    wipe_angle_changed = Signal(float)
    view_changed = Signal(float, float, float)
    side_selected = Signal(str)
    hover_changed = Signal(float, float, bool, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Compare mode + side ownership.
        self._mode: CompareMode = "wipe"
        self._active_side: ViewerSide = "left"
        self._wipe_top_side: ViewerSide = "left"

        # Wipe geometry state.
        self._wipe_position = 0.5
        self._wipe_angle = 90.0
        # Offset of the visible wipe pivot (dot) along divider direction.
        self._wipe_pivot_offset = 0.0

        # Interaction drag state.
        self._dragging_wipe = False
        self._dragging_wipe_from_center_handle = False
        self._dragging_wipe_rotate = False
        # Rotation uses a fixed pivot captured on press for stable dragging.
        self._wipe_rotate_pivot: QPointF | None = None
        self._dragging_pan = False
        self._last_pan_pos: QPointF | None = None
        self._pan_moved = False
        self._space_pan_enabled = False

        # Shared view transform.
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # Image payloads.
        self._image_a: QImage | None = None
        self._image_b: QImage | None = None
        self._image_diff: QImage | None = None

        # View/UI configuration.
        self._placeholder_text = "Drop image here"
        self._alignment_anchor: str = AlignmentAnchor.VIEWPORT.value
        self._background_color = QColor(Qt.black)
        self._wipe_divider_hit_px = 6.0

        # Wipe unsynced nav stores a dedicated transform per side.
        self._wipe_unsynced_nav_enabled = False
        self._wipe_side_transforms: dict[ViewerSide, tuple[float, float, float]] = {
            "left": (1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
        }

        # Per-side source geometry controls.
        self._side_geometry: dict[ViewerSide, _SideGeometry] = {
            "left": _SideGeometry(),
            "right": _SideGeometry(),
        }

        # Temporary drop-hover target.
        self._drop_target_side: ViewerSide | None = None

    def set_background_color(self, color: QColor) -> None:
        """Set viewport background fill color.

        Args:
            color: Target background color.
        """
        if color == self._background_color:
            return
        self._background_color = QColor(color)
        self.update()

    def set_mode(self, mode: CompareMode) -> None:
        """Set the compare display mode and repaint when it changes.

        Args:
            mode: Target compare mode.
        """
        if mode == self._mode:
            return
        self._mode = mode
        self.update()

    def set_images(self, image_a: QImage | None, image_b: QImage | None) -> None:
        """Set left/right display images and repaint.

        Args:
            image_a: Display image for side A (left).
            image_b: Display image for side B (right).
        """
        self._image_a = image_a
        self._image_b = image_b
        self.update()

    def set_diff_image(self, image_diff: QImage | None) -> None:
        """Set diff display image and repaint.

        Args:
            image_diff: Diff visualization image for diff mode.
        """
        self._image_diff = image_diff
        self.update()

    def set_placeholder_text(self, text: str) -> None:
        """Set placeholder text shown when no images are loaded.

        Args:
            text: Placeholder label text.
        """
        self._placeholder_text = text
        self.update()

    def set_alignment_anchor(self, anchor: str) -> None:
        """Set compare-space alignment anchor and repaint.

        Args:
            anchor: Alignment anchor key used by compare-space math.
        """
        if anchor == self._alignment_anchor:
            return
        self._alignment_anchor = anchor
        self.update()

    def set_side_geometry(
        self,
        side: ViewerSide,
        *,
        canvas_policy: str,
        scale_policy: str,
        offset_x: int,
        offset_y: int,
    ) -> None:
        """Set per-side geometry controls and repaint when changed.

        Args:
            side: Target side to update.
            canvas_policy: Canvas sizing policy.
            scale_policy: Source scaling policy in canvas space.
            offset_x: Horizontal offset in source/canvas space.
            offset_y: Vertical offset in source/canvas space.
        """
        current = self._side_geometry[side]
        updated = _SideGeometry(
            canvas_policy=canvas_policy,
            scale_policy=scale_policy,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        if updated == current:
            return
        self._side_geometry[side] = updated
        self.update()

    def set_active_side(self, side: ViewerSide) -> None:
        """Set the active side used by full-mode fallback/selection logic.

        Args:
            side: Side to mark active.
        """
        if side == self._active_side:
            return
        self._active_side = side
        if self._mode == "wipe" and self._wipe_unsynced_nav_enabled:
            zoom, pan_x, pan_y = self._wipe_side_transforms[side]
            self._zoom = zoom
            self._pan_x = pan_x
            self._pan_y = pan_y
        self.update()

    def set_wipe_unsynced_nav_enabled(self, enabled: bool) -> None:
        if enabled == self._wipe_unsynced_nav_enabled:
            return
        self._wipe_unsynced_nav_enabled = enabled
        if enabled:
            self._wipe_side_transforms["left"] = (self._zoom, self._pan_x, self._pan_y)
            self._wipe_side_transforms["right"] = (self._zoom, self._pan_x, self._pan_y)
        else:
            self._wipe_side_transforms["left"] = (self._zoom, self._pan_x, self._pan_y)
            self._wipe_side_transforms["right"] = (self._zoom, self._pan_x, self._pan_y)
        self.update()

    def set_wipe_side_transform(
        self, side: ViewerSide, zoom: float, pan_x: float, pan_y: float
    ) -> None:
        self._wipe_side_transforms[side] = (clamp_zoom(zoom), pan_x, pan_y)
        if self._mode == "wipe" and self._wipe_unsynced_nav_enabled and side == self._active_side:
            self._zoom, self._pan_x, self._pan_y = self._wipe_side_transforms[side]
            self.update()

    def set_wipe_top_side(self, side: ViewerSide) -> None:
        """Set which side is displayed on the wipe-left half.

        Args:
            side: Side that should appear on the left/top wipe region.
        """
        if side == self._wipe_top_side:
            return
        self._wipe_top_side = side
        self.update()

    def set_wipe_position(self, value: float, emit_signal: bool = False) -> None:
        """Set normalized wipe divider position.

        Args:
            value: Divider position in normalized range [0.0, 1.0].
            emit_signal: When True, emit ``wipe_changed`` after update.
        """
        clamped = max(0.0, min(1.0, value))
        if abs(clamped - self._wipe_position) < 1e-6:
            return
        self._wipe_position = clamped
        self.update()
        if emit_signal:
            self.wipe_changed.emit(clamped)

    def set_wipe_angle(self, value: float, emit_signal: bool = False) -> None:
        normalized = ((value + 180.0) % 360.0) - 180.0
        if abs(normalized - self._wipe_angle) < 1e-6:
            return
        self._wipe_angle = normalized
        self.update()
        if emit_signal:
            self.wipe_angle_changed.emit(normalized)

    def side_for_position(self, pos: QPointF) -> ViewerSide | None:
        """Return compare side under a widget-local cursor position.

        Args:
            pos: Widget-local cursor position.

        Returns:
            Detected side when available, otherwise ``None``.
        """
        return self._side_at_position(pos)

    def set_drop_target_side(self, side: ViewerSide | None) -> None:
        """Set drop-hover target side used for temporary outline feedback."""
        if side == self._drop_target_side:
            return
        self._drop_target_side = side
        self.update()

    def set_view_transform(
        self, zoom: float, pan_x: float, pan_y: float, emit_signal: bool = False
    ) -> None:
        """Set zoom/pan transform for compare rendering.

        Args:
            zoom: Target zoom factor.
            pan_x: Horizontal pan offset in view space.
            pan_y: Vertical pan offset in view space.
            emit_signal: When True, emit ``view_changed`` after update.
        """
        if self._mode == "wipe" and self._wipe_unsynced_nav_enabled:
            side = self._active_side
            clamped_zoom = clamp_zoom(zoom)
            clamped_pan = Point2D(float(pan_x), float(pan_y))
            current_zoom, current_pan_x, current_pan_y = self._wipe_side_transforms[side]
            if (
                abs(clamped_zoom - current_zoom) < 1e-6
                and abs(clamped_pan.x - current_pan_x) < 1e-6
                and abs(clamped_pan.y - current_pan_y) < 1e-6
            ):
                return
            self._wipe_side_transforms[side] = (clamped_zoom, clamped_pan.x, clamped_pan.y)
            self._zoom = clamped_zoom
            self._pan_x = clamped_pan.x
            self._pan_y = clamped_pan.y
            self.update()
            if emit_signal:
                self.view_changed.emit(self._zoom, self._pan_x, self._pan_y)
            return

        image = self._reference_image_for_view_math()
        if image is None:
            self._zoom = clamp_zoom(zoom)
            self._pan_x = pan_x
            self._pan_y = pan_y
            self.update()
            if emit_signal:
                self.view_changed.emit(self._zoom, self._pan_x, self._pan_y)
            return

        clamped_zoom = clamp_zoom(zoom)
        # Keep pan responsive at all zoom levels; do not hard-clamp to fit bounds.
        clamped_pan = Point2D(pan_x, pan_y)

        if (
            abs(clamped_zoom - self._zoom) < 1e-6
            and abs(clamped_pan.x - self._pan_x) < 1e-6
            and abs(clamped_pan.y - self._pan_y) < 1e-6
        ):
            return

        self._zoom = clamped_zoom
        self._pan_x = clamped_pan.x
        self._pan_y = clamped_pan.y
        self.update()
        if emit_signal:
            self.view_changed.emit(self._zoom, self._pan_x, self._pan_y)

    def set_space_pan_enabled(self, enabled: bool) -> None:
        """Enable or disable space-modified pan interaction state.

        Args:
            enabled: Whether space-modified pan state is enabled.
        """
        if enabled == self._space_pan_enabled:
            return
        self._space_pan_enabled = enabled
        if not enabled:
            self._reset_drag_state()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        """Render compare background and active compare mode content.

        Args:
            event: Qt paint event.
        """
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), self._background_color)

        image_a = self._image_a
        image_b = self._image_b
        if image_a is None and image_b is None:
            if self._mode == "wipe":
                compare_rect = self._compare_space_rect()
                if compare_rect is not None:
                    self._draw_empty_wipe_placeholder(painter, compare_rect)
                    self._paint_drop_highlight_overlay(painter)
                    return
            self._draw_placeholder(painter)
            self._paint_drop_highlight_overlay(painter)
            return

        compare_rect = self._compare_space_rect()
        if compare_rect is None:
            return

        if self._mode == "full":
            self._paint_full_mode(painter, compare_rect)
            self._paint_drop_highlight_overlay(painter)
            return

        if self._mode == "wipe":
            self._paint_wipe_mode(painter, compare_rect)
            self._paint_drop_highlight_overlay(painter)
            return

        if self._mode == "diff":
            self._paint_diff_mode(painter, compare_rect)
            self._paint_drop_highlight_overlay(painter)
            return

        # "split" is handled by MainWindow routing for now.
        self._paint_full_mode(painter, compare_rect)
        self._paint_drop_highlight_overlay(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle wipe drag start, pan start, and selection press behavior.

        Args:
            event: Qt mouse press event.
        """
        if event.button() == Qt.RightButton:
            self._update_active_side_from_position_for_wipe_nav(event.position())
            if self._reference_image_for_view_math() is None:
                event.ignore()
                return
            self._dragging_pan = True
            self._last_pan_pos = event.position()
            self._pan_moved = True  # avoid click-selection emission on release
            self.grabMouse()
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        compare_rect = self._compare_space_rect()
        if compare_rect is None or not compare_rect.contains(event.position()):
            super().mousePressEvent(event)
            return

        if self._mode == "wipe" and self._is_near_wipe_angle_handle(event.position(), compare_rect):
            self._dragging_wipe_rotate = True
            self._wipe_rotate_pivot = self._wipe_pivot_center(compare_rect)
            self._set_wipe_angle_from_position(event.position(), compare_rect)
            event.accept()
            return

        divider_hit = False
        center_handle_hit = False
        if self._mode == "wipe":
            divider_hit = self._is_near_wipe_divider(event.position(), compare_rect)
            center_handle_hit = self._is_near_wipe_handle(event.position(), compare_rect)
        if divider_hit or (self._mode == "wipe" and center_handle_hit):
            self._dragging_wipe = True
            self._dragging_wipe_from_center_handle = center_handle_hit
            self._set_wipe_from_point(
                event.position(),
                compare_rect,
                emit_signal=True,
                update_pivot_offset=center_handle_hit,
            )
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle wipe dragging, panning, cursor updates, and hover emission.

        Args:
            event: Qt mouse move event.
        """
        if self._dragging_wipe_rotate and self._mode == "wipe":
            compare_rect = self._compare_space_rect()
            if compare_rect is not None:
                self._set_wipe_angle_from_position(event.position(), compare_rect)
            self.setCursor(Qt.CrossCursor)
        elif self._dragging_wipe and self._mode == "wipe":
            compare_rect = self._compare_space_rect()
            if compare_rect is not None:
                self._set_wipe_from_point(
                    event.position(),
                    compare_rect,
                    emit_signal=True,
                    update_pivot_offset=self._dragging_wipe_from_center_handle,
                )
            self.setCursor(Qt.SizeHorCursor)
        elif self._dragging_pan and self._last_pan_pos is not None:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()
            if abs(delta.x()) > 0.0 or abs(delta.y()) > 0.0:
                self._pan_moved = True
            self.set_view_transform(
                self._zoom, self._pan_x + delta.x(), self._pan_y + delta.y(), emit_signal=True
            )
            self._update_wipe_cursor(event.position())
        else:
            self._update_wipe_cursor(event.position())
        self._emit_hover(event.position())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finalize drag interactions and emit side selection clicks.

        Args:
            event: Qt mouse release event.
        """
        if event.button() == Qt.RightButton:
            self.releaseMouse()
            self._reset_drag_state()
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            self.releaseMouse()

            # Treat non-drag release as selection click.
            if not self._dragging_wipe and not self._dragging_wipe_rotate and not self._pan_moved:
                selected_side = self._side_at_position(event.position())
                if selected_side is not None:
                    self.side_selected.emit(selected_side)

            self._reset_drag_state()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Ignore double-click interaction to avoid unintended view changes."""
        event.accept()

    def _reset_drag_state(self) -> None:
        """Clear transient drag interaction state."""
        self._dragging_wipe = False
        self._dragging_wipe_from_center_handle = False
        self._dragging_wipe_rotate = False
        self._wipe_rotate_pivot = None
        self._dragging_pan = False
        self._last_pan_pos = None
        self._pan_moved = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Apply cursor-centric zoom for current mode.

        Args:
            event: Qt wheel event.
        """
        self._update_active_side_from_position_for_wipe_nav(event.position())
        image = self._reference_image_for_view_math()
        if image is None:
            event.ignore()
            return

        if self._mode == "full":
            if self._wheel_zoom_full_mode(event, image):
                event.accept()
            else:
                event.ignore()
            return

        viewport = Size2D(float(max(self.width(), 1)), float(max(self.height(), 1)))
        size_a = self._source_size(self._image_a) or viewport
        size_b = self._source_size(self._image_b) or viewport
        compare_space_size = self._compare_space_size(viewport, size_a, size_b)

        old_rect = image_rect_in_viewport(
            compare_space_size,
            viewport,
            self._zoom,
            Point2D(self._pan_x, self._pan_y),
        )
        if old_rect.width <= 0 or old_rect.height <= 0:
            event.ignore()
            return

        cursor = Point2D(event.position().x(), event.position().y())
        anchor_for_mapping = cursor

        image_point = view_to_image_point(anchor_for_mapping, old_rect, compare_space_size)
        if image_point is None:
            # Keep cursor-centric zoom even outside image bounds by clamping to nearest edge.
            clamped_anchor = Point2D(
                max(old_rect.x, min(old_rect.right, cursor.x)),
                max(old_rect.y, min(old_rect.bottom, cursor.y)),
            )
            anchor_for_mapping = clamped_anchor
            image_point = view_to_image_point(anchor_for_mapping, old_rect, compare_space_size)
        if image_point is None:
            event.ignore()
            return

        zoom_factor = 1.03 if event.angleDelta().y() > 0 else 1.0 / 1.03
        new_zoom = clamp_zoom(self._zoom * zoom_factor)
        new_rect = image_rect_in_viewport(
            compare_space_size, viewport, new_zoom, Point2D(self._pan_x, self._pan_y)
        )

        denom_x = max(compare_space_size.width - 1.0, 1.0)
        denom_y = max(compare_space_size.height - 1.0, 1.0)
        mapped_x = new_rect.x + (image_point.x / denom_x) * new_rect.width
        mapped_y = new_rect.y + (image_point.y / denom_y) * new_rect.height
        adjusted_pan_x = self._pan_x + (cursor.x - mapped_x)
        adjusted_pan_y = self._pan_y + (cursor.y - mapped_y)

        self.set_view_transform(new_zoom, adjusted_pan_x, adjusted_pan_y, emit_signal=True)
        event.accept()

    def _wheel_zoom_full_mode(self, event: QWheelEvent, image: QImage) -> bool:
        compare_rect = self._compare_space_rect()
        if compare_rect is None:
            return False

        side = self._active_display_side_for_full_mode()
        old_target_rect = self._side_target_rect(side, compare_rect, image)
        source_size = Size2D(float(max(image.width(), 1)), float(max(image.height(), 1)))
        cursor = Point2D(event.position().x(), event.position().y())

        anchor = view_to_image_point(cursor, self._rectf_to_rect2d(old_target_rect), source_size)
        if anchor is None:
            clamped = Point2D(
                max(old_target_rect.left(), min(old_target_rect.right(), cursor.x)),
                max(old_target_rect.top(), min(old_target_rect.bottom(), cursor.y)),
            )
            anchor = view_to_image_point(
                clamped, self._rectf_to_rect2d(old_target_rect), source_size
            )
        if anchor is None:
            return False

        # Convert source-anchor to normalized position in source draw rect.
        u = anchor.x / max(source_size.width - 1.0, 1.0)
        v = anchor.y / max(source_size.height - 1.0, 1.0)
        mapped_x_old = old_target_rect.x() + (u * old_target_rect.width())
        mapped_y_old = old_target_rect.y() + (v * old_target_rect.height())

        old_compare_rect = self._rectf_to_rect2d(compare_rect)
        coeff_x = (mapped_x_old - old_compare_rect.x) / max(old_compare_rect.width, 1.0)
        coeff_y = (mapped_y_old - old_compare_rect.y) / max(old_compare_rect.height, 1.0)

        zoom_factor = 1.03 if event.angleDelta().y() > 0 else 1.0 / 1.03
        new_zoom = clamp_zoom(self._zoom * zoom_factor)

        viewport = Size2D(float(max(self.width(), 1)), float(max(self.height(), 1)))
        size_a = self._source_size(self._image_a) or viewport
        size_b = self._source_size(self._image_b) or viewport
        compare_space_size = self._compare_space_size(viewport, size_a, size_b)
        base_rect = image_rect_in_viewport(
            compare_space_size, viewport, new_zoom, Point2D(0.0, 0.0)
        )

        new_pan_x = cursor.x - (base_rect.x + (coeff_x * base_rect.width))
        new_pan_y = cursor.y - (base_rect.y + (coeff_y * base_rect.height))
        self.set_view_transform(new_zoom, new_pan_x, new_pan_y, emit_signal=True)
        return True

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        """Clear hover/cursor state when mouse leaves the widget.

        Args:
            event: Qt leave event.
        """
        self.unsetCursor()
        self.hover_changed.emit(-1.0, -1.0, False, "")
        super().leaveEvent(event)

    def _paint_full_mode(self, painter: QPainter, compare_rect: QRectF) -> None:
        image = self._image_a if self._active_side == "left" else self._image_b
        if image is None:
            self._draw_placeholder(painter)
            return
        target_rect = self._side_target_rect(self._active_side, compare_rect, image)
        self._draw_image(painter, image, target_rect)

    def _paint_wipe_mode(self, painter: QPainter, compare_rect: QRectF) -> None:
        image_a = self._image_a
        image_b = self._image_b
        if image_a is None and image_b is None:
            return

        center = self._wipe_pivot_center(compare_rect)
        direction = self._wipe_direction()
        positive_polygon, negative_polygon = self._wipe_side_polygons(compare_rect, center, direction)

        positive_image: QImage | None = None
        negative_image: QImage | None = None
        positive_side: ViewerSide = "left"
        negative_side: ViewerSide = "right"
        if self._wipe_top_side == "left":
            positive_image = image_a
            negative_image = image_b
            positive_side = "left"
            negative_side = "right"
        else:
            positive_image = image_b
            negative_image = image_a
            positive_side = "right"
            negative_side = "left"

        positive_target = (
            self._side_target_rect(
                positive_side, self._compare_space_rect_for_side(positive_side), positive_image
            )
            if positive_image is not None
            else None
        )
        negative_target = (
            self._side_target_rect(
                negative_side, self._compare_space_rect_for_side(negative_side), negative_image
            )
            if negative_image is not None
            else None
        )

        if positive_target is not None and len(positive_polygon) >= 3:
            painter.save()
            clip_path = QPainterPath()
            clip_path.moveTo(positive_polygon[0])
            for point in positive_polygon[1:]:
                clip_path.lineTo(point)
            clip_path.closeSubpath()
            painter.setClipPath(clip_path)
            self._draw_image(painter, positive_image, positive_target)
            painter.restore()

        if negative_target is not None and len(negative_polygon) >= 3:
            painter.save()
            clip_path = QPainterPath()
            clip_path.moveTo(negative_polygon[0])
            for point in negative_polygon[1:]:
                clip_path.lineTo(point)
            clip_path.closeSubpath()
            painter.setClipPath(clip_path)
            self._draw_image(painter, negative_image, negative_target)
            painter.restore()

        painter.save()
        painter.setPen(QColor(170, 170, 170))
        if positive_image is None and len(positive_polygon) >= 3:
            missing_positive_text = (
                "Drop image A here" if positive_side == "left" else "Drop image B here"
            )
            painter.drawText(
                self._polygon_bounds(positive_polygon).toRect(),
                Qt.AlignCenter,
                missing_positive_text,
            )
        if negative_image is None and len(negative_polygon) >= 3:
            missing_negative_text = (
                "Drop image A here" if negative_side == "left" else "Drop image B here"
            )
            painter.drawText(
                self._polygon_bounds(negative_polygon).toRect(),
                Qt.AlignCenter,
                missing_negative_text,
            )
        painter.restore()

        painter.save()
        line_start, line_end = self._wipe_divider_endpoints(center, direction)
        painter.setPen(QPen(QColor(0, 0, 0, 220), 4))
        painter.drawLine(line_start, line_end)
        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(line_start, line_end)
        self._draw_wipe_center_handle(painter, center)
        self._draw_wipe_angle_handles(painter, compare_rect, center)
        painter.restore()

    def _paint_diff_mode(self, painter: QPainter, compare_rect: QRectF) -> None:
        if self._image_diff is None:
            self._paint_full_mode(painter, compare_rect)
            return
        target_rect = self._fit_image_rect(self._image_diff, compare_rect)
        self._draw_image(painter, self._image_diff, target_rect)

    def _draw_image(self, painter: QPainter, image: QImage | None, target_rect: QRectF) -> None:
        if image is None:
            return
        pixmap = QPixmap.fromImage(image)
        painter.drawPixmap(target_rect.toRect(), pixmap)

    def _fit_image_rect(self, image: QImage, bounds: QRectF) -> QRectF:
        image_w = float(max(image.width(), 1))
        image_h = float(max(image.height(), 1))
        bounds_w = max(bounds.width(), 1.0)
        bounds_h = max(bounds.height(), 1.0)
        scale = min(bounds_w / image_w, bounds_h / image_h)
        draw_w = image_w * scale
        draw_h = image_h * scale
        return QRectF(
            bounds.x() + ((bounds_w - draw_w) * 0.5),
            bounds.y() + ((bounds_h - draw_h) * 0.5),
            draw_w,
            draw_h,
        )

    def _image_target_rect(self) -> QRectF | None:
        return self._compare_space_rect()

    def _set_wipe_from_point(
        self,
        pos: QPointF,
        target_rect: QRectF,
        emit_signal: bool,
        *,
        update_pivot_offset: bool,
    ) -> None:
        """Update wipe position from a cursor point.

        The wipe line always moves by projecting the cursor onto the wipe
        normal. When ``update_pivot_offset`` is true (center-handle drag), the
        dot/pivot offset along the divider direction is updated as well.
        """
        min_proj, max_proj = self._wipe_projection_bounds(target_rect)
        span = max_proj - min_proj
        if span <= 1e-6:
            return
        direction = self._wipe_direction()
        normal = QPointF(-direction.y(), direction.x())
        projection = self._dot(pos, normal)
        relative = (projection - min_proj) / span
        self.set_wipe_position(relative, emit_signal=emit_signal)
        if not update_pivot_offset:
            return
        base_center = self._wipe_center(target_rect)
        self._wipe_pivot_offset = self._dot(
            QPointF(pos.x() - base_center.x(), pos.y() - base_center.y()),
            direction,
        )
        self.update()

    def _side_at_position(self, pos: QPointF) -> ViewerSide | None:
        if self._mode == "full":
            return self._active_side
        if self._mode != "wipe":
            return None

        compare_rect = self._compare_space_rect()
        if compare_rect is None or not compare_rect.contains(pos):
            return None

        center = self._wipe_pivot_center(compare_rect)
        direction = self._wipe_direction()
        normal = QPointF(-direction.y(), direction.x())
        signed = self._dot(QPointF(pos.x() - center.x(), pos.y() - center.y()), normal)
        is_positive_side = signed >= 0.0
        if self._wipe_top_side == "left":
            return "left" if is_positive_side else "right"
        return "right" if is_positive_side else "left"

    def _reference_image_for_view_math(self) -> QImage | None:
        if self._mode == "diff":
            return self._image_diff
        if self._mode == "full":
            return self._image_a if self._active_side == "left" else self._image_b
        return self._image_a or self._image_b

    def _is_near_wipe_divider(self, pos: QPointF, target_rect: QRectF) -> bool:
        center = self._wipe_pivot_center(target_rect)
        direction = self._wipe_direction()
        normal = QPointF(-direction.y(), direction.x())
        distance = abs(self._dot(QPointF(pos.x() - center.x(), pos.y() - center.y()), normal))
        return distance <= self._wipe_divider_hit_px

    def _is_near_wipe_handle(self, pos: QPointF, target_rect: QRectF) -> bool:
        center = self._wipe_pivot_center(target_rect)
        return self._distance_sq(pos, center) <= (12.0 * 12.0)

    def _is_near_wipe_angle_handle(self, pos: QPointF, target_rect: QRectF) -> bool:
        start, end = self._wipe_line_endpoints(target_rect, self._wipe_pivot_center(target_rect))
        return (
            self._distance_sq(pos, start) <= (12.0 * 12.0)
            or self._distance_sq(pos, end) <= (12.0 * 12.0)
        )

    def _set_wipe_angle_from_position(self, pos: QPointF, target_rect: QRectF) -> None:
        """Rotate wipe angle around the active pivot for a cursor position.

        Uses the fixed rotate pivot captured on press when available, and
        selects the equivalent divider orientation closest to current angle to
        avoid 180-degree flips between end handles.
        """
        pivot = self._wipe_rotate_pivot or self._wipe_pivot_center(target_rect)
        dx = pos.x() - pivot.x()
        dy = pos.y() - pivot.y()
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        raw_angle = math.degrees(math.atan2(dy, dx))
        candidate_a = ((raw_angle + 180.0) % 360.0) - 180.0
        candidate_b = ((raw_angle + 180.0 + 180.0) % 360.0) - 180.0
        diff_a = abs(((candidate_a - self._wipe_angle + 180.0) % 360.0) - 180.0)
        diff_b = abs(((candidate_b - self._wipe_angle + 180.0) % 360.0) - 180.0)
        chosen_angle = candidate_a if diff_a <= diff_b else candidate_b
        self.set_wipe_angle(chosen_angle, emit_signal=True)
        min_proj, max_proj = self._wipe_projection_bounds(target_rect)
        span = max_proj - min_proj
        if span <= 1e-6:
            return
        direction = self._wipe_direction()
        normal = QPointF(-direction.y(), direction.x())
        projection = self._dot(pivot, normal)
        relative = (projection - min_proj) / span
        self.set_wipe_position(relative, emit_signal=False)
        base_center = self._wipe_center(target_rect)
        self._wipe_pivot_offset = self._dot(
            QPointF(pivot.x() - base_center.x(), pivot.y() - base_center.y()),
            direction,
        )
        self.update()

    def _update_wipe_cursor(self, pos: QPointF) -> None:
        if self._mode != "wipe":
            self.unsetCursor()
            return
        compare_rect = self._compare_space_rect()
        if compare_rect is None or not compare_rect.contains(pos):
            self.unsetCursor()
            return
        if self._is_near_wipe_angle_handle(pos, compare_rect):
            self.setCursor(Qt.CrossCursor)
            return
        if self._is_near_wipe_handle(pos, compare_rect):
            self.setCursor(Qt.SizeAllCursor)
            return
        if self._is_near_wipe_divider(pos, compare_rect):
            self.setCursor(Qt.SizeHorCursor)
            return
        self.unsetCursor()

    def _draw_wipe_center_handle(self, painter: QPainter, center: QPointF) -> None:
        outer_radius = 7.0
        inner_radius = 4.0
        cx = int(round(center.x()))
        cy = int(round(center.y()))

        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 220), 2))
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.drawEllipse(QPointF(float(cx), float(cy)), outer_radius, outer_radius)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
        painter.setBrush(QColor(25, 25, 25, 200))
        painter.drawEllipse(QPointF(float(cx), float(cy)), inner_radius, inner_radius)
        painter.restore()

    def _draw_wipe_angle_handles(
        self,
        painter: QPainter,
        target_rect: QRectF,
        center: QPointF,
    ) -> None:
        start, end = self._wipe_line_endpoints(target_rect, center)
        radius = 6.0
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 220), 2))
        painter.setBrush(QColor(245, 245, 245, 235))
        painter.drawEllipse(start, radius, radius)
        painter.drawEllipse(end, radius, radius)
        painter.restore()

    def _draw_empty_wipe_placeholder(self, painter: QPainter, compare_rect: QRectF) -> None:
        center = self._wipe_pivot_center(compare_rect)
        direction = self._wipe_direction()
        positive_polygon, negative_polygon = self._wipe_side_polygons(compare_rect, center, direction)

        painter.save()
        line_start, line_end = self._wipe_divider_endpoints(center, direction)
        painter.setPen(QPen(QColor(0, 0, 0, 220), 4))
        painter.drawLine(line_start, line_end)
        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(line_start, line_end)
        self._draw_wipe_center_handle(painter, center)
        self._draw_wipe_angle_handles(painter, compare_rect, center)

        text_color = QColor(170, 170, 170)
        painter.setPen(text_color)
        if len(positive_polygon) >= 3:
            painter.drawText(
                self._polygon_bounds(positive_polygon).toRect(),
                Qt.AlignCenter,
                "Drop image A here",
            )
        if len(negative_polygon) >= 3:
            painter.drawText(
                self._polygon_bounds(negative_polygon).toRect(),
                Qt.AlignCenter,
                "Drop image B here",
            )
        painter.restore()

    def _paint_drop_highlight_overlay(self, painter: QPainter) -> None:
        target_side = self._drop_target_side
        if target_side is None:
            return
        compare_rect = self._compare_space_rect()
        if compare_rect is None:
            return
        target_rect = self._drop_target_rect_for_side(target_side, compare_rect)
        if target_rect is None or target_rect.width() <= 0 or target_rect.height() <= 0:
            return

        painter.save()
        painter.setPen(QPen(QColor(90, 170, 255, 230), 2))
        painter.setBrush(QColor(90, 170, 255, 26))
        painter.drawRect(target_rect)
        painter.restore()

    def _drop_target_rect_for_side(
        self, side: ViewerSide, compare_rect: QRectF
    ) -> QRectF | None:
        if self._mode != "wipe":
            return compare_rect
        center = self._wipe_center(compare_rect)
        direction = self._wipe_direction()
        positive_polygon, negative_polygon = self._wipe_side_polygons(compare_rect, center, direction)
        if self._wipe_top_side == "left":
            target_polygon = positive_polygon if side == "left" else negative_polygon
        else:
            target_polygon = positive_polygon if side == "right" else negative_polygon
        if len(target_polygon) < 3:
            return None
        return self._polygon_bounds(target_polygon)

    def _active_display_side_for_full_mode(self) -> ViewerSide:
        return self._active_side

    def _rectf_to_rect2d(self, rect: QRectF) -> Rect2D:
        return Rect2D(rect.x(), rect.y(), rect.width(), rect.height())

    def _compare_space_rect(self) -> QRectF | None:
        if self._mode == "wipe":
            return QRectF(self.rect())
        return self._compare_space_rect_for_transform(self._zoom, self._pan_x, self._pan_y)

    def _compare_space_rect_for_side(self, side: ViewerSide) -> QRectF:
        if self._mode == "wipe" and self._wipe_unsynced_nav_enabled:
            zoom, pan_x, pan_y = self._wipe_side_transforms[side]
            rect = self._compare_space_rect_for_transform(zoom, pan_x, pan_y)
            if rect is not None:
                return rect
        rect = self._compare_space_rect_for_transform(self._zoom, self._pan_x, self._pan_y)
        if rect is None:
            return QRectF(self.rect())
        return rect

    def _compare_space_rect_for_transform(
        self, zoom: float, pan_x: float, pan_y: float
    ) -> QRectF | None:
        viewport = Size2D(float(max(self.width(), 1)), float(max(self.height(), 1)))
        size_a = self._source_size(self._image_a) or viewport
        size_b = self._source_size(self._image_b) or viewport

        if self._alignment_anchor == AlignmentAnchor.A_SPACE.value and self._image_a is not None:
            compare_space_size = self._canvas_size_for_side("left", size_a, size_a, size_b, viewport)
        elif self._alignment_anchor == AlignmentAnchor.B_SPACE.value and self._image_b is not None:
            compare_space_size = self._canvas_size_for_side(
                "right", size_b, size_a, size_b, viewport
            )
        else:
            compare_space_size = viewport

        rect = image_rect_in_viewport(
            compare_space_size,
            viewport,
            zoom,
            Point2D(pan_x, pan_y),
        )
        return QRectF(rect.x, rect.y, rect.width, rect.height)

    def _source_size(self, image: QImage | None) -> Size2D | None:
        if image is None:
            return None
        return Size2D(float(max(image.width(), 1)), float(max(image.height(), 1)))

    def _canvas_size_for_side(
        self,
        side: ViewerSide,
        source_size: Size2D,
        size_a: Size2D,
        size_b: Size2D,
        viewport: Size2D,
    ) -> Size2D:
        geometry = self._side_geometry[side]
        return resolve_canvas_size(geometry.canvas_policy, source_size, size_a, size_b, viewport)

    def _side_target_rect(
        self, side: ViewerSide, compare_rect: QRectF, image: QImage | None
    ) -> QRectF:
        source_size = self._source_size(image)
        if source_size is None:
            return compare_rect

        viewport = Size2D(float(max(self.width(), 1)), float(max(self.height(), 1)))
        size_a = self._source_size(self._image_a) or viewport
        size_b = self._source_size(self._image_b) or viewport
        compare_space_size = self._compare_space_size(viewport, size_a, size_b)

        canvas_size = self._canvas_size_for_side(side, source_size, size_a, size_b, viewport)
        geometry = self._side_geometry[side]
        source_rect = source_rect_in_canvas(source_size, canvas_size, geometry.scale_policy)
        source_rect = apply_offset(source_rect, Point2D(float(geometry.offset_x), float(geometry.offset_y)))

        # Map side canvas-space rect into compare-space with uniform scale (no stretch).
        canvas_to_compare = min(
            compare_space_size.width / max(canvas_size.width, 1.0),
            compare_space_size.height / max(canvas_size.height, 1.0),
        )
        mapped_canvas_w = canvas_size.width * canvas_to_compare
        mapped_canvas_h = canvas_size.height * canvas_to_compare
        canvas_offset_x = (compare_space_size.width - mapped_canvas_w) * 0.5
        canvas_offset_y = (compare_space_size.height - mapped_canvas_h) * 0.5

        compare_x = canvas_offset_x + (source_rect.x * canvas_to_compare)
        compare_y = canvas_offset_y + (source_rect.y * canvas_to_compare)
        compare_w = source_rect.width * canvas_to_compare
        compare_h = source_rect.height * canvas_to_compare

        ratio_x = compare_rect.width() / max(compare_space_size.width, 1.0)
        ratio_y = compare_rect.height() / max(compare_space_size.height, 1.0)
        return QRectF(
            compare_rect.x() + (compare_x * ratio_x),
            compare_rect.y() + (compare_y * ratio_y),
            compare_w * ratio_x,
            compare_h * ratio_y,
        )

    def _compare_space_size(self, viewport: Size2D, size_a: Size2D, size_b: Size2D) -> Size2D:
        if self._alignment_anchor == AlignmentAnchor.A_SPACE.value and self._image_a is not None:
            return self._canvas_size_for_side("left", size_a, size_a, size_b, viewport)
        if self._alignment_anchor == AlignmentAnchor.B_SPACE.value and self._image_b is not None:
            return self._canvas_size_for_side("right", size_b, size_a, size_b, viewport)
        return viewport

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.save()
        painter.setPen(QPen(Qt.gray))
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder_text)
        painter.restore()

    def _wipe_direction(self) -> QPointF:
        radians = math.radians(self._wipe_angle)
        return QPointF(math.cos(radians), math.sin(radians))

    def _wipe_center(self, target_rect: QRectF) -> QPointF:
        direction = self._wipe_direction()
        normal = QPointF(-direction.y(), direction.x())
        min_proj, max_proj = self._wipe_projection_bounds(target_rect)
        center_projection = min_proj + (self._wipe_position * (max_proj - min_proj))
        rect_center = target_rect.center()
        rect_center_projection = self._dot(rect_center, normal)
        delta = center_projection - rect_center_projection
        return QPointF(
            rect_center.x() + (normal.x() * delta),
            rect_center.y() + (normal.y() * delta),
        )

    def _wipe_pivot_center(self, target_rect: QRectF) -> QPointF:
        center = self._wipe_center(target_rect)
        direction = self._wipe_direction()
        return QPointF(
            center.x() + (direction.x() * self._wipe_pivot_offset),
            center.y() + (direction.y() * self._wipe_pivot_offset),
        )

    def _wipe_divider_endpoints(
        self, center: QPointF, direction: QPointF
    ) -> tuple[QPointF, QPointF]:
        """Return long line endpoints for drawing the wipe divider."""
        length = 10000.0
        return (
            QPointF(center.x() - (direction.x() * length), center.y() - (direction.y() * length)),
            QPointF(center.x() + (direction.x() * length), center.y() + (direction.y() * length)),
        )

    def _wipe_side_polygons(
        self, compare_rect: QRectF, center: QPointF, direction: QPointF
    ) -> tuple[list[QPointF], list[QPointF]]:
        """Split compare rect into positive/negative polygons for wipe rendering."""
        normal = QPointF(-direction.y(), direction.x())
        rect_points = [
            QPointF(compare_rect.left(), compare_rect.top()),
            QPointF(compare_rect.right(), compare_rect.top()),
            QPointF(compare_rect.right(), compare_rect.bottom()),
            QPointF(compare_rect.left(), compare_rect.bottom()),
        ]
        return (
            self._clip_polygon_half_plane(rect_points, center, normal, keep_positive=True),
            self._clip_polygon_half_plane(rect_points, center, normal, keep_positive=False),
        )

    def _wipe_line_endpoints(self, target_rect: QRectF, center: QPointF) -> tuple[QPointF, QPointF]:
        """Compute divider segment endpoints clipped to the target rectangle."""
        direction = self._wipe_direction()
        dx = direction.x()
        dy = direction.y()
        candidates: list[tuple[float, QPointF]] = []

        if abs(dx) > 1e-9:
            t_left = (target_rect.left() - center.x()) / dx
            y_left = center.y() + (t_left * dy)
            if target_rect.top() - 1e-6 <= y_left <= target_rect.bottom() + 1e-6:
                candidates.append((t_left, QPointF(target_rect.left(), y_left)))

            t_right = (target_rect.right() - center.x()) / dx
            y_right = center.y() + (t_right * dy)
            if target_rect.top() - 1e-6 <= y_right <= target_rect.bottom() + 1e-6:
                candidates.append((t_right, QPointF(target_rect.right(), y_right)))

        if abs(dy) > 1e-9:
            t_top = (target_rect.top() - center.y()) / dy
            x_top = center.x() + (t_top * dx)
            if target_rect.left() - 1e-6 <= x_top <= target_rect.right() + 1e-6:
                candidates.append((t_top, QPointF(x_top, target_rect.top())))

            t_bottom = (target_rect.bottom() - center.y()) / dy
            x_bottom = center.x() + (t_bottom * dx)
            if target_rect.left() - 1e-6 <= x_bottom <= target_rect.right() + 1e-6:
                candidates.append((t_bottom, QPointF(x_bottom, target_rect.bottom())))

        if len(candidates) < 2:
            fallback = QPointF(center.x() - (dx * 1000.0), center.y() - (dy * 1000.0))
            fallback_end = QPointF(center.x() + (dx * 1000.0), center.y() + (dy * 1000.0))
            return fallback, fallback_end

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], candidates[-1][1]

    def _wipe_projection_bounds(self, target_rect: QRectF) -> tuple[float, float]:
        direction = self._wipe_direction()
        normal = QPointF(-direction.y(), direction.x())
        corners = (
            QPointF(target_rect.left(), target_rect.top()),
            QPointF(target_rect.right(), target_rect.top()),
            QPointF(target_rect.right(), target_rect.bottom()),
            QPointF(target_rect.left(), target_rect.bottom()),
        )
        projections = [self._dot(point, normal) for point in corners]
        return (min(projections), max(projections))

    def _clip_polygon_half_plane(
        self,
        polygon: list[QPointF],
        origin: QPointF,
        normal: QPointF,
        *,
        keep_positive: bool,
    ) -> list[QPointF]:
        if not polygon:
            return []
        output: list[QPointF] = []

        def signed(point: QPointF) -> float:
            return self._dot(QPointF(point.x() - origin.x(), point.y() - origin.y()), normal)

        def inside(value: float) -> bool:
            return value >= 0.0 if keep_positive else value <= 0.0

        previous = polygon[-1]
        previous_signed = signed(previous)
        previous_inside = inside(previous_signed)
        for current in polygon:
            current_signed = signed(current)
            current_inside = inside(current_signed)

            if current_inside != previous_inside:
                denom = current_signed - previous_signed
                if abs(denom) > 1e-9:
                    t = -previous_signed / denom
                    intersection = QPointF(
                        previous.x() + ((current.x() - previous.x()) * t),
                        previous.y() + ((current.y() - previous.y()) * t),
                    )
                    output.append(intersection)
            if current_inside:
                output.append(current)

            previous = current
            previous_signed = current_signed
            previous_inside = current_inside
        return output

    def _polygon_bounds(self, polygon: list[QPointF]) -> QRectF:
        min_x = min(point.x() for point in polygon)
        min_y = min(point.y() for point in polygon)
        max_x = max(point.x() for point in polygon)
        max_y = max(point.y() for point in polygon)
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def _dot(self, a: QPointF, b: QPointF) -> float:
        return (a.x() * b.x()) + (a.y() * b.y())

    def _distance_sq(self, a: QPointF, b: QPointF) -> float:
        dx = a.x() - b.x()
        dy = a.y() - b.y()
        return (dx * dx) + (dy * dy)

    def _update_active_side_from_position_for_wipe_nav(self, pos: QPointF) -> None:
        if not (self._mode == "wipe" and self._wipe_unsynced_nav_enabled):
            return
        side = self._side_at_position(pos)
        if side is None or side == self._active_side:
            return
        self.set_active_side(side)

    def _emit_hover(self, pos: QPointF) -> None:
        image = self._reference_image_for_view_math()
        target_rect = self._image_target_rect()
        if image is None or target_rect is None:
            self.hover_changed.emit(-1.0, -1.0, False, "")
            return

        image_point = view_to_image_point(
            Point2D(pos.x(), pos.y()),
            image_rect_in_viewport(
                Size2D(float(max(image.width(), 1)), float(max(image.height(), 1))),
                Size2D(float(max(self.width(), 1)), float(max(self.height(), 1))),
                self._zoom,
                Point2D(self._pan_x, self._pan_y),
            ),
            Size2D(float(max(image.width(), 1)), float(max(image.height(), 1))),
        )
        if image_point is None:
            self.hover_changed.emit(-1.0, -1.0, False, "")
            return

        denom_x = max(float(image.width() - 1), 1.0)
        denom_y = max(float(image.height() - 1), 1.0)
        u = max(0.0, min(1.0, image_point.x / denom_x))
        v = max(0.0, min(1.0, image_point.y / denom_y))
        side = self._side_at_position(pos) or ""
        self.hover_changed.emit(u, v, True, side)
