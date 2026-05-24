"""Targeted logic tests for CompareView non-visual behavior."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage

from prism.ui.compare_view import CompareView


def _make_compare_view() -> CompareView:
    view = CompareView.__new__(CompareView)
    view._mode = "full"
    view._active_side = "left"
    view._wipe_top_side = "left"
    view._wipe_divider_hit_px = 10.0
    view._image_a = None
    view._image_b = None
    view._image_diff = None
    view._wipe_position = 0.5
    view._wipe_angle = 90.0
    view._wipe_pivot_offset = 0.0
    return view


def test_side_for_position_full_mode_returns_active_side() -> None:
    view = _make_compare_view()
    view._mode = "full"
    view._active_side = "right"
    assert view.side_for_position(QPointF(10.0, 10.0)) == "right"


def test_side_for_position_non_wipe_non_full_returns_none() -> None:
    view = _make_compare_view()
    view._mode = "diff"
    assert view.side_for_position(QPointF(5.0, 5.0)) is None


def test_side_for_position_wipe_outside_compare_rect_returns_none() -> None:
    view = _make_compare_view()
    view._mode = "wipe"
    view._compare_space_rect = lambda: QRectF(0.0, 0.0, 100.0, 100.0)
    assert view.side_for_position(QPointF(200.0, 200.0)) is None


def test_side_for_position_wipe_respects_wipe_top_side_left() -> None:
    view = _make_compare_view()
    view._mode = "wipe"
    view._wipe_top_side = "left"
    view._compare_space_rect = lambda: QRectF(0.0, 0.0, 100.0, 100.0)
    view._wipe_pivot_center = lambda _rect: QPointF(50.0, 50.0)
    view._wipe_direction = lambda: QPointF(1.0, 0.0)

    # With horizontal divider direction, positive side is y > center.
    assert view.side_for_position(QPointF(50.0, 80.0)) == "left"
    assert view.side_for_position(QPointF(50.0, 20.0)) == "right"


def test_side_for_position_wipe_respects_wipe_top_side_right() -> None:
    view = _make_compare_view()
    view._mode = "wipe"
    view._wipe_top_side = "right"
    view._compare_space_rect = lambda: QRectF(0.0, 0.0, 100.0, 100.0)
    view._wipe_pivot_center = lambda _rect: QPointF(50.0, 50.0)
    view._wipe_direction = lambda: QPointF(1.0, 0.0)

    assert view.side_for_position(QPointF(50.0, 80.0)) == "right"
    assert view.side_for_position(QPointF(50.0, 20.0)) == "left"


def test_reference_image_for_view_math_mode_selection() -> None:
    view = _make_compare_view()
    image_a = QImage(2, 2, QImage.Format.Format_RGB32)
    image_b = QImage(3, 3, QImage.Format.Format_RGB32)
    image_diff = QImage(4, 4, QImage.Format.Format_RGB32)
    view._image_a = image_a
    view._image_b = image_b
    view._image_diff = image_diff

    view._mode = "diff"
    assert view._reference_image_for_view_math() is image_diff

    view._mode = "full"
    view._active_side = "left"
    assert view._reference_image_for_view_math() is image_a
    view._active_side = "right"
    assert view._reference_image_for_view_math() is image_b

    view._mode = "wipe"
    assert view._reference_image_for_view_math() is image_a


def test_drop_target_rect_for_side_non_wipe_returns_compare_rect() -> None:
    view = _make_compare_view()
    view._mode = "full"
    compare_rect = QRectF(1.0, 2.0, 30.0, 40.0)
    assert view._drop_target_rect_for_side("left", compare_rect) == compare_rect


def test_drop_target_rect_for_side_wipe_uses_side_polygon_bounds() -> None:
    view = _make_compare_view()
    view._mode = "wipe"
    view._wipe_top_side = "left"
    view._wipe_center = lambda _rect: QPointF(50.0, 50.0)
    view._wipe_direction = lambda: QPointF(1.0, 0.0)

    positive_polygon = [QPointF(10.0, 10.0), QPointF(30.0, 10.0), QPointF(30.0, 30.0)]
    negative_polygon = [QPointF(60.0, 60.0), QPointF(90.0, 60.0), QPointF(90.0, 90.0)]
    view._wipe_side_polygons = lambda _rect, _center, _dir: (positive_polygon, negative_polygon)

    def _poly_bounds(poly):
        if poly is positive_polygon:
            return QRectF(10.0, 10.0, 20.0, 20.0)
        return QRectF(60.0, 60.0, 30.0, 30.0)

    view._polygon_bounds = _poly_bounds

    compare_rect = QRectF(0.0, 0.0, 100.0, 100.0)
    assert view._drop_target_rect_for_side("left", compare_rect) == QRectF(10.0, 10.0, 20.0, 20.0)
    assert view._drop_target_rect_for_side("right", compare_rect) == QRectF(60.0, 60.0, 30.0, 30.0)
