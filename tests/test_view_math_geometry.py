"""Unit tests for per-source geometry mapping helpers."""

from __future__ import annotations

import unittest

from prism.ui.view_math import (
    Point2D,
    Size2D,
    apply_offset,
    canvas_to_source_point,
    resolve_canvas_size,
    source_rect_in_canvas,
)


class ViewMathGeometryTests(unittest.TestCase):
    def test_resolve_canvas_size_policies(self) -> None:
        source = Size2D(1920, 1080)
        size_a = Size2D(3840, 2160)
        size_b = Size2D(2048, 858)
        viewport = Size2D(1600, 900)

        self.assertEqual(resolve_canvas_size("native", source, size_a, size_b, viewport), source)
        self.assertEqual(resolve_canvas_size("match_a", source, size_a, size_b, viewport), size_a)
        self.assertEqual(resolve_canvas_size("match_b", source, size_a, size_b, viewport), size_b)
        self.assertEqual(
            resolve_canvas_size("viewport", source, size_a, size_b, viewport), viewport
        )

    def test_source_rect_fit_policy_letterboxes(self) -> None:
        source = Size2D(4000, 2000)
        canvas = Size2D(1000, 1000)

        rect = source_rect_in_canvas(source, canvas, "fit")

        self.assertAlmostEqual(rect.width, 1000.0)
        self.assertAlmostEqual(rect.height, 500.0)
        self.assertAlmostEqual(rect.x, 0.0)
        self.assertAlmostEqual(rect.y, 250.0)

    def test_source_rect_fill_policy_crops(self) -> None:
        source = Size2D(4000, 2000)
        canvas = Size2D(1000, 1000)

        rect = source_rect_in_canvas(source, canvas, "fill")

        self.assertAlmostEqual(rect.width, 2000.0)
        self.assertAlmostEqual(rect.height, 1000.0)
        self.assertAlmostEqual(rect.x, -500.0)
        self.assertAlmostEqual(rect.y, 0.0)

    def test_source_rect_stretch_policy_ignores_aspect(self) -> None:
        source = Size2D(4000, 2000)
        canvas = Size2D(1000, 1000)

        rect = source_rect_in_canvas(source, canvas, "stretch")

        self.assertAlmostEqual(rect.width, 1000.0)
        self.assertAlmostEqual(rect.height, 1000.0)
        self.assertAlmostEqual(rect.x, 0.0)
        self.assertAlmostEqual(rect.y, 0.0)

    def test_source_rect_one_to_one_policy_keeps_native_pixels(self) -> None:
        source = Size2D(4000, 2000)
        canvas = Size2D(1000, 1000)

        rect = source_rect_in_canvas(source, canvas, "one_to_one")

        self.assertAlmostEqual(rect.width, 4000.0)
        self.assertAlmostEqual(rect.height, 2000.0)
        self.assertAlmostEqual(rect.x, -1500.0)
        self.assertAlmostEqual(rect.y, -500.0)

    def test_apply_offset_translates_rect(self) -> None:
        moved = apply_offset(
            source_rect_in_canvas(Size2D(100, 50), Size2D(200, 100), "fit"),
            Point2D(12, -8),
        )
        self.assertAlmostEqual(moved.x, 12.0)
        self.assertAlmostEqual(moved.y, -8.0)

    def test_canvas_to_source_point_maps_center(self) -> None:
        source = Size2D(400, 200)
        rect = source_rect_in_canvas(source, Size2D(1000, 1000), "fit")

        mapped = canvas_to_source_point(Point2D(500, 500), rect, source)

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertAlmostEqual(mapped.x, 199.5, places=1)
        self.assertAlmostEqual(mapped.y, 99.5, places=1)

    def test_canvas_to_source_point_returns_none_out_of_bounds(self) -> None:
        source = Size2D(400, 200)
        rect = source_rect_in_canvas(source, Size2D(1000, 1000), "fit")

        mapped = canvas_to_source_point(Point2D(500, 50), rect, source)

        self.assertIsNone(mapped)


if __name__ == "__main__":
    unittest.main()

