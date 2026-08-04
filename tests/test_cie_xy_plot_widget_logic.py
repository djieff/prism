"""Logic tests for CIE xy plot presentation preparation."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QRect, QRectF

from prism.core.scopes.cie_xy import (
    CieGamutOverlay,
    CieWhitePoint,
    CieXyTrace,
    get_cie_rgb_space,
    get_cie_whitepoint,
    spectral_locus_xy,
)
from prism.ui.scopes.cie_xy_plot_widget import (
    OVERLAY_LEGEND_MARGIN,
    OVERLAY_LEGEND_ROW_GAP,
    OVERLAY_LEGEND_ROW_HEIGHT,
    OVERLAY_LEGEND_WIDTH,
    CieXyPlotWidget,
    boosted_trace_rgb,
    closed_spectral_locus_polygon,
    xy_in_polygon,
)


def _trace() -> CieXyTrace:
    density = np.zeros((5, 5), dtype=np.float32)
    density[2, 2] = 1.0
    color_density = np.zeros((5, 5, 3), dtype=np.float32)
    color_density[2, 2, :] = (1.0, 0.0, 0.0)
    return CieXyTrace(
        x_values=np.asarray([0.3127], dtype=np.float32),
        y_values=np.asarray([0.3290], dtype=np.float32),
        density=density,
        color_density=color_density,
        source_size=(8, 6),
        rgb_space_name="sRGB",
    )


def _muted_trace() -> CieXyTrace:
    density = np.zeros((5, 5), dtype=np.float32)
    density[2, 2] = 1.0
    color_density = np.zeros((5, 5, 3), dtype=np.float32)
    color_density[2, 2, :] = (0.65, 0.35, 0.30)
    return CieXyTrace(
        x_values=np.asarray([0.3127], dtype=np.float32),
        y_values=np.asarray([0.3290], dtype=np.float32),
        density=density,
        color_density=color_density,
        source_size=(8, 6),
        rgb_space_name="sRGB",
    )


def test_rebuild_density_image_creates_cached_qimage_and_buffer() -> None:
    trace = _trace()
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._density_image = None
    widget._density_image_buffer = None
    widget._trace_color_mode = "boosted"

    widget._rebuild_density_image(trace)

    assert widget._density_image is not None
    assert widget._density_image.width() == 5
    assert widget._density_image.height() == 5
    assert widget._density_image_buffer is not None
    assert widget._density_image_buffer.shape == (5, 5, 4)
    assert widget._density_image_buffer.dtype == np.uint8
    assert widget._density_image_buffer.flags.c_contiguous
    assert widget._density_image_buffer[2, 2, 0] > 245
    assert widget._density_image_buffer[2, 2, 1] < 5
    assert widget._density_image_buffer[2, 2, 2] < 5
    assert widget._density_image_buffer[2, 2, 3] > 245
    assert widget._density_image_buffer[0, 0, 3] <= 75


def test_boosted_trace_rgb_increases_saturation_without_tinting_gray() -> None:
    rgb = np.asarray([[(0.65, 0.35, 0.30), (0.5, 0.5, 0.5)]], dtype=np.float32)

    boosted = boosted_trace_rgb(rgb)

    original_chroma = float(np.max(rgb[0, 0]) - np.min(rgb[0, 0]))
    boosted_chroma = float(np.max(boosted[0, 0]) - np.min(boosted[0, 0]))
    assert boosted_chroma > original_chroma
    assert boosted[0, 0, 0] > rgb[0, 0, 0]
    assert boosted[0, 1, 0] == pytest.approx(boosted[0, 1, 1])
    assert boosted[0, 1, 1] == pytest.approx(boosted[0, 1, 2])


@pytest.mark.parametrize(
    ("density", "message"),
    [
        (np.zeros((2, 2, 1), dtype=np.float32), "shape"),
        (np.asarray([[0.0, np.nan]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, np.inf]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, -0.1]], dtype=np.float32), "non-negative"),
    ],
)
def test_rebuild_density_image_rejects_invalid_density(
    density: np.ndarray,
    message: str,
) -> None:
    trace = CieXyTrace(
        x_values=np.asarray([0.0], dtype=np.float32),
        y_values=np.asarray([0.0], dtype=np.float32),
        density=density,
        color_density=np.zeros((*density.shape[:2], 3), dtype=np.float32),
        source_size=(1, 1),
        rgb_space_name="sRGB",
    )
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._trace_color_mode = "boosted"

    with pytest.raises(ValueError, match=message):
        widget._rebuild_density_image(trace)


@pytest.mark.parametrize(
    ("color_density", "message"),
    [
        (np.zeros((2, 2), dtype=np.float32), "color density shape"),
        (np.full((2, 2, 3), np.nan, dtype=np.float32), "color density must contain only finite"),
        (np.full((2, 2, 3), -0.1, dtype=np.float32), "color density must be non-negative"),
    ],
)
def test_rebuild_density_image_rejects_invalid_color_density(
    color_density: np.ndarray,
    message: str,
) -> None:
    density = np.ones((2, 2), dtype=np.float32)
    trace = CieXyTrace(
        x_values=np.asarray([0.0], dtype=np.float32),
        y_values=np.asarray([0.0], dtype=np.float32),
        density=density,
        color_density=color_density,
        source_size=(1, 1),
        rgb_space_name="sRGB",
    )
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._trace_color_mode = "boosted"

    with pytest.raises(ValueError, match=message):
        widget._rebuild_density_image(trace)


def test_rebuild_density_image_normal_trace_color_preserves_source_color() -> None:
    trace = _muted_trace()
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._density_image = None
    widget._density_image_buffer = None
    widget._trace_color_mode = "normal"

    widget._rebuild_density_image(trace)

    assert widget._density_image_buffer[2, 2, 0] == pytest.approx(166, abs=1)
    assert widget._density_image_buffer[2, 2, 1] == pytest.approx(89, abs=1)
    assert widget._density_image_buffer[2, 2, 2] == pytest.approx(76, abs=1)


def test_trace_color_mode_rebuilds_existing_trace_density_image() -> None:
    trace = _muted_trace()
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._trace = trace
    widget._density_image = object()
    widget._density_image_buffer = None
    widget._trace_color_mode = "normal"
    updates: list[bool] = []
    widget.update = lambda: updates.append(True)

    widget.set_trace_color_mode("boosted")
    first_rgb = tuple(widget._density_image_buffer[2, 2, 0:3])
    widget.set_trace_color_mode("boosted")

    assert widget._trace_color_mode == "boosted"
    assert first_rgb[0] > 166
    assert updates == [True]


def test_trace_color_mode_rejects_unknown_mode() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._trace_color_mode = "boosted"

    with pytest.raises(ValueError, match="Unsupported CIE xy trace color mode"):
        widget.set_trace_color_mode("false-color")  # type: ignore[arg-type]


def test_set_gamut_overlay_names_de_duplicates_and_updates() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._overlay_names = ()
    widget.update = lambda: None

    widget.set_gamut_overlay_names(("sRGB", None, "Display P3", "sRGB"))

    assert widget._overlay_names == ("sRGB", "Display P3")


def test_set_gamut_overlay_names_rejects_unknown_overlay() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._overlay_names = ()

    with pytest.raises(ValueError, match="Unsupported CIE gamut overlay"):
        widget.set_gamut_overlay_names(("Not A Space",))


def test_set_whitepoint_updates_only_when_changed() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._whitepoint = get_cie_whitepoint("D65")
    updates: list[bool] = []
    widget.update = lambda: updates.append(True)
    d60 = get_cie_whitepoint("D60")

    widget.set_whitepoint(d60)
    widget.set_whitepoint(d60)

    assert widget._whitepoint == d60
    assert updates == [True]


def test_set_whitepoint_accepts_none_to_hide_marker() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._whitepoint = get_cie_whitepoint("D65")
    updates: list[bool] = []
    widget.update = lambda: updates.append(True)

    widget.set_whitepoint(None)

    assert widget._whitepoint is None
    assert updates == [True]


def test_selected_overlays_return_colourspace_overlay_contracts() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._overlay_names = ("sRGB", "ITU-R BT.2020")

    overlays = widget._selected_overlays()

    assert tuple(overlay.name for overlay in overlays) == ("sRGB", "ITU-R BT.2020")
    assert overlays[0].primaries_xy == get_cie_rgb_space("sRGB").primaries_xy
    assert overlays[1].primaries_xy == get_cie_rgb_space("ITU-R BT.2020").primaries_xy


def test_overlay_color_cycles_through_three_distinct_colours() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)

    colors = [widget._overlay_color(index).getRgb()[:3] for index in range(4)]

    assert colors[:3] == [(255, 88, 72), (60, 220, 92), (96, 148, 255)]
    assert colors[3] == colors[0]


def test_overlay_label_text_shortens_long_reference_names() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)

    assert widget._overlay_label_text("ITU-R BT.709") == "BT.709"
    assert widget._overlay_label_text("ITU-R BT.2020") == "BT.2020"
    assert widget._overlay_label_text("Adobe RGB (1998)") == "Adobe RGB"
    assert widget._overlay_label_text("Display P3") == "Display P3"


def test_overlay_label_uses_top_right_legend_with_index_offset() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    plot_rect = QRectF(0.0, 0.0, 220.0, 220.0)
    rgb_space = get_cie_rgb_space("sRGB")
    gamut = CieGamutOverlay(
        name="ITU-R BT.709",
        primaries_xy=rgb_space.primaries_xy,
        whitepoint_xy=rgb_space.whitepoint_xy,
    )

    first = widget._overlay_label(plot_rect, gamut, 0)
    second = widget._overlay_label(plot_rect, gamut, 1)

    assert first.text == "BT.709"
    assert first.point.x == pytest.approx(
        plot_rect.right() - OVERLAY_LEGEND_MARGIN - OVERLAY_LEGEND_WIDTH
    )
    assert first.point.y == pytest.approx(plot_rect.top() + OVERLAY_LEGEND_MARGIN)
    assert first.color.getRgb()[:3] == (255, 88, 72)
    assert second.point.x == pytest.approx(first.point.x)
    assert second.point.y == pytest.approx(
        first.point.y + OVERLAY_LEGEND_ROW_HEIGHT + OVERLAY_LEGEND_ROW_GAP
    )
    assert second.color.getRgb()[:3] == (60, 220, 92)


def test_whitepoint_label_appears_below_gamut_legend_entries() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._overlay_names = ("sRGB", "Display P3")
    plot_rect = QRectF(0.0, 0.0, 220.0, 220.0)
    whitepoint = CieWhitePoint(name="D60", xy=(0.32161671, 0.33761992))

    label = widget._whitepoint_label(plot_rect, whitepoint)

    assert label.text == "White: D60"
    assert label.point.x == pytest.approx(
        plot_rect.right() - OVERLAY_LEGEND_MARGIN - OVERLAY_LEGEND_WIDTH
    )
    assert label.point.y == pytest.approx(
        plot_rect.top()
        + OVERLAY_LEGEND_MARGIN
        + (2.0 * (OVERLAY_LEGEND_ROW_HEIGHT + OVERLAY_LEGEND_ROW_GAP))
    )
    assert label.color.getRgb()[:3] == (230, 230, 210)


def test_plot_rect_is_centered_square_in_wide_bounds() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)

    plot_rect = widget._plot_rect(QRect(0, 0, 400, 300))

    assert isinstance(plot_rect, QRectF)
    assert plot_rect.width() == pytest.approx(248.0)
    assert plot_rect.height() == pytest.approx(248.0)
    assert plot_rect.left() == pytest.approx(89.0)
    assert plot_rect.top() == pytest.approx(18.0)


def test_plot_rect_is_centered_square_in_tall_bounds() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)

    plot_rect = widget._plot_rect(QRect(0, 0, 300, 500))

    assert isinstance(plot_rect, QRectF)
    assert plot_rect.width() == pytest.approx(242.0)
    assert plot_rect.height() == pytest.approx(242.0)
    assert plot_rect.left() == pytest.approx(42.0)
    assert plot_rect.top() == pytest.approx(121.0)


def test_plot_point_maps_cie_bounds_to_plot_rect() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    plot_rect = QRectF(10.0, 20.0, 80.0, 80.0)

    bottom_left = widget._plot_point(plot_rect, 0.0, 0.0)
    top_right = widget._plot_point(plot_rect, 0.8, 0.9)
    d65 = widget._plot_point(plot_rect, 0.3127, 0.3290)

    assert bottom_left.x == pytest.approx(10.0)
    assert bottom_left.y == pytest.approx(100.0)
    assert top_right.x == pytest.approx(90.0)
    assert top_right.y == pytest.approx(20.0)
    assert d65.x == pytest.approx(41.27)
    assert d65.y == pytest.approx(70.7555555556)


def test_overlay_polygon_closes_triangle_in_plot_coordinates() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    plot_rect = QRectF(0.0, 0.0, 80.0, 80.0)
    rgb_space = get_cie_rgb_space("sRGB")
    gamut = CieGamutOverlay(
        name=rgb_space.name,
        primaries_xy=rgb_space.primaries_xy,
        whitepoint_xy=rgb_space.whitepoint_xy,
    )

    points = widget._overlay_polygon(plot_rect, gamut)

    assert len(points) == 4
    assert points[0] == points[-1]
    assert points[0].x == pytest.approx(64.0)
    assert points[0].y == pytest.approx(50.6666666667)


def test_spectral_locus_is_available_for_empty_graph() -> None:
    widget = CieXyPlotWidget.__new__(CieXyPlotWidget)
    widget._spectral_locus = np.asarray(
        [
            (0.17, 0.01),
            (0.20, 0.70),
            (0.73, 0.27),
        ],
        dtype=np.float32,
    )

    assert widget._spectral_locus.shape == (3, 2)
    assert widget._spectral_locus.dtype == np.float32


def test_closed_spectral_locus_polygon_adds_line_of_purples_closure() -> None:
    locus = np.asarray(
        [
            (0.17, 0.01),
            (0.20, 0.70),
            (0.73, 0.27),
        ],
        dtype=np.float32,
    )

    polygon = closed_spectral_locus_polygon(locus)

    assert polygon.shape == (4, 2)
    assert polygon.dtype == np.float32
    assert polygon.flags.c_contiguous
    assert polygon[0] == pytest.approx(locus[0])
    assert polygon[-2] == pytest.approx(locus[-1])
    assert polygon[-1] == pytest.approx(locus[0])


def test_closed_spectral_locus_polygon_preserves_already_closed_locus() -> None:
    locus = np.asarray(
        [
            (0.17, 0.01),
            (0.20, 0.70),
            (0.73, 0.27),
            (0.17, 0.01),
        ],
        dtype=np.float32,
    )

    polygon = closed_spectral_locus_polygon(locus)

    assert polygon.shape == locus.shape
    assert np.array_equal(polygon, locus)


@pytest.mark.parametrize(
    ("locus", "message"),
    [
        (np.zeros((3,), dtype=np.float32), "shape"),
        (np.zeros((2, 2), dtype=np.float32), "at least three"),
        (np.asarray([(0.0, 0.0), (0.1, np.nan), (0.2, 0.2)], dtype=np.float32), "finite"),
    ],
)
def test_closed_spectral_locus_polygon_rejects_invalid_locus(
    locus: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        closed_spectral_locus_polygon(locus)


def test_horseshoe_polygon_contains_d65_and_rejects_obvious_outside_points() -> None:
    polygon = closed_spectral_locus_polygon(spectral_locus_xy())

    assert xy_in_polygon(0.3127, 0.3290, polygon) is True
    assert xy_in_polygon(0.76, 0.86, polygon) is False
    assert xy_in_polygon(0.02, 0.02, polygon) is False
