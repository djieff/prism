"""Logic tests for vectorscope plot presentation preparation."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QRect, QRectF

from prism.core.scopes.vectorscope import VectorscopeTrace, vectorscope_chroma_scale
from prism.ui.scopes.vectorscope_plot_widget import VectorscopePlotWidget


def _trace() -> VectorscopeTrace:
    density = np.zeros((5, 5), dtype=np.float32)
    density[2, 2] = 1.0
    color_density = np.zeros((5, 5, 3), dtype=np.float32)
    color_density[2, 2, :] = (1.0, 0.0, 0.0)
    coefficients = (0.2126, 0.7152, 0.0722)
    return VectorscopeTrace(
        x_values=np.asarray([0.0], dtype=np.float32),
        y_values=np.asarray([0.0], dtype=np.float32),
        density=density,
        color_density=color_density,
        source_size=(8, 6),
        signal_standard="ITU-R BT.709",
        y_prime_coefficients=coefficients,
        plot_scale=vectorscope_chroma_scale(coefficients),
    )


def test_rebuild_heatmap_creates_cached_qimage_and_buffer() -> None:
    trace = _trace()
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)
    widget._color_mode = "Teal"
    widget._heatmap = None
    widget._heatmap_buffer = None

    widget._rebuild_heatmap(trace)

    assert widget._heatmap is not None
    assert widget._heatmap.width() == 5
    assert widget._heatmap.height() == 5
    assert widget._heatmap_buffer is not None
    assert widget._heatmap_buffer.shape == (5, 5, 3)
    assert widget._heatmap_buffer.dtype == np.uint8
    assert widget._heatmap_buffer.flags.c_contiguous
    assert widget._heatmap_buffer[2, 2, 1] > widget._heatmap_buffer[2, 2, 0]
    assert widget._heatmap_buffer[2, 2, 1] > widget._heatmap_buffer[0, 0, 1]


@pytest.mark.parametrize(
    ("density", "message"),
    [
        (np.zeros((2, 2, 1), dtype=np.float32), "shape"),
        (np.asarray([[0.0, np.nan]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, np.inf]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, -0.1]], dtype=np.float32), "non-negative"),
    ],
)
def test_rebuild_heatmap_rejects_invalid_density(
    density: np.ndarray,
    message: str,
) -> None:
    trace = VectorscopeTrace(
        x_values=np.asarray([0.0], dtype=np.float32),
        y_values=np.asarray([0.0], dtype=np.float32),
        density=density,
        color_density=np.zeros((*density.shape[:2], 3), dtype=np.float32),
        source_size=(1, 1),
        signal_standard="ITU-R BT.709",
        y_prime_coefficients=(0.2126, 0.7152, 0.0722),
        plot_scale=vectorscope_chroma_scale((0.2126, 0.7152, 0.0722)),
    )
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)

    with pytest.raises(ValueError, match=message):
        widget._rebuild_heatmap(trace)


def test_targets_for_trace_use_trace_signal_coefficients() -> None:
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)

    bt709 = widget._targets_for_trace(_trace())
    bt2020 = widget._targets_for_trace(
        VectorscopeTrace(
            x_values=np.asarray([0.0], dtype=np.float32),
            y_values=np.asarray([0.0], dtype=np.float32),
            density=np.zeros((5, 5), dtype=np.float32),
            color_density=np.zeros((5, 5, 3), dtype=np.float32),
            source_size=(1, 1),
            signal_standard="ITU-R BT.2020",
            y_prime_coefficients=(0.2627, 0.6780, 0.0593),
            plot_scale=1.1713260724235575,
        )
    )

    assert tuple(target.label for target in bt709) == ("R", "Y", "G", "Cy", "B", "Mg")
    assert bt709[0].x == pytest.approx(-0.19234516, abs=1e-6)
    assert bt709[0].y == pytest.approx(0.83940664, abs=1e-6)
    assert bt709[4].x == pytest.approx(0.83940664, abs=1e-6)
    assert bt709[4].y == pytest.approx(-0.0769687, abs=1e-6)
    assert all(np.hypot(target.x, target.y) <= 1.0 for target in bt709)
    assert bt2020[0].x != pytest.approx(bt709[0].x)
    assert bt2020[2].y != pytest.approx(bt709[2].y)


def test_targets_for_current_standard_are_available_without_trace() -> None:
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)
    widget._signal_standard = "ITU-R BT.709"
    widget._target_coefficients = (0.2126, 0.7152, 0.0722)
    widget._target_plot_scale = vectorscope_chroma_scale(widget._target_coefficients)

    targets = widget._targets_for_current_standard()

    assert tuple(target.label for target in targets) == ("R", "Y", "G", "Cy", "B", "Mg")
    assert all(np.hypot(target.x, target.y) <= 1.0 for target in targets)


def test_set_signal_standard_updates_empty_graticule_targets() -> None:
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)
    widget._signal_standard = "ITU-R BT.709"
    widget._target_coefficients = (0.2126, 0.7152, 0.0722)
    widget._target_plot_scale = vectorscope_chroma_scale(widget._target_coefficients)
    widget.update = lambda: None

    before = widget._targets_for_current_standard()
    widget.set_signal_standard("ITU-R BT.2020")
    after = widget._targets_for_current_standard()

    assert widget._signal_standard == "ITU-R BT.2020"
    assert widget._target_coefficients == pytest.approx((0.2627, 0.6780, 0.0593))
    assert before[0].x != pytest.approx(after[0].x)


def test_set_color_mode_rebuilds_heatmap_from_source_color() -> None:
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)
    widget._trace = _trace()
    widget._color_mode = "Teal"
    widget._heatmap = None
    widget._heatmap_buffer = None
    widget.update = lambda: None

    widget._rebuild_heatmap(widget._trace)
    teal = widget._heatmap_buffer.copy()

    widget.set_color_mode("Source Color")

    assert widget._color_mode == "Source Color"
    assert widget._heatmap_buffer is not None
    assert widget._heatmap_buffer[2, 2, 0] > widget._heatmap_buffer[2, 2, 1]
    assert not np.array_equal(teal, widget._heatmap_buffer)


def test_scope_rect_is_centered_square_with_margin() -> None:
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)

    scope_rect = widget._scope_rect(QRect(0, 0, 400, 300))

    assert isinstance(scope_rect, QRectF)
    assert scope_rect.width() == pytest.approx(252.0)
    assert scope_rect.height() == pytest.approx(252.0)
    assert scope_rect.left() == pytest.approx(74.0)
    assert scope_rect.top() == pytest.approx(24.0)


def test_scope_point_maps_center_and_edges() -> None:
    widget = VectorscopePlotWidget.__new__(VectorscopePlotWidget)
    scope_rect = QRectF(10.0, 20.0, 100.0, 100.0)

    center = widget._scope_point(scope_rect, 0.0, 0.0)
    right_top = widget._scope_point(scope_rect, 1.0, 1.0)

    assert center.x() == pytest.approx(60.0)
    assert center.y() == pytest.approx(70.0)
    assert right_top.x() == pytest.approx(110.0)
    assert right_top.y() == pytest.approx(20.0)
