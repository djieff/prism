"""Logic tests for LUT volume widget projection state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prism.core.lut_volume_projection import LutVolumeProjection
from prism.io.lut_loader import LutVolumeData
from prism.ui import lut_volume_widget as volume_widget_module
from prism.ui.lut_volume_widget import LutVolumeWidget


def _volume_data() -> LutVolumeData:
    values = np.zeros((2, 2, 2, 3), dtype=np.float32)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                values[z, y, x] = [x, y, z]
    return LutVolumeData(
        path=Path("identity.cube"),
        format="cube",
        size_x=2,
        size_y=2,
        size_z=2,
        values=values,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def _widget_stub() -> LutVolumeWidget:
    widget = LutVolumeWidget.__new__(LutVolumeWidget)
    widget._volume_data = None
    widget._projection = None
    widget._projection_mode = "RGB isometric"
    widget._sample_limit = 50_000
    widget._use_output_positions = True
    widget._error_text = None
    return widget


def test_rebuild_projection_uses_current_widget_settings(monkeypatch) -> None:
    projection = LutVolumeProjection(
        xy=np.zeros((1, 2), dtype=np.float32),
        colors_rgb=np.zeros((1, 3), dtype=np.float32),
        sample_indices=np.asarray([0], dtype=np.int64),
        total_point_count=8,
        projected_point_count=1,
        mode="RB plane",
    )
    calls: list[tuple[np.ndarray, str, int, bool]] = []

    def _project(values, *, mode, sample_limit, use_output_positions):
        calls.append((values, mode, sample_limit, use_output_positions))
        return projection

    monkeypatch.setattr(volume_widget_module, "project_lut_volume", _project)
    widget = _widget_stub()
    widget._volume_data = _volume_data()
    widget._projection_mode = "RB plane"
    widget._sample_limit = 12
    widget._use_output_positions = False

    widget._rebuild_projection()

    assert widget._projection is projection
    assert widget._error_text is None
    assert len(calls) == 1
    assert calls[0][0] is widget._volume_data.values
    assert calls[0][1:] == ("RB plane", 12, False)


def test_rebuild_projection_records_error_text(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise ValueError("bad volume")

    monkeypatch.setattr(volume_widget_module, "project_lut_volume", _raise)
    widget = _widget_stub()
    widget._volume_data = _volume_data()

    widget._rebuild_projection()

    assert widget._projection is None
    assert widget.status_text() == "Volume unavailable: bad volume"


def test_status_text_reports_empty_and_projected_states() -> None:
    widget = _widget_stub()

    assert widget.status_text() == "Volume view requires a 3D LUT."

    widget._volume_data = _volume_data()
    widget._projection = LutVolumeProjection(
        xy=np.zeros((8, 2), dtype=np.float32),
        colors_rgb=np.zeros((8, 3), dtype=np.float32),
        sample_indices=np.arange(8, dtype=np.int64),
        total_point_count=8,
        projected_point_count=8,
        mode="RGB isometric",
    )

    assert widget.status_text() == "Volume: 2x2x2 | shown: 8/8 | mode: RGB isometric"


def test_set_sample_limit_rejects_non_positive_values() -> None:
    widget = _widget_stub()

    with pytest.raises(ValueError, match="sample_limit"):
        widget.set_sample_limit(0)


def test_plot_rect_is_centered_square(monkeypatch) -> None:
    widget = _widget_stub()
    monkeypatch.setattr(widget, "width", lambda: 1000)
    monkeypatch.setattr(widget, "height", lambda: 500)

    rect = widget._plot_rect()

    assert rect.width() == rect.height()
    assert rect.width() == 436.0
    assert rect.left() == 297.0
    assert rect.top() == 30.0
