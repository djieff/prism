"""Logic tests for waveform plot presentation preparation."""

from __future__ import annotations

import numpy as np

from prism.core.scope_waveform import WaveformTrace
from prism.core.scope_waveform_science import WAVEFORM_DENSITY_FILTER_SIGMA
from prism.ui import waveform_plot_widget as plot_widget_module
from prism.ui.waveform_plot_widget import WaveformPlotWidget


def _trace() -> WaveformTrace:
    density = np.zeros((4, 6), dtype=np.float32)
    return WaveformTrace(
        x_values=np.linspace(0.0, 1.0, 6, dtype=np.float32),
        density_r=density.copy(),
        density_g=density.copy(),
        density_b=density.copy(),
        density_luma=density.copy(),
        source_size=(6, 4),
        signal_standard="ITU-R BT.709",
        y_prime_coefficients=(0.2126, 0.7152, 0.0722),
    )


def test_prepared_density_channels_filters_once_and_caches(monkeypatch) -> None:
    trace = _trace()
    prepared = tuple(
        np.full((4, 6), value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.4)
    )
    calls: list[tuple[tuple[np.ndarray, ...], tuple[float, float]]] = []

    def _prepare(densities, *, sigma):
        calls.append((densities, sigma))
        return prepared

    monkeypatch.setattr(plot_widget_module, "prepare_waveform_densities_for_render", _prepare)
    widget = WaveformPlotWidget.__new__(WaveformPlotWidget)
    widget._prepared_densities = None

    first = widget._prepared_density_channels(trace)
    second = widget._prepared_density_channels(trace)

    assert first is prepared
    assert second is prepared
    assert len(calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            calls[0][0],
            (
                trace.density_r,
                trace.density_g,
                trace.density_b,
                trace.density_luma,
            ),
            strict=True,
        )
    )
    assert calls[0][1] == WAVEFORM_DENSITY_FILTER_SIGMA
