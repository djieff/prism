"""Tests for waveform scientific helper contracts."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from prism.core.scope_waveform_science import (
    WAVEFORM_DENSITY_FILTER_SIGMA,
    WaveformSignalStandard,
    filter_waveform_density,
    prepare_waveform_densities_for_render,
    waveform_y_prime_coefficients,
)


@pytest.mark.parametrize(
    ("standard", "expected"),
    [
        ("ITU-R BT.709", (0.2126, 0.7152, 0.0722)),
        ("ITU-R BT.2020", (0.2627, 0.6780, 0.0593)),
    ],
)
def test_waveform_y_prime_coefficients_match_encoded_signal_weights(
    standard: WaveformSignalStandard,
    expected: tuple[float, float, float],
) -> None:
    coefficients = waveform_y_prime_coefficients(standard)

    assert coefficients.shape == (3,)
    assert coefficients.dtype == np.float32
    assert np.array_equal(coefficients, np.asarray(expected, dtype=np.float32))


def test_waveform_y_prime_coefficients_are_defensive_and_immutable() -> None:
    first = waveform_y_prime_coefficients("ITU-R BT.709")
    second = waveform_y_prime_coefficients("ITU-R BT.709")

    assert first is not second
    assert not np.shares_memory(first, second)
    assert first.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        first[0] = np.float32(0.0)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.setflags(write=True)


def test_waveform_y_prime_coefficients_reject_unsupported_standard() -> None:
    unsupported = cast(WaveformSignalStandard, "sRGB")
    with pytest.raises(ValueError, match="Unsupported waveform signal standard"):
        waveform_y_prime_coefficients(unsupported)


def test_filter_waveform_density_returns_float32_copy_without_mutating_input() -> None:
    density = np.zeros((5, 7), dtype=np.float64)
    density[2, 3] = 1.0
    original = density.copy()

    filtered = filter_waveform_density(density, sigma=(0.5, 0.5))

    assert filtered.shape == density.shape
    assert filtered.dtype == np.float32
    assert filtered.flags.c_contiguous
    assert not np.shares_memory(filtered, density)
    assert np.array_equal(density, original)
    assert np.all(np.isfinite(filtered))
    assert np.all(filtered >= 0.0)


@pytest.mark.parametrize(
    ("density", "message"),
    [
        (np.zeros((2, 2, 1), dtype=np.float32), "shape"),
        (np.asarray([[0.0, np.nan]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, np.inf]], dtype=np.float32), "finite"),
        (np.asarray([[0.0, -0.1]], dtype=np.float32), "non-negative"),
    ],
)
def test_filter_waveform_density_rejects_invalid_density(
    density: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        filter_waveform_density(density, sigma=(0.5, 0.5))


@pytest.mark.parametrize(
    "sigma",
    [
        (0.5,),
        (0.5, 0.5, 0.5),
        (-0.1, 0.5),
        (np.nan, 0.5),
        (np.inf, 0.5),
    ],
)
def test_filter_waveform_density_rejects_invalid_sigma(sigma: tuple[float, ...]) -> None:
    density = np.ones((2, 2), dtype=np.float32)
    invalid_sigma = cast(tuple[float, float], sigma)

    with pytest.raises(ValueError, match="sigma"):
        filter_waveform_density(density, sigma=invalid_sigma)


def test_filter_waveform_density_preserves_level_position_and_impulse_symmetry() -> None:
    density = np.zeros((9, 9), dtype=np.float32)
    density[4, 4] = 1.0

    filtered = filter_waveform_density(
        density,
        sigma=WAVEFORM_DENSITY_FILTER_SIGMA,
    )

    assert np.unravel_index(np.argmax(filtered), filtered.shape) == (4, 4)
    assert filtered[3, 4] == pytest.approx(filtered[5, 4])
    assert filtered[4, 3] == pytest.approx(filtered[4, 5])
    assert float(np.sum(filtered)) == pytest.approx(1.0, abs=1e-6)


def test_prepare_waveform_densities_uses_shared_normalization_and_preserves_raw() -> None:
    amplitudes = (1.0, 0.5, 0.25, 0.75)
    raw_channels = []
    for amplitude in amplitudes:
        density = np.zeros((9, 9), dtype=np.float32)
        density[4, 4] = amplitude
        raw_channels.append(density)
    raw = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple(raw_channels),
    )
    originals = tuple(density.copy() for density in raw)

    prepared = prepare_waveform_densities_for_render(raw)

    peaks = tuple(float(np.max(density)) for density in prepared)
    assert peaks == pytest.approx(amplitudes)
    for source, original, result in zip(raw, originals, prepared, strict=True):
        assert np.array_equal(source, original)
        assert not np.shares_memory(source, result)
        assert result.dtype == np.float32
        assert result.flags.c_contiguous


def test_prepare_waveform_densities_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    raw = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple(rng.random((8, 12), dtype=np.float32) for _ in range(4)),
    )

    first = prepare_waveform_densities_for_render(raw)
    second = prepare_waveform_densities_for_render(raw)

    for first_channel, second_channel in zip(first, second, strict=True):
        assert np.array_equal(first_channel, second_channel)


def test_prepare_waveform_densities_rejects_mismatched_shapes() -> None:
    raw = (
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="matching shapes"):
        prepare_waveform_densities_for_render(raw)
