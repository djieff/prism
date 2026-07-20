"""Scientific helpers for waveform signal standards and presentation data."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal

import numpy as np

WaveformSignalStandard = Literal["ITU-R BT.709", "ITU-R BT.2020"]
WaveformDensityChannels = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

DEFAULT_WAVEFORM_SIGNAL_STANDARD: WaveformSignalStandard = "ITU-R BT.709"
WAVEFORM_DENSITY_FILTER_SIGMA: tuple[float, float] = (0.5, 0.5)

SUPPORTED_WAVEFORM_SIGNAL_STANDARDS: tuple[WaveformSignalStandard, ...] = (
    DEFAULT_WAVEFORM_SIGNAL_STANDARD,
    "ITU-R BT.2020",
)


@lru_cache(maxsize=len(SUPPORTED_WAVEFORM_SIGNAL_STANDARDS))
def _cached_y_prime_coefficients(standard: WaveformSignalStandard) -> np.ndarray:
    """Return the cached read-only coefficient source for one signal standard."""
    if standard not in SUPPORTED_WAVEFORM_SIGNAL_STANDARDS:
        raise ValueError(f"Unsupported waveform signal standard: {standard!r}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message='"Matplotlib" related API features are not available.*',
        )
        from colour import WEIGHTS_YCBCR

    kr, kb = np.asarray(WEIGHTS_YCBCR[standard], dtype=np.float64)
    coefficients = np.asarray((kr, 1.0 - kr - kb, kb), dtype=np.float32)
    coefficients.setflags(write=False)
    return coefficients


def waveform_y_prime_coefficients(standard: WaveformSignalStandard) -> np.ndarray:
    """Return immutable float32 RGB coefficients for encoded Y' computation."""
    cached = _cached_y_prime_coefficients(standard)
    return np.frombuffer(cached.tobytes(), dtype=np.float32)


def filter_waveform_density(
    density: np.ndarray,
    *,
    sigma: tuple[float, float],
) -> np.ndarray:
    """Return a Gaussian-filtered float32 copy suitable for waveform rendering."""
    from scipy.ndimage import gaussian_filter

    source = np.asarray(density, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("Expected waveform density shape (H, W)")
    if not np.all(np.isfinite(source)):
        raise ValueError("Waveform density must contain only finite values")
    if np.any(source < 0.0):
        raise ValueError("Waveform density must be non-negative")

    sigma_values = np.asarray(sigma, dtype=np.float64)
    if sigma_values.shape != (2,):
        raise ValueError("sigma must contain exactly two values: (y, x)")
    if not np.all(np.isfinite(sigma_values)) or np.any(sigma_values < 0.0):
        raise ValueError("sigma values must be finite and non-negative")

    filtered = gaussian_filter(
        source,
        sigma=(float(sigma_values[0]), float(sigma_values[1])),
        mode="reflect",
        output=np.float32,
    )
    if not np.all(np.isfinite(filtered)) or np.any(filtered < 0.0):
        raise ValueError("Filtered waveform density must be finite and non-negative")
    return np.ascontiguousarray(filtered, dtype=np.float32)


def prepare_waveform_densities_for_render(
    densities: WaveformDensityChannels,
    *,
    sigma: tuple[float, float] = WAVEFORM_DENSITY_FILTER_SIGMA,
) -> WaveformDensityChannels:
    """Filter four waveform channels and normalize them with one shared factor."""
    expected_shape = np.asarray(densities[0]).shape
    if any(np.asarray(density).shape != expected_shape for density in densities[1:]):
        raise ValueError("Waveform density channels must have matching shapes")

    filtered: WaveformDensityChannels = (
        filter_waveform_density(densities[0], sigma=sigma),
        filter_waveform_density(densities[1], sigma=sigma),
        filter_waveform_density(densities[2], sigma=sigma),
        filter_waveform_density(densities[3], sigma=sigma),
    )
    maximum = max(float(np.max(density)) for density in filtered)
    if maximum <= 0.0:
        return filtered

    inverse = np.float32(1.0 / maximum)
    normalized: WaveformDensityChannels = (
        np.ascontiguousarray(filtered[0] * inverse, dtype=np.float32),
        np.ascontiguousarray(filtered[1] * inverse, dtype=np.float32),
        np.ascontiguousarray(filtered[2] * inverse, dtype=np.float32),
        np.ascontiguousarray(filtered[3] * inverse, dtype=np.float32),
    )
    return normalized
