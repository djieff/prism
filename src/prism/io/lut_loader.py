"""LUT parsing helpers for LUT inspection plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from prism.core.lut_interpolation import (
    evaluate_piecewise_linear,
    normalize_values_to_unit_from_points,
    sample_lut3d_trilinear,
)


class LutLoadError(ValueError):
    """Raised when a LUT file cannot be parsed for plotting."""


@dataclass(frozen=True)
class LutPlotData:
    """Structured plot-ready LUT data."""

    path: Path
    format: str
    source_kind: str
    channels: int
    x_values: np.ndarray
    y_values: np.ndarray
    domain_min: float
    domain_max: float
    csp_has_shaper: bool | None = None


@dataclass(frozen=True)
class LutVolumeData:
    """Structured full-volume 3D LUT data for volumetric inspection."""

    path: Path
    format: str
    size_x: int
    size_y: int
    size_z: int
    values: np.ndarray
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    has_shaper: bool | None = None


@dataclass(frozen=True)
class LutInspectionData:
    """Combined LUT data for curve and optional volume inspection views."""

    plot: LutPlotData
    volume: LutVolumeData | None


def load_lut_plot_data(path: Path) -> LutPlotData:
    """Load LUT file and return structured curve data for plotting."""
    suffix = path.suffix.lower()
    loader = _LUT_PLOT_LOADERS.get(suffix)
    if loader is not None:
        return loader(path)
    raise LutLoadError(f"Unsupported LUT format: {path.suffix or '<none>'}")


def load_lut_volume_data(path: Path) -> LutVolumeData | None:
    """Load LUT file and return full 3D volume data when available."""
    suffix = path.suffix.lower()
    loader = _LUT_VOLUME_LOADERS.get(suffix)
    if loader is not None:
        return loader(path)
    raise LutLoadError(f"Unsupported LUT format: {path.suffix or '<none>'}")


def load_lut_inspection_data(path: Path) -> LutInspectionData:
    """Load LUT file and return curve data plus optional 3D volume data."""
    return LutInspectionData(
        plot=load_lut_plot_data(path),
        volume=load_lut_volume_data(path),
    )


def _load_spi1d(path: Path) -> LutPlotData:
    """Parse a `.spi1d` LUT into plot-ready 1D curve data."""
    lines = path.read_text(encoding="utf-8").splitlines()

    length: int | None = None
    components = 1
    domain_min = 0.0
    domain_max = 1.0
    values: list[list[float]] = []
    in_values_block = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            in_values_block = True
            continue
        if "}" in line:
            in_values_block = False
            continue

        if in_values_block:
            row = [float(token) for token in line.split()]
            values.append(row)
            continue

        tokens = line.split()
        key = tokens[0]
        if key == "Length" and len(tokens) >= 2:
            length = int(tokens[1])
        elif key == "Components" and len(tokens) >= 2:
            components = int(tokens[1])
        elif key == "From" and len(tokens) >= 3:
            domain_min = float(tokens[1])
            domain_max = float(tokens[2])

    if length is None:
        raise LutLoadError(f"Invalid .spi1d (missing Length): {path}")
    if not values:
        raise LutLoadError(f"Invalid .spi1d (no LUT values): {path}")

    value_array = np.asarray(values, dtype=np.float32)
    if value_array.shape[0] != length:
        raise LutLoadError(
            f"Invalid .spi1d row count ({value_array.shape[0]} != {length}): {path}"
        )
    if value_array.shape[1] != components:
        raise LutLoadError(
            f"Invalid .spi1d component count ({value_array.shape[1]} != {components}): {path}"
        )

    x_values = np.linspace(domain_min, domain_max, length, dtype=np.float32)
    return LutPlotData(
        path=path,
        format="spi1d",
        source_kind="1d",
        channels=components,
        x_values=x_values,
        y_values=value_array,
        domain_min=domain_min,
        domain_max=domain_max,
    )


def _load_cube(path: Path) -> LutPlotData:
    """Parse a `.cube` LUT as either 1D data or 3D neutral-axis projection."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if non_empty and non_empty[0] == "CSPLUTV100":
        return _load_csp_style_3d_cube(path, non_empty, source_format="cube")

    domain_min = 0.0
    domain_max = 1.0
    lut_1d_size: int | None = None
    lut_3d_size: int | None = None
    values: list[tuple[float, float, float]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        key = tokens[0]
        if key == "TITLE":
            continue
        if key == "DOMAIN_MIN":
            if len(tokens) != 4:
                raise LutLoadError(f"Invalid .cube DOMAIN_MIN row: {path}")
            domain_triplet = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            if not _components_are_equal(domain_triplet):
                raise LutLoadError(f"Unsupported .cube DOMAIN_MIN components: {path}")
            domain_min = domain_triplet[0]
            continue
        if key == "DOMAIN_MAX":
            if len(tokens) != 4:
                raise LutLoadError(f"Invalid .cube DOMAIN_MAX row: {path}")
            domain_triplet = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            if not _components_are_equal(domain_triplet):
                raise LutLoadError(f"Unsupported .cube DOMAIN_MAX components: {path}")
            domain_max = domain_triplet[0]
            continue
        if key == "LUT_1D_SIZE" and len(tokens) >= 2:
            lut_1d_size = int(tokens[1])
            continue
        if key == "LUT_3D_SIZE" and len(tokens) >= 2:
            lut_3d_size = int(tokens[1])
            continue

        if len(tokens) >= 3:
            values.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))

    if lut_1d_size is not None:
        return _build_1d_cube_plot(path, lut_1d_size, values, domain_min, domain_max)
    if lut_3d_size is not None:
        return _build_3d_cube_projection(path, lut_3d_size, values, domain_min, domain_max)
    raise LutLoadError(f"Invalid .cube (missing LUT_1D_SIZE/LUT_3D_SIZE): {path}")


def _load_csp(path: Path) -> LutPlotData:
    """Parse a `.csp` LUT (CSPLUTV100) into neutral-axis plot data."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty or non_empty[0] != "CSPLUTV100":
        raise LutLoadError(f"Invalid .csp header: {path}")
    return _load_csp_style_3d_cube(path, non_empty, source_format="csp")


def _load_spi3d(path: Path) -> LutPlotData:
    """Parse a `.spi3d` LUT and extract neutral-axis projection samples."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(non_empty) < 3 or not non_empty[0].startswith("SPILUT"):
        raise LutLoadError(f"Invalid .spi3d header: {path}")

    size_tokens = non_empty[2].split()
    if len(size_tokens) < 3:
        raise LutLoadError(f"Invalid .spi3d size row: {path}")
    size_x = int(size_tokens[0])
    size_y = int(size_tokens[1])
    size_z = int(size_tokens[2])
    if size_x <= 1 or size_y <= 1 or size_z <= 1:
        raise LutLoadError(f"Invalid .spi3d dimensions: {path}")

    values: list[tuple[float, float, float]] = []
    for row in non_empty[3:]:
        tokens = row.split()
        if len(tokens) < 6:
            continue
        values.append((float(tokens[3]), float(tokens[4]), float(tokens[5])))

    expected_rows = size_x * size_y * size_z
    if len(values) != expected_rows:
        raise LutLoadError(f"Invalid .spi3d row count ({len(values)} != {expected_rows}): {path}")

    diag_size = min(size_x, size_y, size_z)
    diag_values = _extract_neutral_axis(values, size_x, size_y, sample_count=diag_size)

    x_values = np.linspace(0.0, 1.0, diag_size, dtype=np.float32)
    return LutPlotData(
        path=path,
        format="spi3d",
        source_kind="3d_neutral_axis",
        channels=3,
        x_values=x_values,
        y_values=diag_values,
        domain_min=0.0,
        domain_max=1.0,
    )


def _load_houdini_lut(path: Path) -> LutPlotData:
    """Parse a Houdini `.lut` file containing per-channel 1D curves."""
    lines = path.read_text(encoding="utf-8").splitlines()
    channel_values: dict[str, list[float]] = {"R": [], "G": [], "B": []}
    in_channel: str | None = None
    domain_min = 0.0
    domain_max = 1.0

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("From"):
            tokens = line.split()
            if len(tokens) >= 3:
                domain_min = float(tokens[-2])
                domain_max = float(tokens[-1])
            continue
        if line in ("R {", "G {", "B {"):
            in_channel = line[0]
            continue
        if line == "}":
            in_channel = None
            continue
        if in_channel is not None:
            channel_values[in_channel].append(float(line))

    length = len(channel_values["R"])
    if length == 0:
        raise LutLoadError(f"Invalid .lut (missing channel data): {path}")
    if len(channel_values["G"]) != length or len(channel_values["B"]) != length:
        raise LutLoadError(f"Invalid .lut (channel length mismatch): {path}")

    y_values = np.column_stack(
        [
            np.asarray(channel_values["R"], dtype=np.float32),
            np.asarray(channel_values["G"], dtype=np.float32),
            np.asarray(channel_values["B"], dtype=np.float32),
        ]
    )
    x_values = np.linspace(domain_min, domain_max, length, dtype=np.float32)
    return LutPlotData(
        path=path,
        format="lut",
        source_kind="1d",
        channels=3,
        x_values=x_values,
        y_values=y_values,
        domain_min=domain_min,
        domain_max=domain_max,
    )


def _load_3dl(path: Path) -> LutPlotData:
    """Parse a `.3dl` LUT and derive neutral-axis projection curve data."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(non_empty) < 2:
        raise LutLoadError(f"Invalid .3dl (too short): {path}")

    index_values = [int(token) for token in non_empty[0].split()]
    size = len(index_values)
    if size <= 1:
        raise LutLoadError(f"Invalid .3dl index row: {path}")

    values: list[tuple[float, float, float]] = []
    for row in non_empty[1:]:
        tokens = row.split()
        if len(tokens) < 3:
            continue
        values.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))

    expected_rows = size * size * size
    if len(values) != expected_rows:
        raise LutLoadError(f"Invalid .3dl row count ({len(values)} != {expected_rows}): {path}")

    diag_values = _extract_neutral_axis(values, size, size, sample_count=size)

    max_output = float(np.max(diag_values))
    if max_output <= 0.0:
        raise LutLoadError(f"Invalid .3dl output scale in: {path}")
    diag_values /= max_output
    x_values = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return LutPlotData(
        path=path,
        format="3dl",
        source_kind="3d_neutral_axis",
        channels=3,
        x_values=x_values,
        y_values=diag_values,
        domain_min=0.0,
        domain_max=1.0,
    )


def _load_cube_volume(path: Path) -> LutVolumeData | None:
    """Parse a `.cube` LUT and return full 3D volume data when present."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if non_empty and non_empty[0] == "CSPLUTV100":
        return _load_csp_style_3d_volume(path, non_empty, source_format="cube")

    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    lut_1d_size: int | None = None
    lut_3d_size: int | None = None
    values: list[tuple[float, float, float]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        key = tokens[0]
        if key == "TITLE":
            continue
        if key == "DOMAIN_MIN":
            if len(tokens) != 4:
                raise LutLoadError(f"Invalid .cube DOMAIN_MIN row: {path}")
            domain_min = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            if not _components_are_equal(domain_min):
                raise LutLoadError(f"Unsupported .cube DOMAIN_MIN components: {path}")
            continue
        if key == "DOMAIN_MAX":
            if len(tokens) != 4:
                raise LutLoadError(f"Invalid .cube DOMAIN_MAX row: {path}")
            domain_max = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            if not _components_are_equal(domain_max):
                raise LutLoadError(f"Unsupported .cube DOMAIN_MAX components: {path}")
            continue
        if key == "LUT_1D_SIZE" and len(tokens) >= 2:
            lut_1d_size = int(tokens[1])
            continue
        if key == "LUT_3D_SIZE" and len(tokens) >= 2:
            lut_3d_size = int(tokens[1])
            continue
        if len(tokens) >= 3:
            values.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))

    if lut_1d_size is not None:
        return None
    if lut_3d_size is None:
        raise LutLoadError(f"Invalid .cube (missing LUT_1D_SIZE/LUT_3D_SIZE): {path}")
    if lut_3d_size <= 1:
        raise LutLoadError(f"Invalid 3D .cube size: {lut_3d_size}")

    expected_rows = lut_3d_size * lut_3d_size * lut_3d_size
    if len(values) != expected_rows:
        raise LutLoadError(
            f"Invalid 3D .cube row count ({len(values)} != {expected_rows}): {path}"
        )

    value_array = np.asarray(values, dtype=np.float32).reshape(
        (lut_3d_size, lut_3d_size, lut_3d_size, 3)
    )
    return LutVolumeData(
        path=path,
        format="cube",
        size_x=lut_3d_size,
        size_y=lut_3d_size,
        size_z=lut_3d_size,
        values=value_array,
        domain_min=domain_min,
        domain_max=domain_max,
    )


def _load_csp_volume(path: Path) -> LutVolumeData:
    """Parse a `.csp` LUT into full 3D volume data."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty or non_empty[0] != "CSPLUTV100":
        raise LutLoadError(f"Invalid .csp header: {path}")
    return _load_csp_style_3d_volume(path, non_empty, source_format="csp")


def _load_spi3d_volume(path: Path) -> LutVolumeData:
    """Parse a `.spi3d` LUT into full 3D volume data."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(non_empty) < 3 or not non_empty[0].startswith("SPILUT"):
        raise LutLoadError(f"Invalid .spi3d header: {path}")

    size_tokens = non_empty[2].split()
    if len(size_tokens) < 3:
        raise LutLoadError(f"Invalid .spi3d size row: {path}")
    size_x = int(size_tokens[0])
    size_y = int(size_tokens[1])
    size_z = int(size_tokens[2])
    if size_x <= 1 or size_y <= 1 or size_z <= 1:
        raise LutLoadError(f"Invalid .spi3d dimensions: {path}")

    value_array = np.zeros((size_z, size_y, size_x, 3), dtype=np.float32)
    row_count = 0
    for row in non_empty[3:]:
        tokens = row.split()
        if len(tokens) < 6:
            continue
        x = int(tokens[0])
        y = int(tokens[1])
        z = int(tokens[2])
        if not (0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z):
            raise LutLoadError(f"Invalid .spi3d sample index in: {path}")
        value_array[z, y, x] = (
            float(tokens[3]),
            float(tokens[4]),
            float(tokens[5]),
        )
        row_count += 1

    expected_rows = size_x * size_y * size_z
    if row_count != expected_rows:
        raise LutLoadError(f"Invalid .spi3d row count ({row_count} != {expected_rows}): {path}")

    return LutVolumeData(
        path=path,
        format="spi3d",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        values=value_array,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def _load_3dl_volume(path: Path) -> LutVolumeData:
    """Parse a `.3dl` LUT into normalized full 3D volume data."""
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(non_empty) < 2:
        raise LutLoadError(f"Invalid .3dl (too short): {path}")

    index_values = [int(token) for token in non_empty[0].split()]
    size = len(index_values)
    if size <= 1:
        raise LutLoadError(f"Invalid .3dl index row: {path}")

    values: list[tuple[float, float, float]] = []
    for row in non_empty[1:]:
        tokens = row.split()
        if len(tokens) < 3:
            continue
        values.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))

    expected_rows = size * size * size
    if len(values) != expected_rows:
        raise LutLoadError(f"Invalid .3dl row count ({len(values)} != {expected_rows}): {path}")

    value_array = np.asarray(values, dtype=np.float32).reshape((size, size, size, 3))
    max_output = float(np.max(value_array))
    if max_output <= 0.0:
        raise LutLoadError(f"Invalid .3dl output scale in: {path}")
    value_array = np.asarray(value_array / max_output, dtype=np.float32)

    return LutVolumeData(
        path=path,
        format="3dl",
        size_x=size,
        size_y=size,
        size_z=size,
        values=value_array,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
    )


def _load_csp_style_3d_volume(
    path: Path,
    lines: list[str],
    source_format: str,
) -> LutVolumeData:
    """Parse CSPLUTV100 full 3D volume data."""
    if len(lines) < 3 or lines[1] != "3D":
        raise LutLoadError(f"Unsupported CSPLUT structure in: {path}")

    index = 2
    if index < len(lines) and lines[index] == "BEGIN METADATA":
        while index < len(lines) and lines[index] != "END METADATA":
            index += 1
        if index >= len(lines):
            raise LutLoadError(f"Invalid CSPLUT metadata block in: {path}")
        index += 1

    channel_domains: list[tuple[float, float]] = []
    has_shaper = False
    for _ in range(3):
        if index >= len(lines):
            raise LutLoadError(f"Incomplete CSPLUT domain block in: {path}")
        entries = int(lines[index])
        index += 1
        if entries < 2:
            raise LutLoadError(f"Invalid CSPLUT domain entry count ({entries}) in: {path}")
        points: list[tuple[float, float]] = []
        for _entry in range(entries):
            if index >= len(lines):
                raise LutLoadError(f"Incomplete CSPLUT domain values in: {path}")
            tokens = lines[index].split()
            index += 1
            if len(tokens) < 2:
                raise LutLoadError(f"Invalid CSPLUT domain row in: {path}")
            points.append((float(tokens[0]), float(tokens[1])))

        if entries == 2:
            in_min, in_max = points[0]
            out_min, out_max = points[1]
            if not (
                abs(in_min - 0.0) < 1e-9
                and abs(in_max - 1.0) < 1e-9
                and abs(out_min - 0.0) < 1e-9
                and abs(out_max - 1.0) < 1e-9
            ):
                has_shaper = True
            channel_domains.append((in_min, in_max))
            continue

        has_shaper = True
        channel_domains.append((points[0][0], points[-1][0]))

    if index >= len(lines):
        raise LutLoadError(f"Missing CSPLUT cube size line in: {path}")
    dims_tokens = lines[index].split()
    index += 1
    if len(dims_tokens) < 3:
        raise LutLoadError(f"Invalid CSPLUT cube size row in: {path}")
    size_x = int(dims_tokens[0])
    size_y = int(dims_tokens[1])
    size_z = int(dims_tokens[2])
    if size_x <= 1 or size_y <= 1 or size_z <= 1:
        raise LutLoadError(f"Invalid CSPLUT cube size ({size_x} {size_y} {size_z}) in: {path}")

    values: list[tuple[float, float, float]] = []
    for row in lines[index:]:
        if row.startswith("#"):
            continue
        tokens = row.split()
        if len(tokens) < 3:
            continue
        values.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))

    expected_rows = size_x * size_y * size_z
    if len(values) != expected_rows:
        raise LutLoadError(
            f"Invalid CSPLUT row count ({len(values)} != {expected_rows}): {path}"
        )

    volume_domain_min = (
        float(channel_domains[0][0]),
        float(channel_domains[1][0]),
        float(channel_domains[2][0]),
    )
    volume_domain_max = (
        float(channel_domains[0][1]),
        float(channel_domains[1][1]),
        float(channel_domains[2][1]),
    )

    return LutVolumeData(
        path=path,
        format=source_format,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        values=np.asarray(values, dtype=np.float32).reshape((size_z, size_y, size_x, 3)),
        domain_min=volume_domain_min,
        domain_max=volume_domain_max,
        has_shaper=has_shaper,
    )


def _load_csp_style_3d_cube(
    path: Path,
    lines: list[str],
    source_format: str,
) -> LutPlotData:
    """Parse CSPLUTV100 3D data used by `.csp` or CSP-style `.cube` files."""
    if len(lines) < 3 or lines[1] != "3D":
        raise LutLoadError(f"Unsupported CSPLUT structure in: {path}")

    index = 2
    if index < len(lines) and lines[index] == "BEGIN METADATA":
        while index < len(lines) and lines[index] != "END METADATA":
            index += 1
        if index >= len(lines):
            raise LutLoadError(f"Invalid CSPLUT metadata block in: {path}")
        index += 1

    channel_domains: list[tuple[float, float]] = []
    channel_prelut_points: list[list[tuple[float, float]]] = []
    csp_has_shaper = False
    for _ in range(3):
        if index >= len(lines):
            raise LutLoadError(f"Incomplete CSPLUT domain block in: {path}")
        entries = int(lines[index])
        index += 1
        if entries < 2:
            raise LutLoadError(f"Invalid CSPLUT domain entry count ({entries}) in: {path}")
        points: list[tuple[float, float]] = []
        for _entry in range(entries):
            if index >= len(lines):
                raise LutLoadError(f"Incomplete CSPLUT domain values in: {path}")
            tokens = lines[index].split()
            index += 1
            if len(tokens) < 2:
                raise LutLoadError(f"Invalid CSPLUT domain row in: {path}")
            points.append((float(tokens[0]), float(tokens[1])))
        if entries == 2:
            # In CSP, 2-entry preluts are commonly stored as:
            #   line 1: input min/max
            #   line 2: output min/max
            in_min, in_max = points[0]
            out_min, out_max = points[1]
            prelut_points = [(in_min, out_min), (in_max, out_max)]
            if not (
                abs(in_min - 0.0) < 1e-9
                and abs(in_max - 1.0) < 1e-9
                and abs(out_min - 0.0) < 1e-9
                and abs(out_max - 1.0) < 1e-9
            ):
                csp_has_shaper = True
            channel_prelut_points.append(prelut_points)
            channel_domains.append((in_min, in_max))
            continue

        csp_has_shaper = True
        channel_prelut_points.append(points)
        channel_domains.append((points[0][0], points[-1][0]))

    if index >= len(lines):
        raise LutLoadError(f"Missing CSPLUT cube size line in: {path}")
    dims_tokens = lines[index].split()
    index += 1
    if len(dims_tokens) < 3:
        raise LutLoadError(f"Invalid CSPLUT cube size row in: {path}")
    size_x = int(dims_tokens[0])
    size_y = int(dims_tokens[1])
    size_z = int(dims_tokens[2])
    if size_x <= 1 or size_y <= 1 or size_z <= 1:
        raise LutLoadError(f"Invalid CSPLUT cube size ({size_x} {size_y} {size_z}) in: {path}")

    values: list[tuple[float, float, float]] = []
    for row in lines[index:]:
        if row.startswith("#"):
            continue
        tokens = row.split()
        if len(tokens) < 3:
            continue
        values.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))

    expected_rows = size_x * size_y * size_z
    if len(values) != expected_rows:
        raise LutLoadError(
            f"Invalid CSPLUT row count ({len(values)} != {expected_rows}): {path}"
        )

    domain_min = channel_domains[0][0]
    domain_max = channel_domains[0][1]
    diag_size = min(size_x, size_y, size_z)
    x_values = np.linspace(domain_min, domain_max, diag_size, dtype=np.float32)
    value_array = np.asarray(values, dtype=np.float32).reshape((size_z, size_y, size_x, 3))

    if not csp_has_shaper:
        diag_values = np.zeros((diag_size, 3), dtype=np.float32)
        for i in range(diag_size):
            diag_values[i] = value_array[i, i, i]
    else:
        coordinates = np.zeros((diag_size, 3), dtype=np.float32)
        for channel in range(3):
            prelut_array = np.asarray(channel_prelut_points[channel], dtype=np.float32)
            shaper_outputs = evaluate_piecewise_linear(prelut_array, x_values)
            coordinates[:, channel] = normalize_values_to_unit_from_points(
                prelut_array,
                shaper_outputs,
            )
        diag_values = sample_lut3d_trilinear(value_array, coordinates)

    return LutPlotData(
        path=path,
        format=source_format,
        source_kind="3d_neutral_axis",
        channels=3,
        x_values=x_values,
        y_values=diag_values,
        domain_min=domain_min,
        domain_max=domain_max,
        csp_has_shaper=csp_has_shaper,
    )


def _extract_neutral_axis(
    values: list[tuple[float, float, float]],
    size_x: int,
    size_y: int,
    sample_count: int,
) -> np.ndarray:
    """Extract `r=g=b=i` lattice samples from flat LUT rows (red-fastest ordering)."""
    diag_values = np.zeros((sample_count, 3), dtype=np.float32)
    for i in range(sample_count):
        idx = i + (i * size_x) + (i * size_x * size_y)
        diag_values[i] = values[idx]
    return diag_values


def _components_are_equal(components: tuple[float, float, float]) -> bool:
    """Return `True` when all three channel components are effectively equal."""
    return (
        abs(components[0] - components[1]) < 1e-9
        and abs(components[1] - components[2]) < 1e-9
    )


def _build_1d_cube_plot(
    path: Path,
    size: int,
    values: list[tuple[float, float, float]],
    domain_min: float,
    domain_max: float,
) -> LutPlotData:
    """Build `LutPlotData` for a validated 1D `.cube` payload."""
    if size <= 1:
        raise LutLoadError(f"Invalid 1D .cube size: {size}")
    if len(values) != size:
        raise LutLoadError(f"Invalid 1D .cube row count ({len(values)} != {size}): {path}")

    value_array = np.asarray(values, dtype=np.float32)
    x_values = np.linspace(domain_min, domain_max, size, dtype=np.float32)
    return LutPlotData(
        path=path,
        format="cube",
        source_kind="1d",
        channels=3,
        x_values=x_values,
        y_values=value_array,
        domain_min=domain_min,
        domain_max=domain_max,
    )


def _build_3d_cube_projection(
    path: Path,
    size: int,
    values: list[tuple[float, float, float]],
    domain_min: float,
    domain_max: float,
) -> LutPlotData:
    """Build neutral-axis projection `LutPlotData` for a validated 3D `.cube` payload."""
    if size <= 1:
        raise LutLoadError(f"Invalid 3D .cube size: {size}")

    expected_rows = size * size * size
    if len(values) != expected_rows:
        raise LutLoadError(
            f"Invalid 3D .cube row count ({len(values)} != {expected_rows}): {path}"
        )

    # .cube tables are commonly serialized with red index changing fastest.
    # We extract neutral-axis samples at exact lattice points: r=g=b=i.
    diag_values = _extract_neutral_axis(values, size, size, sample_count=size)

    x_values = np.linspace(domain_min, domain_max, size, dtype=np.float32)
    return LutPlotData(
        path=path,
        format="cube",
        source_kind="3d_neutral_axis",
        channels=3,
        x_values=x_values,
        y_values=diag_values,
        domain_min=domain_min,
        domain_max=domain_max,
    )


# Static dispatch map avoids rebuilding a suffix->loader dict on every load call.
_LUT_PLOT_LOADERS: dict[str, Callable[[Path], LutPlotData]] = {
    ".spi1d": _load_spi1d,
    ".cube": _load_cube,
    ".csp": _load_csp,
    ".spi3d": _load_spi3d,
    ".lut": _load_houdini_lut,
    ".3dl": _load_3dl,
}

_LUT_VOLUME_LOADERS: dict[str, Callable[[Path], LutVolumeData | None]] = {
    ".spi1d": lambda _path: None,
    ".cube": _load_cube_volume,
    ".csp": _load_csp_volume,
    ".spi3d": _load_spi3d_volume,
    ".lut": lambda _path: None,
    ".3dl": _load_3dl_volume,
}
