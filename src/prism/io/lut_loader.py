"""LUT parsing helpers for LUT inspection plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


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


def load_lut_plot_data(path: Path) -> LutPlotData:
    """Load LUT file and return structured curve data for plotting."""
    suffix = path.suffix.lower()
    loader = _LUT_PLOT_LOADERS.get(suffix)
    if loader is not None:
        return loader(path)
    raise LutLoadError(f"Unsupported LUT format: {path.suffix or '<none>'}")


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
        diag_values = np.zeros((diag_size, 3), dtype=np.float32)
        for idx, t in enumerate(x_values):
            shaper_outputs = [
                _evaluate_piecewise_linear(channel_prelut_points[channel], float(t))
                for channel in range(3)
            ]
            normalized = [
                _normalize_to_unit_from_points(channel_prelut_points[channel], shaper_outputs[channel])
                for channel in range(3)
            ]
            diag_values[idx] = _sample_3d_lut_trilinear(
                value_array,
                normalized[0],
                normalized[1],
                normalized[2],
            )

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


def _evaluate_piecewise_linear(points: list[tuple[float, float]], x: float) -> float:
    """Evaluate a piecewise-linear function at `x` from ordered `(x, y)` points."""
    if not points:
        return x
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        if x <= x1:
            span = x1 - x0
            if abs(span) < 1e-12:
                return y1
            t = (x - x0) / span
            return y0 + (t * (y1 - y0))
    return points[-1][1]


def _normalize_to_unit_from_points(points: list[tuple[float, float]], value: float) -> float:
    """Normalize a value to `[0, 1]` using first/last output range from prelut points."""
    if not points:
        return max(0.0, min(1.0, value))
    out_min = points[0][1]
    out_max = points[-1][1]
    span = out_max - out_min
    if abs(span) < 1e-12:
        return 0.0
    t = (value - out_min) / span
    return max(0.0, min(1.0, t))


def _sample_3d_lut_trilinear(
    cube: np.ndarray, nx: float, ny: float, nz: float
) -> np.ndarray:
    """Trilinear-sample a `(z, y, x, c)` LUT cube at normalized coordinates."""
    z_size, y_size, x_size, _channels = cube.shape
    fx = nx * max(x_size - 1, 1)
    fy = ny * max(y_size - 1, 1)
    fz = nz * max(z_size - 1, 1)

    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    z0 = int(np.floor(fz))
    x1 = min(x0 + 1, x_size - 1)
    y1 = min(y0 + 1, y_size - 1)
    z1 = min(z0 + 1, z_size - 1)

    tx = fx - x0
    ty = fy - y0
    tz = fz - z0

    c000 = cube[z0, y0, x0]
    c100 = cube[z0, y0, x1]
    c010 = cube[z0, y1, x0]
    c110 = cube[z0, y1, x1]
    c001 = cube[z1, y0, x0]
    c101 = cube[z1, y0, x1]
    c011 = cube[z1, y1, x0]
    c111 = cube[z1, y1, x1]

    c00 = c000 * (1.0 - tx) + (c100 * tx)
    c10 = c010 * (1.0 - tx) + (c110 * tx)
    c01 = c001 * (1.0 - tx) + (c101 * tx)
    c11 = c011 * (1.0 - tx) + (c111 * tx)

    c0 = c00 * (1.0 - ty) + (c10 * ty)
    c1 = c01 * (1.0 - ty) + (c11 * ty)
    return c0 * (1.0 - tz) + (c1 * tz)


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
