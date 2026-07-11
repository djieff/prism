"""Tests for LUT parsing used by LUT inspection plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from prism.io import lut_loader
from prism.io.lut_loader import LutLoadError, load_lut_plot_data


class LutLoaderTests(unittest.TestCase):
    def _fixture(self, pattern: str) -> Path:
        root = Path("samples/LUTs")
        candidates = sorted(root.rglob(pattern))
        self.assertTrue(candidates, f"Missing fixture: {pattern}")
        preferred = [path for path in candidates if "old" not in path.parts]
        return preferred[0] if preferred else candidates[0]

    def _csp_style_cube_fixture(self) -> Path | None:
        root = Path("samples/LUTs")
        cubes = sorted(root.rglob("*.cube"))
        preferred = [path for path in cubes if "old" not in path.parts] + [
            path for path in cubes if "old" in path.parts
        ]
        for path in preferred:
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
            except Exception:
                continue
            if first_line == "CSPLUTV100":
                return path
        return None

    def test_loads_spi1d_sample(self) -> None:
        path = self._fixture("*.spi1d")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "spi1d")
        self.assertEqual(data.source_kind, "1d")
        self.assertGreater(data.x_values.shape[0], 4)
        self.assertEqual(data.y_values.shape[0], data.x_values.shape[0])
        self.assertEqual(data.y_values.shape[1], data.channels)

    def test_loads_3d_cube_as_neutral_axis_projection(self) -> None:
        path = self._fixture("*.cube")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "cube")
        self.assertEqual(data.source_kind, "3d_neutral_axis")
        self.assertEqual(data.channels, 3)
        self.assertGreater(data.x_values.shape[0], 4)
        self.assertEqual(data.y_values.shape[0], data.x_values.shape[0])
        self.assertEqual(data.y_values.shape[1], 3)

    def test_loads_csp_style_3d_cube_with_cube_extension(self) -> None:
        path = self._csp_style_cube_fixture()
        if path is None:
            self.skipTest("No CSP-style .cube fixture (CSPLUTV100) present in samples/LUTs")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "cube")
        self.assertEqual(data.source_kind, "3d_neutral_axis")
        self.assertEqual(data.channels, 3)
        self.assertGreater(data.x_values.shape[0], 4)
        self.assertEqual(data.y_values.shape, (data.x_values.shape[0], 3))

    def test_loads_csp_file(self) -> None:
        path = self._fixture("*.csp")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "csp")
        self.assertEqual(data.source_kind, "3d_neutral_axis")
        self.assertEqual(data.x_values.shape[0], 33)
        self.assertEqual(data.y_values.shape, (33, 3))

    def test_loads_spi3d_file(self) -> None:
        path = self._fixture("*.spi3d")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "spi3d")
        self.assertEqual(data.source_kind, "3d_neutral_axis")
        self.assertEqual(data.x_values.shape[0], 33)
        self.assertEqual(data.y_values.shape, (33, 3))

    def test_loads_houdini_lut_file(self) -> None:
        path = self._fixture("*.lut")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "lut")
        self.assertEqual(data.source_kind, "1d")
        self.assertEqual(data.channels, 3)
        self.assertEqual(data.x_values.shape[0], 1024)
        self.assertEqual(data.y_values.shape, (1024, 3))

    def test_loads_3dl_file(self) -> None:
        path = self._fixture("*.3dl")
        data = load_lut_plot_data(path)
        self.assertEqual(data.format, "3dl")
        self.assertEqual(data.source_kind, "3d_neutral_axis")
        self.assertEqual(data.channels, 3)
        self.assertEqual(data.x_values.shape[0], 17)
        self.assertEqual(data.y_values.shape, (17, 3))

    def test_loads_synthetic_1d_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "simple_1d.cube"
            lut_path.write_text(
                "\n".join(
                    [
                        'TITLE "Simple"',
                        "DOMAIN_MIN 0.0 0.0 0.0",
                        "DOMAIN_MAX 1.0 1.0 1.0",
                        "LUT_1D_SIZE 4",
                        "0.0 0.0 0.0",
                        "0.25 0.20 0.30",
                        "0.5 0.45 0.55",
                        "1.0 1.0 1.0",
                    ]
                ),
                encoding="utf-8",
            )
            data = load_lut_plot_data(lut_path)
            self.assertEqual(data.source_kind, "1d")
            self.assertEqual(data.channels, 3)
            self.assertEqual(data.x_values.shape[0], 4)
            self.assertEqual(data.y_values.shape, (4, 3))

    def test_loads_synthetic_shaped_csp_with_interpolated_neutral_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "shaped.csp"
            rows = ["CSPLUTV100", "3D"]
            rows.extend(
                [
                    "2",
                    "0 1",
                    "-0.5 1.5",
                    "2",
                    "0 1",
                    "0 1",
                    "2",
                    "0 1",
                    "0 1",
                    "3 3 3",
                ]
            )
            for z in range(3):
                for y in range(3):
                    for x in range(3):
                        rows.append(f"{x / 2.0:.1f} {y / 2.0:.1f} {z / 2.0:.1f}")
            lut_path.write_text("\n".join(rows), encoding="utf-8")

            data = load_lut_plot_data(lut_path)

            self.assertEqual(data.format, "csp")
            self.assertEqual(data.source_kind, "3d_neutral_axis")
            self.assertTrue(data.csp_has_shaper)
            self.assertEqual(data.x_values.shape[0], 3)
            self.assertEqual(data.y_values.shape, (3, 3))
            self.assertTrue(
                (abs(data.y_values[0] - [0.0, 0.0, 0.0]) < 1e-6).all()
            )
            self.assertTrue(
                (abs(data.y_values[1] - [0.5, 0.5, 0.5]) < 1e-6).all()
            )
            self.assertTrue(
                (abs(data.y_values[2] - [1.0, 1.0, 1.0]) < 1e-6).all()
            )

    def test_raises_for_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "file.spi3d"
            lut_path.write_text("Version 1", encoding="utf-8")
            with self.assertRaises(LutLoadError):
                load_lut_plot_data(lut_path)

    def test_raises_for_invalid_spi1d(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "broken.spi1d"
            lut_path.write_text(
                "\n".join(["Version 1", "Length 4", "Components 1", "From 0 1", "{", "0.0", "}"]),
                encoding="utf-8",
            )
            with self.assertRaises(LutLoadError):
                load_lut_plot_data(lut_path)

    def test_dispatches_loader_by_case_insensitive_suffix(self) -> None:
        sentinel = object()
        loader = mock.Mock(return_value=sentinel)
        with mock.patch.dict(lut_loader._LUT_PLOT_LOADERS, {".spi1d": loader}):
            result = load_lut_plot_data(Path("fake.SPI1D"))
        self.assertIs(result, sentinel)
        loader.assert_called_once_with(Path("fake.SPI1D"))

    def test_raises_for_cube_with_mixed_domain_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "mixed_domain.cube"
            lut_path.write_text(
                "\n".join(
                    [
                        "DOMAIN_MIN 0.0 0.1 0.0",
                        "DOMAIN_MAX 1.0 1.0 1.0",
                        "LUT_1D_SIZE 2",
                        "0.0 0.0 0.0",
                        "1.0 1.0 1.0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(LutLoadError):
                load_lut_plot_data(lut_path)

    def test_raises_for_cube_with_mixed_domain_max_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "mixed_domain_max.cube"
            lut_path.write_text(
                "\n".join(
                    [
                        "DOMAIN_MIN 0.0 0.0 0.0",
                        "DOMAIN_MAX 1.0 0.9 1.0",
                        "LUT_1D_SIZE 2",
                        "0.0 0.0 0.0",
                        "1.0 1.0 1.0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(LutLoadError):
                load_lut_plot_data(lut_path)

    def test_raises_for_cube_with_invalid_domain_min_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "invalid_domain_min.cube"
            lut_path.write_text(
                "\n".join(
                    [
                        "DOMAIN_MIN 0.0 0.0",
                        "DOMAIN_MAX 1.0 1.0 1.0",
                        "LUT_1D_SIZE 2",
                        "0.0 0.0 0.0",
                        "1.0 1.0 1.0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(LutLoadError):
                load_lut_plot_data(lut_path)

    def test_raises_for_cube_with_invalid_domain_max_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "invalid_domain_max.cube"
            lut_path.write_text(
                "\n".join(
                    [
                        "DOMAIN_MIN 0.0 0.0 0.0",
                        "DOMAIN_MAX 1.0 1.0",
                        "LUT_1D_SIZE 2",
                        "0.0 0.0 0.0",
                        "1.0 1.0 1.0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(LutLoadError):
                load_lut_plot_data(lut_path)


if __name__ == "__main__":
    unittest.main()
