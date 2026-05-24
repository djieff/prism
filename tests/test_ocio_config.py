"""Tests for OCIO config loading and metadata extraction helpers."""

from __future__ import annotations

import builtins
import types
from pathlib import Path

import pytest

from prism.io import ocio_config


def test_load_ocio_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ocio_config.load_ocio_config("Z:/does/not/exist/config.ocio")


def test_load_ocio_config_import_error_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "test.ocio"
    config_path.write_text("ocio")

    real_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):
        if name == "PyOpenColorIO":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="PyOpenColorIO is not available"):
        ocio_config.load_ocio_config(str(config_path))


def test_load_ocio_config_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "test.ocio"
    config_path.write_text("ocio")

    class _FakeConfigApi:
        @staticmethod
        def CreateFromFile(path: str) -> str:
            return f"loaded:{path}"

    fake_ocio = types.SimpleNamespace(Config=_FakeConfigApi)
    monkeypatch.setitem(__import__("sys").modules, "PyOpenColorIO", fake_ocio)

    loaded = ocio_config.load_ocio_config(str(config_path))

    assert loaded == f"loaded:{config_path}"


def test_load_ocio_config_backend_failure_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bad.ocio"
    config_path.write_text("ocio")

    class _FakeConfigApi:
        @staticmethod
        def CreateFromFile(path: str) -> str:
            del path
            raise RuntimeError("parse failed")

    fake_ocio = types.SimpleNamespace(Config=_FakeConfigApi)
    monkeypatch.setitem(__import__("sys").modules, "PyOpenColorIO", fake_ocio)

    with pytest.raises(ValueError, match="Failed to load OCIO config"):
        ocio_config.load_ocio_config(str(config_path))


def test_list_colorspaces_returns_names_in_order() -> None:
    class _Colorspace:
        def __init__(self, name: str) -> None:
            self._name = name

        def getName(self) -> str:
            return self._name

    class _Config:
        def getColorSpaces(self):
            return [_Colorspace("lin"), _Colorspace("srgb")]

    assert ocio_config.list_colorspaces(_Config()) == ["lin", "srgb"]


def test_list_colorspaces_failure_raises_value_error() -> None:
    class _Config:
        def getColorSpaces(self):
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="Failed to read colorspaces"):
        ocio_config.list_colorspaces(_Config())


def test_list_looks_uses_getLooks_when_available() -> None:
    class _Look:
        def __init__(self, name: str) -> None:
            self._name = name

        def getName(self) -> str:
            return self._name

    class _Config:
        def getLooks(self):
            return [_Look("show"), _Look("shot")]

        def getNumLooks(self) -> int:
            return 0

        def getLookNameByIndex(self, index: int) -> str:
            del index
            return ""

    assert ocio_config.list_looks(_Config()) == ["show", "shot"]


def test_list_looks_falls_back_to_indexed_api() -> None:
    class _Config:
        def getLooks(self):
            raise RuntimeError("legacy API only")

        def getNumLooks(self) -> int:
            return 2

        def getLookNameByIndex(self, index: int) -> str:
            return ["film", "video"][index]

    assert ocio_config.list_looks(_Config()) == ["film", "video"]


def test_list_looks_failure_raises_value_error() -> None:
    class _Config:
        def getLooks(self):
            raise RuntimeError("nope")

        def getNumLooks(self) -> int:
            raise RuntimeError("nope")

        def getLookNameByIndex(self, index: int) -> str:
            del index
            return ""

    with pytest.raises(ValueError, match="Failed to read looks"):
        ocio_config.list_looks(_Config())


def test_list_context_variables_handles_defaults_and_missing_defaults() -> None:
    class _Config:
        def getEnvironmentVarNames(self):
            return ["SHOW", "SHOT"]

        def getEnvironmentVarDefault(self, name: str):
            if name == "SHOW":
                return "test_show"
            raise RuntimeError("missing default")

    assert ocio_config.list_context_variables(_Config()) == {
        "SHOW": "test_show",
        "SHOT": "",
    }


def test_list_context_variables_name_read_failure_raises_value_error() -> None:
    class _Config:
        def getEnvironmentVarNames(self):
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="Failed to read context variable names"):
        ocio_config.list_context_variables(_Config())
