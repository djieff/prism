"""Tests for OCIO processor build/apply helpers."""

from __future__ import annotations

import types

import numpy as np
import pytest

from prism.core import ocio_processor


def test_build_ocio_processor_requires_input_output() -> None:
    with pytest.raises(ValueError, match="Input and output colorspaces are required"):
        ocio_processor.build_ocio_processor(object(), "", "out")

    with pytest.raises(ValueError, match="Input and output colorspaces are required"):
        ocio_processor.build_ocio_processor(object(), "in", "")


def test_build_ocio_processor_without_look_uses_simple_path() -> None:
    calls: list[tuple] = []

    class _Config:
        def getProcessor(self, *args):
            calls.append(args)
            return "processor"

    result = ocio_processor.build_ocio_processor(_Config(), "lin", "srgb")
    assert result == "processor"
    assert calls == [("lin", "srgb")]


def test_build_ocio_processor_with_context_values_uses_context_processor_path() -> None:
    calls: list[tuple] = []
    vars_set: list[tuple[str, str]] = []

    class _Context:
        def setStringVar(self, name: str, value: str) -> None:
            vars_set.append((name, value))

    class _Config:
        def getCurrentContext(self):
            return _Context()

        def getProcessor(self, *args):
            calls.append(args)
            return "ctx_processor"

    result = ocio_processor.build_ocio_processor(
        _Config(),
        "lin",
        "srgb",
        context_values={"SHOW": "demo", "SHOT": "010"},
    )
    assert result == "ctx_processor"
    assert vars_set == [("SHOW", "demo"), ("SHOT", "010")]
    assert len(calls) == 1
    assert len(calls[0]) == 3  # context, input, output


def test_build_ocio_processor_with_look_uses_look_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    look_calls: dict[str, str] = {}
    calls: list[tuple] = []

    class _FakeLookTransform:
        def setSrc(self, value: str) -> None:
            look_calls["src"] = value

        def setDst(self, value: str) -> None:
            look_calls["dst"] = value

        def setLooks(self, value: str) -> None:
            look_calls["look"] = value

    fake_ocio = types.SimpleNamespace(
        LookTransform=_FakeLookTransform,
        TRANSFORM_DIR_FORWARD="forward",
    )
    monkeypatch.setitem(__import__("sys").modules, "PyOpenColorIO", fake_ocio)

    class _Config:
        def getProcessor(self, *args):
            calls.append(args)
            return "look_processor"

    result = ocio_processor.build_ocio_processor(_Config(), "lin", "srgb", look="show_look")
    assert result == "look_processor"
    assert look_calls == {"src": "lin", "dst": "srgb", "look": "show_look"}
    assert len(calls) == 1
    assert len(calls[0]) == 1  # look transform only


def test_build_ocio_processor_errors_wrapped() -> None:
    class _Config:
        def getProcessor(self, *args):
            del args
            raise RuntimeError("backend failed")

    with pytest.raises(ValueError, match="Failed to build OCIO processor"):
        ocio_processor.build_ocio_processor(_Config(), "lin", "srgb")


def test_apply_ocio_transform_validates_shape() -> None:
    class _Processor:
        def getDefaultCPUProcessor(self):
            return object()

    with pytest.raises(ValueError, match="Expected image buffer shape"):
        ocio_processor.apply_ocio_transform(np.zeros((2, 2), dtype=np.float32), _Processor())


def test_apply_ocio_transform_applies_cpu_processor() -> None:
    captured_shapes: list[tuple[int, int]] = []

    class _CPU:
        def applyRGB(self, pixels: np.ndarray) -> None:
            captured_shapes.append(pixels.shape)
            pixels[:] = pixels * 2.0

    class _Processor:
        def getDefaultCPUProcessor(self):
            return _CPU()

    image = np.ones((2, 3, 3), dtype=np.float32)
    output = ocio_processor.apply_ocio_transform(image, _Processor())

    assert captured_shapes == [(6, 3)]
    assert np.allclose(output, np.full((2, 3, 3), 2.0, dtype=np.float32))
    # Ensure output is a separate array copy.
    assert output is not image


def test_apply_ocio_transform_wraps_backend_errors() -> None:
    class _CPU:
        def applyRGB(self, pixels: np.ndarray) -> None:
            del pixels
            raise RuntimeError("bad apply")

    class _Processor:
        def getDefaultCPUProcessor(self):
            return _CPU()

    image = np.ones((1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Failed to apply OCIO transform"):
        ocio_processor.apply_ocio_transform(image, _Processor())
