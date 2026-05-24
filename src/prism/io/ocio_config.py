"""OCIO config loading helpers for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_ocio_config(config_path: str) -> Any:
    """Load an OCIO config from disk.

    Args:
        config_path: Filesystem path to an ``.ocio`` config file.

    Returns:
        Loaded OCIO config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        RuntimeError: If the OCIO Python bindings are unavailable.
        ValueError: If the config cannot be parsed by OCIO.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"OCIO config file not found: {config_path}")

    try:
        import PyOpenColorIO as ocio
    except ImportError as exc:
        raise RuntimeError(
            "PyOpenColorIO is not available in this environment."
        ) from exc

    try:
        return ocio.Config.CreateFromFile(str(path))
    except Exception as exc:  # pragma: no cover - library-level exception types vary
        raise ValueError(f"Failed to load OCIO config: {config_path}") from exc


def list_colorspaces(config: Any) -> list[str]:
    """Return colorspace names in config-defined order.

    Args:
        config: Loaded OCIO config object.

    Returns:
        Ordered colorspace names from the config.

    Raises:
        ValueError: If colorspace enumeration fails in the OCIO backend.
    """
    names: list[str] = []

    try:
        for colorspace in config.getColorSpaces():
            names.append(colorspace.getName())
    except Exception as exc:  # pragma: no cover - library-level exception types vary
        raise ValueError("Failed to read colorspaces from OCIO config.") from exc

    return names


def list_looks(config: Any) -> list[str]:
    """Return look names in config-defined order.

    Args:
        config: Loaded OCIO config object.

    Returns:
        Ordered look names from the config.

    Raises:
        ValueError: If look enumeration fails in the OCIO backend.
    """
    names: list[str] = []

    try:
        looks = config.getLooks()
        if looks is not None:
            for look in looks:
                names.append(look.getName())
            return names
    except Exception:
        pass

    try:
        for index in range(config.getNumLooks()):
            names.append(config.getLookNameByIndex(index))
    except Exception as exc:  # pragma: no cover - library-level exception types vary
        raise ValueError("Failed to read looks from OCIO config.") from exc

    return names


def list_context_variables(config: Any) -> dict[str, str]:
    """Return config-declared OCIO context variable defaults.

    Args:
        config: Loaded OCIO config object.

    Returns:
        Mapping of context variable name to default value string.

    Raises:
        ValueError: If context variable names cannot be read from the config.
    """
    names: list[str] = []
    try:
        names = [str(name) for name in config.getEnvironmentVarNames()]
    except Exception as exc:  # pragma: no cover - library-level exception types vary
        raise ValueError("Failed to read context variable names from OCIO config.") from exc

    values: dict[str, str] = {}
    for name in names:
        try:
            default_value = config.getEnvironmentVarDefault(name)
        except Exception:
            default_value = ""
        values[name] = str(default_value) if default_value is not None else ""
    return values
