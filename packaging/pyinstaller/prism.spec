# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import os
from importlib import util
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata


ROOT = Path(SPECPATH).parents[1]
SRC = ROOT / "src"


def _package_dir(module_name: str) -> Path | None:
    spec = util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def _add_existing_dir(datas: list[tuple[str, str]], source: Path, dest: str) -> None:
    if source.exists():
        datas.append((str(source), dest))


def _add_dlls_from_dir(binaries: list[tuple[str, str]], source: Path, dest: str) -> None:
    if not source.exists():
        return
    for dll_path in source.glob("*.dll"):
        binaries.append((str(dll_path), dest))


datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []
excludes = [
    "argcomplete",
    "Cython",
    "cython",
    "IPython",
    "matplotlib",
    "PIL",
    "pygments",
    "pytest",
    "sphinx",
    "twisted",
    "zope",
]

# Prism package metadata is required because prism.__version__ uses
# importlib.metadata.version("prism-viewer").
for distribution_name in ("prism-viewer", "colour-science"):
    try:
        datas += copy_metadata(distribution_name)
    except Exception:
        pass

# Prism UI assets are loaded package-relative at runtime.
_add_existing_dir(datas, SRC / "prism" / "ui" / "assets", "prism/ui/assets")

# OpenImageIO ships important runtime DLLs/data inside its wheel.
openimageio_dir = _package_dir("OpenImageIO")
if openimageio_dir is not None:
    _add_dlls_from_dir(binaries, openimageio_dir / "bin", "OpenImageIO/bin")
    _add_existing_dir(datas, openimageio_dir / "share", "OpenImageIO/share")

# PyOpenColorIO needs its runtime DLL, but Prism does not need OCIO command-line tools.
pyopencolorio_dir = _package_dir("PyOpenColorIO")
if pyopencolorio_dir is not None:
    ocio_bin = pyopencolorio_dir / "bin"
    if ocio_bin.exists():
        for dll_path in ocio_bin.glob("OpenColorIO*.dll"):
            binaries.append((str(dll_path), "PyOpenColorIO/bin"))


a = Analysis(
    [str(SRC / "prism" / "main.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(Path(SPECPATH) / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Prism",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=os.environ.get("PRISM_PYINSTALLER_CONSOLE", "0") == "1",
    icon=str(SRC / "prism" / "ui" / "assets" / "icons" / "app_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Prism",
)
