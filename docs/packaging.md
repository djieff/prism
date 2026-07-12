# Packaging

## Purpose

Describe Prism packaging, installation, entrypoint behavior, and packaged UI assets.

## Current Setup

Prism uses setuptools with `pyproject.toml` and `src/` layout discovery.

Key metadata:
- project name: `prism-viewer`
- version source: `prism.__version__` (dynamic)
- Python requirement: `>=3.11,<3.15`
- console script: `prism = prism.main:main`

Runtime dependencies:
- `PySide6`
- `OpenImageIO`
- `OpenColorIO`
- `numpy`
- `scipy>=1.17,<2`
- `colour-science>=0.4.7,<0.5`

Install workflow (repo root):

```powershell
python -m pip install -e .
```

Then launch:

```powershell
prism
```

## Entry Point

Application entrypoint module:
- `src/prism/main.py`

Contract:
- `main() -> int`
- module guard raises `SystemExit(main())`

Behavior:
- creates `QApplication`
- installs Ctrl+C signal handling compatibility for console-launched runs
- constructs and shows `MainWindow`

## Assets

UI asset loading is package-relative (not cwd-relative) via `importlib.resources` in UI code.

Included package data (from `pyproject.toml`):
- `prism/ui/assets/icons/*.png`
- `prism/ui/assets/images/*.png`

Current icon usage:
- Windows prefers `app_icon.ico` at runtime with PNG fallback in UI loader path.
- Linux and macOS use PNG runtime window icon loading.
- Startup sets platform identity hints in `src/prism/main.py`:
  - Windows AppUserModelID: `com.prism.viewer`
  - Linux desktop file name: `com.prism.viewer`
  - macOS application name: `Prism Viewer`

About banner usage:
- `about_banner.png` is loaded and displayed in `Help -> About Prism Viewer`.

## Platform Packaging Notes

### Windows
- Use a Windows AppUserModelID and `.ico` application icon to ensure the correct taskbar icon and application identity.
- Current compiled desktop release target is Windows `onedir` via PyInstaller.
- Build command from the repo root:

```powershell
C:\venv\prism_venv\Scripts\python.exe tools\build_pyinstaller.py --mode release --clean --noconfirm
```

- Final artifact:

```text
dist\pyinstaller\Prism\Prism.exe
```

- Required validation before treating the artifact as releasable:
  - full source tests pass
  - Ruff passes
  - mypy passes
  - `pip check` passes
  - `Prism.exe --frozen-smoke` prints `prism_frozen_smoke_ok`
  - hidden/offscreen startup smoke passes
  - manual desktop smoke confirms image loading, Waveform Monitor, and LUT Inspector

### Linux
- Future packaged builds may provide a `.desktop` launcher for desktop integration and application menus.
- Linux compiled desktop bundles are deferred until a native Linux build/test environment is available.
- Do not claim Linux support from a Windows-built artifact.

### macOS
- Future packaged builds may include a `.app` bundle and `.icns` application icon for proper Finder and Dock integration.
- macOS compiled desktop bundles are deferred until a native macOS build/test environment is available.
- Do not claim macOS support from a Windows-built artifact.

## PyInstaller Release Matrix

Current compiled desktop support:

| Platform | Status | Notes |
| --- | --- | --- |
| Windows x64 | current release target | built locally as unsigned `onedir` |
| macOS | deferred | requires native macOS build/test environment |
| Linux | deferred | requires native Linux build/test environment |

Deferred compiled-package work:
- signed installers
- macOS signing/notarization
- AppImage/Flatpak/Snap
- `onefile` bundles
- auto-update infrastructure
