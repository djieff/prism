# Packaging

## Purpose

Describe Prism packaging, installation, entrypoint behavior, and packaged UI assets.

## Current Setup

Prism uses setuptools with `pyproject.toml` and `src/` layout discovery.

Key metadata:
- project name: `prism-viewer`
- version source: `prism.__version__` (dynamic)
- Python requirement: `>=3.11`
- console script: `prism = prism.main:main`

Runtime dependencies:
- `PySide6`
- `OpenImageIO`
- `OpenColorIO`
- `numpy`

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

About banner usage:
- `about_banner.png` is loaded and displayed in `Help -> About Prism Viewer`.

## Notes

- Editable installs are the primary local-dev workflow.
- If `prism` command is unavailable, ensure the active shell uses the intended venv.
