# Prism PyInstaller Packaging

This folder contains the PyInstaller workflow for local desktop bundles.

Current release target: Windows `onedir` build from the Prism development venv.

macOS and Linux compiled bundles are deferred until native build/test environments are available. Do not treat the Windows artifact as a cross-platform bundle.

Release/windowed build:

```powershell
C:\venv\prism_venv\Scripts\python.exe tools\build_pyinstaller.py --mode release --clean --noconfirm
```

Debug/console build:

```powershell
C:\venv\prism_venv\Scripts\python.exe tools\build_pyinstaller.py --mode debug --clean --noconfirm
```

Frozen smoke check:

```powershell
.\dist\pyinstaller\Prism\Prism.exe --frozen-smoke
```

Manual launch:

```powershell
.\dist\pyinstaller\Prism\Prism.exe
```

Generated output is intentionally ignored by git:

- `build/`
- `dist/`
- `*.egg-info/`

Use:

```powershell
git clean -fdX
```

to remove ignored generated packaging output and caches.

Notes:

- PyInstaller builds are platform-specific. Build Windows on Windows, macOS on macOS, and Linux on Linux.
- The current release scope is Windows only.
- macOS and Linux are future targets, not blockers for the current Windows artifact.
- The first supported artifact shape is `onedir`; `onefile` is deferred.
- OpenImageIO and OpenColorIO native DLL collection is intentionally explicit in the spec because those packages do not currently have obvious PyInstaller hooks in the local environment.
- The spec excludes obvious non-runtime developer/test packages such as pytest, pygments, matplotlib, PIL, Sphinx, Cython, Twisted, and Zope.
- Current known non-blocking warning classes include optional Windows-inapplicable POSIX modules, optional NumPy/SciPy pseudo-imports, and optional OpenImageIO pseudo-submodule warnings. Treat new top-level Prism dependency warnings separately.
