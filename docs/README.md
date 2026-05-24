# Prism Docs

This directory contains architecture and subsystem guidance for Prism.

## Index

- [Architecture](./architecture.md)
- [Core](./core.md)
- [I/O](./io.md)
- [UI](./ui.md)
- [OCIO Context](./ocio-context.md)
- [Packaging](./packaging.md)

## Scope

These docs explain subsystem responsibilities, boundaries, and data flow.
Implementation details should remain in code-level docstrings and plans.

## Quality Checks

Run from repository root (using `C:\venv\prism_venv`):

```powershell
C:\venv\prism_venv\Scripts\python -m ruff check src tests
C:\venv\prism_venv\Scripts\python -m mypy
C:\venv\prism_venv\Scripts\python -m pytest -q
```

## Contributor Workflow

### Setup

1. Create/activate the project virtual environment.
2. Install Prism in editable mode from repository root:

```powershell
C:\venv\prism_venv\Scripts\python -m pip install -e .
```

### Pre-PR Checklist

Run these before opening a PR:

```powershell
C:\venv\prism_venv\Scripts\python -m ruff check src tests
C:\venv\prism_venv\Scripts\python -m mypy
C:\venv\prism_venv\Scripts\python -m pytest -q
```
