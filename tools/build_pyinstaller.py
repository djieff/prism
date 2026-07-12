"""Build Prism with PyInstaller."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "packaging" / "pyinstaller" / "prism.spec"
DEFAULT_DIST = ROOT / "dist" / "pyinstaller"
DEFAULT_WORK = ROOT / "build" / "pyinstaller"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Prism desktop bundles with PyInstaller.")
    parser.add_argument(
        "--mode",
        choices=("release", "debug"),
        default="release",
        help="Build mode. Debug keeps a console window for diagnostics.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="Path to the PyInstaller spec file.",
    )
    parser.add_argument(
        "--distpath",
        type=Path,
        default=DEFAULT_DIST,
        help="PyInstaller output directory.",
    )
    parser.add_argument(
        "--workpath",
        type=Path,
        default=DEFAULT_WORK,
        help="PyInstaller work directory.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Ask PyInstaller to clean its cache before building.",
    )
    parser.add_argument(
        "--noconfirm",
        action="store_true",
        help="Replace previous output without prompting.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = args.spec.resolve()
    if not spec.exists():
        raise FileNotFoundError(f"Missing PyInstaller spec: {spec}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec),
        "--distpath",
        str(args.distpath.resolve()),
        "--workpath",
        str(args.workpath.resolve()),
    ]
    if args.clean:
        command.append("--clean")
    if args.noconfirm:
        command.append("--noconfirm")

    env = os.environ.copy()
    env["PRISM_PYINSTALLER_CONSOLE"] = "1" if args.mode == "debug" else "0"

    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
