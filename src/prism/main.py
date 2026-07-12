"""Application entry point for Prism."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from prism.ui.main_window import MainWindow

WINDOWS_APP_ID = "com.prism.viewer"
LINUX_DESKTOP_FILE_NAME = "com.prism.viewer"
MACOS_APP_NAME = "Prism Viewer"
FROZEN_SMOKE_ARG = "--frozen-smoke"


def _set_windows_app_user_model_id() -> None:
    """Set an explicit AppUserModelID so taskbar icon/grouping resolves to Prism."""
    if not sys.platform.startswith("win"):
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        # Keep startup resilient on non-standard runtimes.
        return


def _configure_cross_platform_app_identity(app: QApplication) -> None:
    """Set platform-specific app identity hints used by desktop shells."""
    if sys.platform.startswith("linux"):
        set_desktop_file_name = getattr(app, "setDesktopFileName", None)
        if callable(set_desktop_file_name):
            set_desktop_file_name(LINUX_DESKTOP_FILE_NAME)
        return
    if sys.platform == "darwin":
        set_application_name = getattr(app, "setApplicationName", None)
        if callable(set_application_name):
            set_application_name(MACOS_APP_NAME)


def _run_frozen_smoke() -> int:
    """Run non-interactive checks for frozen desktop bundles."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import colour  # noqa: F401
    import numpy as np
    import OpenImageIO  # noqa: F401
    import PyOpenColorIO  # noqa: F401

    from prism.core.lut_interpolation import sample_lut3d_trilinear
    from prism.core.scope_waveform_science import waveform_y_prime_coefficients
    from prism.io.lut_loader import load_lut_plot_data
    from prism.ui.lut_inspection_window import LutInspectionWindow

    app = QApplication.instance() or QApplication([sys.argv[0], FROZEN_SMOKE_ARG])
    _configure_cross_platform_app_identity(app)

    window = MainWindow()
    if window.windowTitle() != "Prism Viewer":
        raise RuntimeError("Main window title smoke check failed")

    lut_window = LutInspectionWindow()
    with tempfile.TemporaryDirectory() as tmp:
        lut_path = Path(tmp) / "identity.cube"
        lut_path.write_text(
            "\n".join(
                [
                    'TITLE "Identity"',
                    "DOMAIN_MIN 0 0 0",
                    "DOMAIN_MAX 1 1 1",
                    "LUT_1D_SIZE 2",
                    "0 0 0",
                    "1 1 1",
                ]
            ),
            encoding="utf-8",
        )
        data = load_lut_plot_data(lut_path)
        lut_window.load_lut_path(lut_path)

    if data.y_values.shape != (2, 3):
        raise RuntimeError("LUT smoke check failed")

    coeffs = waveform_y_prime_coefficients("ITU-R BT.709")
    if coeffs.shape != (3,):
        raise RuntimeError("Colour waveform coefficient smoke check failed")

    cube = np.zeros((2, 2, 2, 3), dtype=np.float32)
    cube[1, 1, 1] = 1.0
    sampled = sample_lut3d_trilinear(
        cube,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    )
    if sampled.shape != (2, 3):
        raise RuntimeError("SciPy LUT interpolation smoke check failed")

    print("prism_frozen_smoke_ok")
    return 0


def main() -> None:
    """Launch the Prism application event loop."""
    # Allow Ctrl+C from the launching console to terminate the Qt event loop.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _set_windows_app_user_model_id()

    if FROZEN_SMOKE_ARG in sys.argv:
        raise SystemExit(_run_frozen_smoke())

    app = QApplication(sys.argv)
    _configure_cross_platform_app_identity(app)
    signal_pump = QTimer()
    signal_pump.setInterval(100)
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start()

    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
