"""Application entry point for Prism."""

from __future__ import annotations

import ctypes
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from prism.ui.main_window import MainWindow

WINDOWS_APP_ID = "com.prism.viewer"
LINUX_DESKTOP_FILE_NAME = "com.prism.viewer"
MACOS_APP_NAME = "Prism Viewer"


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


def main() -> None:
    """Launch the Prism application event loop."""
    # Allow Ctrl+C from the launching console to terminate the Qt event loop.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _set_windows_app_user_model_id()

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
