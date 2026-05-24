"""Application entry point for Prism."""

from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from prism.ui.main_window import MainWindow


def main() -> None:
    """Launch the Prism application event loop."""
    # Allow Ctrl+C from the launching console to terminate the Qt event loop.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    signal_pump = QTimer()
    signal_pump.setInterval(100)
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start()

    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
