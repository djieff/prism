"""Dock wrapper for OCIO context variable editing."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QWidget

from prism.ui.context_variables_panel import ContextVariablesPanel


class ContextVariablesDock(QDockWidget):
    """Floating-capable dock that hosts the OCIO context panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("OCIO Context Variables", parent)
        self.setObjectName("ocio_context_variables_dock")
        self.setToolTip("Values used by OCIO to resolve context-dependent transforms.")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        self._panel = ContextVariablesPanel(self)
        self.setWidget(self._panel)

    @property
    def panel(self) -> ContextVariablesPanel:
        """Return the hosted context-variable panel.

        Returns:
            Embedded panel instance.
        """
        return self._panel

    def set_variables(self, values: dict[str, str]) -> None:
        """Proxy to panel variable-set API.

        Args:
            values: Mapping of variable name to value.
        """
        self._panel.set_variables(values)

    def get_values(self) -> dict[str, str]:
        """Proxy to panel value collection API.

        Returns:
            Mapping of variable name to current value.
        """
        return self._panel.get_values()

    def clear(self) -> None:
        """Proxy to panel clear API."""
        self._panel.clear()

    def show_no_variables_message(self) -> None:
        """Proxy for config-loaded empty-variable state."""
        self._panel.show_no_variables_message()
