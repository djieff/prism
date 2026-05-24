"""UI panel for editing OCIO context variables."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ContextVariablesPanel(QWidget):
    """Editable OCIO context-variable panel with dynamic row rebuilding."""

    context_values_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._values: dict[str, str] = {}
        self._inputs: dict[str, QLineEdit] = {}
        self._input_order: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._help_label = QLabel(
            "Values used by OCIO to resolve context-dependent transforms.", self
        )
        self._help_label.setWordWrap(True)
        root.addWidget(self._help_label)

        self._message_label = QLabel("No OCIO config loaded.", self)
        self._message_label.setWordWrap(True)
        root.addWidget(self._message_label)

        self._rows_container = QWidget(self)
        self._rows_layout = QGridLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setHorizontalSpacing(10)
        self._rows_layout.setVerticalSpacing(6)
        root.addWidget(self._rows_container)

        root.addStretch(1)
        self._show_message("No OCIO config loaded.")

    def set_variables(self, values: dict[str, str]) -> None:
        """Rebuild panel rows from context-variable values.

        Args:
            values: Mapping of variable name to editable value.
        """
        normalized = {str(name): str(value) for name, value in values.items()}
        self._values = dict(sorted(normalized.items(), key=lambda pair: pair[0]))
        self._rebuild_rows()

    def get_values(self) -> dict[str, str]:
        """Return current variable values from panel state.

        Returns:
            Mapping of variable name to current value.
        """
        return dict(self._values)

    def clear(self) -> None:
        """Reset panel to no-config empty state."""
        self._values = {}
        self._clear_rows()
        self._show_message("No OCIO config loaded.")

    def show_no_variables_message(self) -> None:
        """Show empty-state message for config without context variables."""
        self._values = {}
        self._clear_rows()
        self._show_message("No OCIO context variables detected.")

    def _show_message(self, text: str) -> None:
        self._message_label.setText(text)
        self._message_label.setVisible(True)
        self._rows_container.setVisible(False)

    def _show_rows(self) -> None:
        self._message_label.setVisible(False)
        self._rows_container.setVisible(True)

    def _rebuild_rows(self) -> None:
        self._clear_rows()
        if not self._values:
            self._show_message("No OCIO context variables detected.")
            return

        self._show_rows()
        self._input_order = []
        for row, (name, value) in enumerate(self._values.items()):
            name_label = QLabel(name, self._rows_container)
            input_edit = QLineEdit(value, self._rows_container)
            input_edit.installEventFilter(self)
            input_edit.editingFinished.connect(
                lambda variable=name, editor=input_edit: self._on_edit_finished(
                    variable, editor
                )
            )

            self._rows_layout.addWidget(name_label, row, 0)
            self._rows_layout.addWidget(input_edit, row, 1)
            self._inputs[name] = input_edit
            self._input_order.append(name)

    def _clear_rows(self) -> None:
        self._inputs.clear()
        self._input_order = []
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                self._clear_layout(layout)

    def _clear_layout(self, layout: QHBoxLayout | QGridLayout | QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout)

    def _on_edit_finished(self, variable: str, editor: QLineEdit) -> None:
        new_value = editor.text()
        if self._values.get(variable) == new_value:
            return
        self._values[variable] = new_value
        self.context_values_changed.emit(self.get_values())

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # type: ignore[override]
        if (
            isinstance(watched, QLineEdit)
            and event.type() == QEvent.KeyPress
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key_Tab
        ):
            if self._move_focus_between_inputs(
                reverse=bool(event.modifiers() & Qt.ShiftModifier)
            ):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _move_focus_between_inputs(self, *, reverse: bool) -> bool:
        if not self._input_order:
            return False

        focused = self.focusWidget()
        if not isinstance(focused, QLineEdit):
            return False

        current_index = -1
        for index, variable in enumerate(self._input_order):
            if self._inputs.get(variable) is focused:
                current_index = index
                break
        if current_index < 0:
            return False

        step = -1 if reverse else 1
        next_index = (current_index + step) % len(self._input_order)
        next_editor = self._inputs.get(self._input_order[next_index])
        if next_editor is None:
            return False
        next_editor.setFocus(Qt.TabFocusReason)
        next_editor.selectAll()
        return True
