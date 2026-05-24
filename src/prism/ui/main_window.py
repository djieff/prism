"""Main window for the Prism viewer."""

from __future__ import annotations

import platform
from importlib import resources
from pathlib import Path

import numpy as np
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, qVersion
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QImage,
    QIntValidator,
    QKeyEvent,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from prism import __version__ as prism_version
from prism.core.frame_service import FrameService
from prism.core.ocio_processor import apply_ocio_transform, build_ocio_processor
from prism.core.source_models import create_source_from_path
from prism.core.ui_tokens import (
    COMPARE_MODES_REQUIRING_BOTH_IMAGES,
    COMPARE_MODES_WITH_SIDE_CONTEXT,
    AlignmentAnchor,
    CompareModeData,
    ViewerChannel,
)
from prism.core.viewer_state import (
    CompareMode,
    CompareState,
    CompareViewState,
    PanelState,
    ViewerSide,
)
from prism.io.image_loader import float_rgb_to_qimage, qimage_to_float_rgb
from prism.io.ocio_config import (
    list_colorspaces,
    list_context_variables,
    list_looks,
    load_ocio_config,
)
from prism.ui.compare_view import CompareView
from prism.ui.context_variables_dock import ContextVariablesDock
from prism.ui.status_formatters import panel_frame_suffix, persistent_status_message, side_label

# Layout constants (maintainability-focused; no behavior changes).
ZERO_MARGINS = (0, 0, 0, 0)
TOOLBAR_SPACING = 8
VIEWER_SPACING = 6
SIDE_BLOCK_SPACING = 12
FOOTER_SPACING = 10
TONE_SLIDER_WIDTH = 240
TONE_VALUE_LABEL_MIN_WIDTH = 44
FRAME_BUTTON_WIDTH = 30
FRAME_EDIT_WIDTH = 72
TOP_TOOLBAR_SIDE_GAP = 24


class MainWindow(QMainWindow):
    """Viewer with config/image loading and OCIO processing refresh."""

    def __init__(self) -> None:
        super().__init__()
        self._apply_app_icon()
        self.setWindowTitle("Prism Viewer")
        self.resize(1000, 700)
        self.setAcceptDrops(True)
        self._compare_state = CompareState()
        self._frame_service = FrameService(cache_size=24)
        self._display_images: dict[ViewerSide, QImage | None] = {"left": None, "right": None}
        self._processed_display_buffers: dict[ViewerSide, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._diff_display_image: QImage | None = None
        self._diff_status_message = "Load both A and B for Diff mode"
        self._compare_view_state = CompareViewState()
        self._channel_view = ViewerChannel.RGB.value
        self._sync_nav_enabled = True
        self._split_view_transforms: dict[ViewerSide, tuple[float, float, float]] = {
            "left": (
                self._compare_view_state.zoom,
                self._compare_view_state.pan_x,
                self._compare_view_state.pan_y,
            ),
            "right": (
                self._compare_view_state.zoom,
                self._compare_view_state.pan_x,
                self._compare_view_state.pan_y,
            ),
        }
        self._panel_status_messages: dict[ViewerSide, str] = {
            "left": "Waiting for image",
            "right": "Waiting for image",
        }
        self._loaded_ocio_config = None
        self._loaded_ocio_config_path: str | None = None
        self._ocio_context_values: dict[str, str] = {}
        self._hotkeys_action: QAction | None = None
        self._hotkeys_dialog: QDialog | None = None
        self._viewer_background_group: QActionGroup | None = None
        self._persistent_status_message = ""
        self._global_luminance = 1.0
        self._global_exposure_stops = 0.0
        self._hovered_view_side: ViewerSide | None = None
        self._tone_refresh_timer = QTimer(self)
        self._tone_refresh_timer.setSingleShot(True)
        self._tone_refresh_timer.setInterval(16)

        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.addLayout(self._build_top_toolbar())
        root_layout.addWidget(self._build_viewer_area(), 1)
        root_layout.addWidget(self._build_side_controls_block())
        root_layout.addLayout(self._build_footer_hud())
        self._clear_hud_sample()

        self._context_variables_dock = ContextVariablesDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self._context_variables_dock)
        self._context_variables_dock.hide()
        self._set_viewer_background(QColor(Qt.black))

        self._build_menu_bar()
        self.statusBar()
        self._wire_signals()
        self._tone_refresh_timer.timeout.connect(self._refresh_display_from_cached_buffers)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._install_spacebar_filters()
        self._update_combo_width(self.compare_mode_combo)
        self._update_side_combo_widths()
        self._update_active_side_ui()
        self._sync_compare_mode_combo_from_state()
        self._update_sync_nav_visibility()
        self._update_view_mode()
        self._refresh_processed_view()
        self._update_side_transform_controls_enabled()
        self._update_persistent_status_message()
        self._update_global_tone_control_labels()

    def _build_top_toolbar(self) -> QHBoxLayout:
        controls_layout = QHBoxLayout()
        controls_layout.addSpacing(TOP_TOOLBAR_SIDE_GAP)
        tone_controls_container = QWidget(self)
        tone_controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tone_controls_layout = QHBoxLayout(tone_controls_container)
        tone_controls_layout.setContentsMargins(*ZERO_MARGINS)
        tone_controls_layout.setSpacing(TOOLBAR_SPACING)

        tone_controls_layout.addWidget(QLabel("Luminance", tone_controls_container))
        self.luminance_slider = QSlider(Qt.Horizontal, tone_controls_container)
        self.luminance_slider.setRange(0, 200)
        self.luminance_slider.setValue(100)
        self.luminance_slider.setMinimumWidth(TONE_SLIDER_WIDTH)
        self.luminance_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.luminance_slider.setToolTip("Global linear display gain.")
        tone_controls_layout.addWidget(self.luminance_slider)
        self.luminance_value_label = QLabel("1.00x", tone_controls_container)
        self.luminance_value_label.setMinimumWidth(TONE_VALUE_LABEL_MIN_WIDTH)
        tone_controls_layout.addWidget(self.luminance_value_label)

        tone_controls_layout.addWidget(QLabel("Exposure", tone_controls_container))
        self.exposure_slider = QSlider(Qt.Horizontal, tone_controls_container)
        self.exposure_slider.setRange(-80, 80)
        self.exposure_slider.setValue(0)
        self.exposure_slider.setMinimumWidth(TONE_SLIDER_WIDTH)
        self.exposure_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.exposure_slider.setToolTip("Global display exposure in f-stops.")
        tone_controls_layout.addWidget(self.exposure_slider)
        self.exposure_value_label = QLabel("0.00", tone_controls_container)
        self.exposure_value_label.setMinimumWidth(TONE_VALUE_LABEL_MIN_WIDTH)
        tone_controls_layout.addWidget(self.exposure_value_label)
        controls_layout.addWidget(tone_controls_container, 3)
        controls_layout.addSpacing(TOP_TOOLBAR_SIDE_GAP)

        self.compare_mode_combo = QComboBox(self)
        self.compare_mode_combo.addItem("Split", CompareModeData.SPLIT.value)
        self.compare_mode_combo.addItem("Wipe", CompareModeData.WIPE.value)
        self.compare_mode_combo.addItem("Full (A)", CompareModeData.FULL_LEFT.value)
        self.compare_mode_combo.addItem("Full (B)", CompareModeData.FULL_RIGHT.value)
        self.compare_mode_combo.addItem("Diff", CompareModeData.DIFF.value)
        self.compare_mode_combo.setToolTip("Choose how images are compared in the viewer.")
        controls_layout.addWidget(QLabel("Mode", self))
        controls_layout.addWidget(self.compare_mode_combo)

        self.sync_nav_checkbox = QCheckBox("Sync Nav", self)
        self.sync_nav_checkbox.setChecked(True)
        self.sync_nav_checkbox.setToolTip("Keep zoom and pan synchronized between A and B views.")
        controls_layout.addWidget(self.sync_nav_checkbox)
        return controls_layout

    def _build_viewer_area(self) -> QWidget:
        self.viewer_container = QWidget(self)
        self.viewer_layout = QHBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(*ZERO_MARGINS)
        self.viewer_layout.setSpacing(VIEWER_SPACING)

        self.split_container = QWidget(self.viewer_container)
        self.split_layout = QHBoxLayout(self.split_container)
        self.split_layout.setContentsMargins(*ZERO_MARGINS)
        self.split_layout.setSpacing(VIEWER_SPACING)

        self.left_split_view = CompareView(self.split_container)
        self.left_split_view.set_mode("full")
        self.left_split_view.set_active_side("left")
        self.left_split_view.set_placeholder_text("Drop image A here")
        self.left_split_view.set_space_pan_enabled(False)
        self.left_split_view.set_view_transform(
            self._compare_view_state.zoom,
            self._compare_view_state.pan_x,
            self._compare_view_state.pan_y,
        )
        self.right_split_view = CompareView(self.split_container)
        self.right_split_view.set_mode("full")
        self.right_split_view.set_active_side("right")
        self.right_split_view.set_placeholder_text("Drop image B here")
        self.right_split_view.set_space_pan_enabled(False)
        self.right_split_view.set_view_transform(
            self._compare_view_state.zoom,
            self._compare_view_state.pan_x,
            self._compare_view_state.pan_y,
        )
        self.split_layout.addWidget(self.left_split_view, 1)
        self.split_layout.addWidget(self.right_split_view, 1)

        self.compare_view = CompareView(self.viewer_container)
        self.compare_view.set_mode(self._compare_view_state.mode)
        self.compare_view.set_placeholder_text("Drop image A here")
        self.compare_view.set_wipe_position(self._compare_view_state.wipe_position)
        self.compare_view.set_wipe_angle(self._compare_view_state.wipe_angle)
        self.compare_view.set_wipe_top_side(self._compare_view_state.wipe_top_side)
        self.compare_view.set_active_side(self._compare_state.active_side)
        self.compare_view.set_alignment_anchor(self._compare_state.alignment_anchor)
        self.compare_view.set_space_pan_enabled(False)
        self.compare_view.set_view_transform(
            self._compare_view_state.zoom,
            self._compare_view_state.pan_x,
            self._compare_view_state.pan_y,
        )

        self.viewer_layout.addWidget(self.split_container, 1)
        self.viewer_layout.addWidget(self.compare_view, 1)
        return self.viewer_container

    def _build_side_controls_block(self) -> QWidget:
        left_frame_controls = self._build_side_frame_controls("left")
        right_frame_controls = self._build_side_frame_controls("right")

        side_controls_block = QWidget(self)
        side_controls_block_layout = QHBoxLayout(side_controls_block)
        side_controls_block_layout.setContentsMargins(*ZERO_MARGINS)
        side_controls_block_layout.setSpacing(SIDE_BLOCK_SPACING)
        left_side_controls = self._build_side_transform_controls("left")
        right_side_controls = self._build_side_transform_controls("right")
        side_controls_separator = self._build_vertical_separator()
        left_section = QWidget(self)
        left_section_layout = QVBoxLayout(left_section)
        left_section_layout.setContentsMargins(*ZERO_MARGINS)
        left_section_layout.setSpacing(VIEWER_SPACING)
        left_section_layout.addWidget(left_side_controls)

        self.left_bypass_checkbox = QCheckBox("Bypass", self)
        self.left_bypass_checkbox.setToolTip(
            "Show the image without applying the selected OCIO transform."
        )
        self.right_bypass_checkbox = QCheckBox("Bypass", self)
        self.right_bypass_checkbox.setToolTip(
            "Show the image without applying the selected OCIO transform."
        )
        side_slot_width = max(
            self.left_bypass_checkbox.sizeHint().width(),
            self.right_bypass_checkbox.sizeHint().width(),
        )

        left_bypass_container = QWidget(self)
        left_bypass_layout = QGridLayout(left_bypass_container)
        left_bypass_layout.setContentsMargins(*ZERO_MARGINS)
        left_bypass_layout.setSpacing(VIEWER_SPACING)
        left_bypass_layout.setColumnMinimumWidth(0, side_slot_width)
        left_bypass_layout.setColumnMinimumWidth(2, side_slot_width)
        left_bypass_layout.setColumnStretch(1, 1)
        left_bypass_layout.addWidget(self.left_bypass_checkbox, 0, 0, alignment=Qt.AlignLeft)
        left_bypass_layout.addWidget(left_frame_controls, 0, 1, alignment=Qt.AlignCenter)
        self.left_bypass_checkbox.setStyleSheet(
            "margin-left: 1px; margin-top: 1px; margin-bottom: 0px;"
        )

        right_bypass_container = QWidget(self)
        right_bypass_layout = QGridLayout(right_bypass_container)
        right_bypass_layout.setContentsMargins(*ZERO_MARGINS)
        right_bypass_layout.setSpacing(VIEWER_SPACING)
        right_bypass_layout.setColumnMinimumWidth(0, side_slot_width)
        right_bypass_layout.setColumnMinimumWidth(2, side_slot_width)
        right_bypass_layout.setColumnStretch(1, 1)
        right_bypass_layout.addWidget(right_frame_controls, 0, 1, alignment=Qt.AlignCenter)
        right_bypass_layout.addWidget(self.right_bypass_checkbox, 0, 2, alignment=Qt.AlignRight)

        left_section_layout.addWidget(left_bypass_container)

        right_section = QWidget(self)
        right_section_layout = QVBoxLayout(right_section)
        right_section_layout.setContentsMargins(*ZERO_MARGINS)
        right_section_layout.setSpacing(VIEWER_SPACING)
        right_section_layout.addWidget(right_side_controls)
        right_section_layout.addWidget(right_bypass_container)

        section_min_width = max(left_section.sizeHint().width(), right_section.sizeHint().width())
        left_section.setMinimumWidth(section_min_width)
        right_section.setMinimumWidth(section_min_width)
        left_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        side_controls_block_layout.addWidget(left_section, 1)
        side_controls_block_layout.addWidget(side_controls_separator)
        side_controls_block_layout.addWidget(right_section, 1)
        return side_controls_block

    def _build_footer_hud(self) -> QHBoxLayout:
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(*ZERO_MARGINS)
        footer_layout.setSpacing(FOOTER_SPACING)

        self._hud_resolution_label = QLabel("-x-", self)
        self._hud_coord_label = QLabel("x=- y=-", self)
        self._hud_r_label = QLabel("----", self)
        self._hud_g_label = QLabel("----", self)
        self._hud_b_label = QLabel("----", self)
        self._hud_r_label.setStyleSheet("color: #ff5a5a;")
        self._hud_g_label.setStyleSheet("color: #57d957;")
        self._hud_b_label.setStyleSheet("color: #5aa0ff;")

        self._populate_footer_hud_slots(footer_layout)
        footer_layout.addStretch(1)
        return footer_layout

    def _populate_footer_hud_slots(self, layout: QHBoxLayout) -> None:
        """Populate footer HUD slots in a fixed order."""
        layout.addWidget(self._hud_resolution_label)
        layout.addWidget(self._build_inline_separator())
        layout.addWidget(self._hud_coord_label)
        layout.addWidget(self._build_inline_separator())
        layout.addWidget(self._hud_r_label)
        layout.addWidget(self._hud_g_label)
        layout.addWidget(self._hud_b_label)

    def _build_vertical_separator(self) -> QFrame:
        separator = QFrame(self)
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def _build_inline_separator(self) -> QLabel:
        return QLabel("|", self)

    def _apply_app_icon(self) -> None:
        icon_path = (
            "assets/icons/app_icon.ico"
            if platform.system().lower() == "windows"
            else "assets/icons/app_icon.png"
        )
        icon = self._load_ui_icon(icon_path)
        if icon is None and icon_path.endswith(".ico"):
            icon = self._load_ui_icon("assets/icons/app_icon.png")
        if icon is None:
            return
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        open_image_a_action = QAction("Open Image A...", self)
        open_image_a_action.triggered.connect(
            lambda: self._open_image_for_side("left")
        )
        file_menu.addAction(open_image_a_action)

        open_image_b_action = QAction("Open Image B...", self)
        open_image_b_action.triggered.connect(
            lambda: self._open_image_for_side("right")
        )
        file_menu.addAction(open_image_b_action)

        open_config_action = QAction("Open OCIO Config...", self)
        open_config_action.triggered.connect(self._on_browse_config_clicked)
        file_menu.addAction(open_config_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menu_bar.addMenu("View")
        context_vars_action = self._context_variables_dock.toggleViewAction()
        context_vars_action.setText("OCIO Context Variables")
        view_menu.addAction(context_vars_action)

        background_menu = view_menu.addMenu("Background")
        self._viewer_background_group = QActionGroup(self)
        self._viewer_background_group.setExclusive(True)

        presets: list[tuple[str, QColor]] = [
            ("Black", QColor(0, 0, 0)),
            ("Dark Gray", QColor(36, 36, 36)),
            ("Mid Gray", QColor(64, 64, 64)),
            ("Light Gray", QColor(112, 112, 112)),
        ]
        for label, color in presets:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, selected_color=color: self._set_viewer_background(
                    selected_color
                )
            )
            self._viewer_background_group.addAction(action)
            background_menu.addAction(action)
            if label == "Black":
                action.setChecked(True)

        help_menu = menu_bar.addMenu("Help")
        self._hotkeys_action = QAction("Hotkeys", self)
        self._hotkeys_action.triggered.connect(self._show_hotkeys_dialog)
        help_menu.addAction(self._hotkeys_action)

        about_action = QAction("About Prism Viewer", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _open_image_for_side(self, side: ViewerSide) -> None:
        default_dir = self._default_config_directory()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Image for {side_label(side)}",
            str(default_dir),
            "Images (*.exr *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.dpx);;All Files (*)",
        )
        if not file_path:
            self._show_temporary_status(
                f"Open image {side_label(side)} cancelled", timeout_ms=1500
            )
            return
        self._load_dropped_image(file_path, side)

    def _show_hotkeys_dialog(self) -> None:
        if self._hotkeys_dialog is not None and self._hotkeys_dialog.isVisible():
            self._hotkeys_dialog.close()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Prism Viewer Hotkeys")
        dialog.setModal(False)
        dialog.setMinimumWidth(420)
        self._hotkeys_dialog = dialog
        dialog.destroyed.connect(self._on_hotkeys_dialog_destroyed)

        layout = QGridLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(8)

        hotkeys: list[tuple[str, str]] = [
            ("Fit to Window", "F"),
            ("Show Hotkeys", "H"),
            ("Toggle OCIO Context Variables", "E"),
            ("Switch A/B", "Tab"),
            ("Next Frame", "Right or PageDown"),
            ("Previous Frame", "Left or PageUp"),
            ("Input/Output Selection", "Up/Down"),
            ("Channel Solo Red", "R"),
            ("Channel Solo Green", "G"),
            ("Channel Solo Blue", "B"),
            ("Restore RGB/Luminance/Exposure", "Z"),
            ("Compare Mode Split/Wipe/Full(A)/Full(B)/Diff", "1 / 2 / 3 / 4 / 5"),
            ("Pan Drag", "Right Click + Drag"),
        ]

        for row, (action_text, shortcut_text) in enumerate(hotkeys):
            action_label = QLabel(action_text, dialog)
            shortcut_label = QLabel(shortcut_text, dialog)
            shortcut_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(action_label, row, 0)
            layout.addWidget(shortcut_label, row, 1)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_hotkeys_dialog_destroyed(self) -> None:
        self._hotkeys_dialog = None

    def _show_about_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("About Prism Viewer")
        dialog.setModal(True)
        dialog.setMinimumWidth(420)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        banner = self._load_ui_pixmap("assets/images/about_banner.png")
        if banner is not None:
            banner_label = QLabel(dialog)
            scaled_banner = banner.scaledToWidth(388, Qt.SmoothTransformation)
            banner_label.setPixmap(scaled_banner)
            banner_label.setAlignment(Qt.AlignCenter)
            root.addWidget(banner_label)

        title_label = QLabel("Prism Viewer", dialog)
        title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title_label)

        for label, value in self._about_identity_items():
            row = QHBoxLayout()
            key_label = QLabel(f"{label}:", dialog)
            key_label.setMinimumWidth(80)
            value_label = QLabel(value, dialog)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(key_label)
            row.addWidget(value_label, 1)
            root.addLayout(row)

        actions_row = QHBoxLayout()
        actions_row.addStretch(1)

        diagnostics_button = QPushButton("Show Diagnostics...", dialog)
        diagnostics_button.clicked.connect(self._show_diagnostics_dialog)
        actions_row.addWidget(diagnostics_button)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        actions_row.addWidget(close_button)

        root.addLayout(actions_row)
        dialog.exec()

    def _show_diagnostics_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Diagnostics")
        dialog.setModal(True)
        dialog.setMinimumWidth(520)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title_label = QLabel("Diagnostics", dialog)
        title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        for row, (key, value) in enumerate(self._diagnostics_items()):
            key_label = QLabel(f"{key}:", dialog)
            value_label = QLabel(value, dialog)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            grid.addWidget(key_label, row, 0, alignment=Qt.AlignTop)
            grid.addWidget(value_label, row, 1)
        root.addLayout(grid)

        feedback_label = QLabel("", dialog)
        root.addWidget(feedback_label)

        actions_row = QHBoxLayout()
        actions_row.addStretch(1)

        copy_button = QPushButton("Copy Diagnostics", dialog)

        def _copy_diagnostics() -> None:
            clipboard = QApplication.clipboard()
            if clipboard is None:
                feedback_label.setText("Clipboard unavailable")
                return
            clipboard.setText(self._diagnostics_text())
            feedback_label.setText("Diagnostics copied to clipboard")

        copy_button.clicked.connect(_copy_diagnostics)
        actions_row.addWidget(copy_button)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        actions_row.addWidget(close_button)

        root.addLayout(actions_row)
        dialog.exec()

    def _build_side_transform_controls(self, side: ViewerSide) -> QWidget:
        side_label = "A" if side == "left" else "B"
        container = QWidget(self)
        row = QHBoxLayout(container)
        row.setContentsMargins(*ZERO_MARGINS)
        row.setSpacing(10)
        row.addWidget(QLabel(side_label, container))

        input_combo = QComboBox(container)
        input_combo.setPlaceholderText("Input colorspace")
        input_combo.setToolTip("Colorspace the image is assumed to be in before processing.")
        input_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        input_pair = QWidget(container)
        input_pair_layout = QHBoxLayout(input_pair)
        input_pair_layout.setContentsMargins(*ZERO_MARGINS)
        input_pair_layout.setSpacing(3)
        input_pair_layout.addWidget(QLabel("Input", input_pair))
        input_pair_layout.addWidget(input_combo, 1)
        row.addWidget(input_pair, 1)

        output_combo = QComboBox(container)
        output_combo.setPlaceholderText("Output colorspace")
        output_combo.setToolTip("Target colorspace used for display or comparison.")
        output_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        output_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        output_pair = QWidget(container)
        output_pair_layout = QHBoxLayout(output_pair)
        output_pair_layout.setContentsMargins(*ZERO_MARGINS)
        output_pair_layout.setSpacing(3)
        output_pair_layout.addWidget(QLabel("Output", output_pair))
        output_pair_layout.addWidget(output_combo, 1)
        row.addWidget(output_pair, 1)

        looks_combo = QComboBox(container)
        looks_combo.setPlaceholderText("Looks")
        looks_combo.setToolTip("Optional OCIO look applied during the transform.")
        looks_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        looks_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        looks_pair = QWidget(container)
        looks_pair_layout = QHBoxLayout(looks_pair)
        looks_pair_layout.setContentsMargins(*ZERO_MARGINS)
        looks_pair_layout.setSpacing(3)
        looks_pair_layout.addWidget(QLabel("Looks", looks_pair))
        looks_pair_layout.addWidget(looks_combo, 1)
        row.addWidget(looks_pair, 1)

        if side == "left":
            self.left_input_colorspace_combo = input_combo
            self.left_output_colorspace_combo = output_combo
            self.left_looks_combo = looks_combo
        else:
            self.right_input_colorspace_combo = input_combo
            self.right_output_colorspace_combo = output_combo
            self.right_looks_combo = looks_combo

        return container

    def _build_side_frame_controls(self, side: ViewerSide) -> QWidget:
        container = QWidget(self)
        row = QHBoxLayout(container)
        row.setContentsMargins(*ZERO_MARGINS)
        row.setSpacing(VIEWER_SPACING)
        row.addWidget(QLabel("Frame", container))

        prev_button = QPushButton("<", container)
        prev_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        row.addWidget(prev_button)

        frame_index_edit = QLineEdit("-", container)
        frame_index_edit.setAlignment(Qt.AlignCenter)
        frame_index_edit.setFixedWidth(FRAME_EDIT_WIDTH)
        frame_index_edit.setValidator(QIntValidator(0, 999999999, self))
        row.addWidget(frame_index_edit)

        next_button = QPushButton(">", container)
        next_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        row.addWidget(next_button)

        if side == "left":
            self.left_prev_frame_button = prev_button
            self.left_next_frame_button = next_button
            self.left_frame_index_edit = frame_index_edit
        else:
            self.right_prev_frame_button = prev_button
            self.right_next_frame_button = next_button
            self.right_frame_index_edit = frame_index_edit
        return container

    def _update_side_combo_widths(self) -> None:
        for combo in (
            self.left_input_colorspace_combo,
            self.left_output_colorspace_combo,
            self.left_looks_combo,
            self.right_input_colorspace_combo,
            self.right_output_colorspace_combo,
            self.right_looks_combo,
        ):
            self._set_expandable_combo_min_width(combo)

    def _set_expandable_combo_min_width(self, combo: QComboBox) -> None:
        """Set a content-based minimum width while allowing horizontal expansion."""
        font_metrics = combo.fontMetrics()
        texts = [combo.itemText(i) for i in range(combo.count())]
        if combo.placeholderText():
            texts.append(combo.placeholderText())
        if not texts:
            texts = [" "]
        longest_text_width = max(font_metrics.horizontalAdvance(text) for text in texts)
        horizontal_padding = 48  # arrow + frame + margins
        combo.setMinimumWidth(longest_text_width + horizontal_padding)
        combo.setMaximumWidth(16777215)

    def _wire_signals(self) -> None:
        self.compare_mode_combo.currentIndexChanged.connect(self._on_compare_mode_changed)
        self.sync_nav_checkbox.toggled.connect(self._on_sync_nav_toggled)
        self.left_prev_frame_button.clicked.connect(lambda: self._on_prev_frame_clicked("left"))
        self.left_next_frame_button.clicked.connect(lambda: self._on_next_frame_clicked("left"))
        self.left_frame_index_edit.editingFinished.connect(
            lambda: self._on_frame_index_edit_finished("left")
        )
        self.right_prev_frame_button.clicked.connect(lambda: self._on_prev_frame_clicked("right"))
        self.right_next_frame_button.clicked.connect(lambda: self._on_next_frame_clicked("right"))
        self.right_frame_index_edit.editingFinished.connect(
            lambda: self._on_frame_index_edit_finished("right")
        )
        self.left_split_view.side_selected.connect(self._set_active_side)
        self.right_split_view.side_selected.connect(self._set_active_side)
        self.compare_view.side_selected.connect(self._set_active_side)
        self.compare_view.wipe_changed.connect(self._on_wipe_changed)
        self.compare_view.wipe_angle_changed.connect(self._on_wipe_angle_changed)
        self.compare_view.view_changed.connect(self._on_compare_view_changed)
        self.compare_view.hover_changed.connect(self._on_view_hover_changed)
        self.left_split_view.view_changed.connect(self._on_compare_view_changed)
        self.right_split_view.view_changed.connect(self._on_compare_view_changed)
        self.left_split_view.hover_changed.connect(self._on_view_hover_changed)
        self.right_split_view.hover_changed.connect(self._on_view_hover_changed)
        self.left_input_colorspace_combo.currentIndexChanged.connect(
            lambda index: self._on_side_input_colorspace_changed("left", index)
        )
        self.left_output_colorspace_combo.currentIndexChanged.connect(
            lambda index: self._on_side_output_colorspace_changed("left", index)
        )
        self.left_looks_combo.currentIndexChanged.connect(
            lambda index: self._on_side_look_changed("left", index)
        )
        self.left_bypass_checkbox.toggled.connect(
            lambda checked: self._on_side_bypass_toggled("left", checked)
        )
        self.right_input_colorspace_combo.currentIndexChanged.connect(
            lambda index: self._on_side_input_colorspace_changed("right", index)
        )
        self.right_output_colorspace_combo.currentIndexChanged.connect(
            lambda index: self._on_side_output_colorspace_changed("right", index)
        )
        self.right_looks_combo.currentIndexChanged.connect(
            lambda index: self._on_side_look_changed("right", index)
        )
        self.right_bypass_checkbox.toggled.connect(
            lambda checked: self._on_side_bypass_toggled("right", checked)
        )
        self.luminance_slider.valueChanged.connect(self._on_luminance_slider_changed)
        self.exposure_slider.valueChanged.connect(self._on_exposure_slider_changed)
        self.luminance_slider.sliderReleased.connect(self._on_tone_slider_released)
        self.exposure_slider.sliderReleased.connect(self._on_tone_slider_released)
        self._context_variables_dock.panel.context_values_changed.connect(
            self._on_context_values_changed
        )

    def _on_browse_config_clicked(self) -> None:
        default_dir = self._default_config_directory()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OCIO Config",
            str(default_dir),
            "OCIO Config (*.ocio);;All Files (*)",
        )
        if not file_path:
            self._show_temporary_status("Config selection cancelled", timeout_ms=1500)
            return

        self._load_config(file_path)

    def _on_side_input_colorspace_changed(self, side: ViewerSide, index: int) -> None:
        if index < 0:
            return
        combo = (
            self.left_input_colorspace_combo
            if side == "left"
            else self.right_input_colorspace_combo
        )
        self._compare_state.panel(side).input_colorspace = combo.currentText().strip()
        self._refresh_side_with_active_status(side)

    def _on_side_output_colorspace_changed(self, side: ViewerSide, index: int) -> None:
        if index < 0:
            return
        combo = (
            self.left_output_colorspace_combo
            if side == "left"
            else self.right_output_colorspace_combo
        )
        self._compare_state.panel(side).output_colorspace = combo.currentText().strip()
        self._refresh_side_with_active_status(side)

    def _on_side_look_changed(self, side: ViewerSide, index: int) -> None:
        if index < 0:
            return
        combo = self.left_looks_combo if side == "left" else self.right_looks_combo
        selected_look = combo.currentText().strip()
        self._compare_state.panel(side).look = (
            selected_look if selected_look and selected_look != "(None)" else None
        )
        self._refresh_side_with_active_status(side)

    def _on_side_bypass_toggled(self, side: ViewerSide, checked: bool) -> None:
        self._compare_state.panel(side).bypass = checked
        self._refresh_side_with_active_status(side)

    def _on_luminance_slider_changed(self, value: int) -> None:
        self._global_luminance = max(0.0, value / 100.0)
        self._update_global_tone_control_labels()
        self._schedule_tone_refresh()

    def _on_exposure_slider_changed(self, value: int) -> None:
        self._global_exposure_stops = value / 10.0
        self._update_global_tone_control_labels()
        self._schedule_tone_refresh()

    def _on_tone_slider_released(self) -> None:
        self._tone_refresh_timer.stop()
        self._refresh_display_from_cached_buffers()

    def _reset_viewer_adjustments(self) -> None:
        self.luminance_slider.setValue(100)
        self.exposure_slider.setValue(0)
        self._set_channel_view(ViewerChannel.RGB.value)

    def _on_compare_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        mode_data = self.compare_mode_combo.itemData(index)
        if not isinstance(mode_data, str):
            return
        if not self._apply_compare_mode_data(mode_data):
            return
        self._update_sync_nav_visibility()
        self._clear_inspector_status()
        self._refresh_processed_view()
        self._set_active_side_status(self._compare_state.active_side)

    def _on_context_values_changed(self, values: dict[str, str]) -> None:
        self._ocio_context_values = dict(values)
        self._refresh_processed_view()

    def _on_wipe_changed(self, wipe_value: float) -> None:
        self._compare_view_state.wipe_position = wipe_value

    def _on_wipe_angle_changed(self, wipe_angle: float) -> None:
        self._compare_view_state.wipe_angle = wipe_angle

    def _on_sync_nav_toggled(self, checked: bool) -> None:
        self._sync_nav_enabled = checked
        self._clear_inspector_status()
        self.compare_view.set_wipe_unsynced_nav_enabled(not checked)
        if checked:
            active_side = self._compare_state.active_side
            zoom, pan_x, pan_y = self._split_view_transforms[active_side]
            self._split_view_transforms["left"] = (zoom, pan_x, pan_y)
            self._split_view_transforms["right"] = (zoom, pan_x, pan_y)
            self._set_compare_view_transform(zoom, pan_x, pan_y)
        else:
            active_side = self._compare_state.active_side
            self._set_compare_view_transform(*self._split_view_transforms[active_side])
        self._apply_split_view_transforms()

    def _on_compare_view_changed(self, zoom: float, pan_x: float, pan_y: float) -> None:
        """Propagate view transform updates between compare and split views."""
        sender = self.sender()
        if sender is self.compare_view:
            self._set_compare_view_transform(zoom, pan_x, pan_y)
            if self._sync_nav_enabled:
                self._split_view_transforms["left"] = (zoom, pan_x, pan_y)
                self._split_view_transforms["right"] = (zoom, pan_x, pan_y)
                self._apply_split_view_transforms()
            else:
                active_side = self._compare_state.active_side
                self._split_view_transforms[active_side] = (zoom, pan_x, pan_y)
            return

        if sender is self.left_split_view:
            side: ViewerSide = "left"
        elif sender is self.right_split_view:
            side = "right"
        else:
            return

        self._split_view_transforms[side] = (zoom, pan_x, pan_y)
        if self._sync_nav_enabled:
            other_side: ViewerSide = "right" if side == "left" else "left"
            self._split_view_transforms[other_side] = (zoom, pan_x, pan_y)
            self._set_compare_view_transform(zoom, pan_x, pan_y)
            self._apply_split_view_transforms()
            return

        if self._compare_state.active_side == side:
            self._set_compare_view_transform(zoom, pan_x, pan_y)

    def _on_prev_frame_clicked(self, side: ViewerSide) -> None:
        self._step_source_frame(side, -1)

    def _on_next_frame_clicked(self, side: ViewerSide) -> None:
        self._step_source_frame(side, 1)

    def _on_frame_index_edit_finished(self, side: ViewerSide) -> None:
        panel = self._compare_state.panel(side)
        if panel.loaded_source is None:
            self._update_frame_controls()
            return

        frame_edit = self.left_frame_index_edit if side == "left" else self.right_frame_index_edit
        text = frame_edit.text().strip()
        if not text:
            self._update_frame_controls()
            return
        try:
            requested_number = int(text)
        except ValueError:
            self._update_frame_controls()
            return

        target_index = self._resolve_frame_number_to_index(panel, requested_number)
        if target_index == panel.current_frame_index:
            self._update_frame_controls()
            return
        self._set_source_frame(side, target_index)

    def _step_active_source_frame(self, delta: int) -> None:
        side = self._compare_state.active_side
        self._step_source_frame(side, delta)

    def _step_source_frame(self, side: ViewerSide, delta: int) -> None:
        panel = self._compare_state.panel(side)
        if panel.loaded_source is None or panel.frame_count <= 1:
            return
        self._clear_inspector_status()
        target_index = panel.current_frame_index + delta
        self._set_source_frame(side, target_index)

    def _set_active_source_frame(self, target_index: int) -> None:
        side = self._compare_state.active_side
        self._set_source_frame(side, target_index)

    def _set_source_frame(self, side: ViewerSide, target_index: int) -> None:
        panel = self._compare_state.panel(side)
        loaded = self._load_panel_frame(side, panel, target_index)
        if not loaded:
            return
        self._refresh_side_with_active_status(side)
        self._update_frame_controls()

    def _on_view_hover_changed(
        self, u: float, v: float, inside: bool, side_hint: str
    ) -> None:
        sender = self.sender()
        if isinstance(sender, QWidget) and not sender.isVisible():
            return

        if not inside:
            self._hovered_view_side = None
            self._clear_hud_sample()
            return

        sample_a = self._sample_qimage_at_uv(self._display_images["left"], u, v)
        sample_b = self._sample_qimage_at_uv(self._display_images["right"], u, v)

        hovered_side: ViewerSide | None = None
        if side_hint == "left":
            hovered_side = "left"
        elif side_hint == "right":
            hovered_side = "right"
        elif sender is self.left_split_view:
            hovered_side = "left"
        elif sender is self.right_split_view:
            hovered_side = "right"
        elif sample_a is not None:
            hovered_side = "left"
        elif sample_b is not None:
            hovered_side = "right"

        if hovered_side is None:
            self._hovered_view_side = None
            self._clear_hud_sample()
            return

        active_sample = sample_a if hovered_side == "left" else sample_b
        if active_sample is None:
            self._hovered_view_side = None
            self._clear_hud_sample()
            return
        self._hovered_view_side = hovered_side

        image = self._display_images[hovered_side]
        width = image.width() if image is not None else 0
        height = image.height() if image is not None else 0
        x, y, rgb = active_sample
        self._set_hud_sample(
            width=width,
            height=height,
            x=x,
            y=y,
            r=rgb[0],
            g=rgb[1],
            b=rgb[2],
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag-enter for local-file drops used by image/config loading.

        Args:
            event: Qt drag-enter event.
        """
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        self._clear_drop_target_highlight()
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Update drop guidance while dragging files over viewer regions.

        Args:
            event: Qt drag-move event.
        """
        if not event.mimeData().hasUrls():
            self._clear_drop_target_highlight()
            event.ignore()
            return

        self._update_drop_target_highlight(self._side_for_drop_position(event.position()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._clear_drop_target_highlight()
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped local files and route them to load handlers.

        Args:
            event: Qt drop event.
        """
        target_side = self._side_for_drop_position(event.position())
        self._clear_drop_target_highlight()
        local_files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if local_files:
            self._handle_dropped_files(local_files, target_side)
            event.acceptProposedAction()
            return
        self._show_temporary_status("Drop failed: no local file detected")
        event.ignore()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Refresh rendered pixmaps after window resize.

        Args:
            event: Qt resize event.
        """
        super().resizeEvent(event)
        self._rescale_visible_pixmaps()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle window-level keyboard shortcuts and navigation controls.

        Args:
            event: Qt key-press event.
        """
        key = event.key()

        if self._is_text_input_focus_active():
            super().keyPressEvent(event)
            return

        if event.modifiers() != Qt.NoModifier:
            super().keyPressEvent(event)
            return

        if key in {Qt.Key_Up, Qt.Key_Down}:
            focused_combo = self._focused_colorspace_combo()
            if focused_combo is not None and not focused_combo.view().isVisible():
                delta = -1 if key == Qt.Key_Up else 1
                self._step_combo_selection(focused_combo, delta)
                event.accept()
                return

        if key in {Qt.Key_PageUp, Qt.Key_PageDown} and not event.isAutoRepeat():
            delta = -1 if key == Qt.Key_PageUp else 1
            self._step_active_source_frame(delta)
            event.accept()
            return

        if key in {Qt.Key_Left, Qt.Key_Right} and not event.isAutoRepeat():
            focused_combo = self.focusWidget()
            if isinstance(focused_combo, QComboBox) and focused_combo.view().isVisible():
                super().keyPressEvent(event)
                return

            delta = -1 if key == Qt.Key_Left else 1
            self._step_active_source_frame(delta)
            event.accept()
            return

        if key == Qt.Key_Tab and not event.isAutoRepeat():
            focused_combo = self.focusWidget()
            if isinstance(focused_combo, QComboBox) and focused_combo.view().isVisible():
                super().keyPressEvent(event)
                return

            next_side: ViewerSide = (
                "right" if self._compare_state.active_side == "left" else "left"
            )
            self._set_active_side(next_side)
            event.accept()
            return

        if key in {Qt.Key_R, Qt.Key_G, Qt.Key_B, Qt.Key_Z} and not event.isAutoRepeat():
            focused_combo = self.focusWidget()
            if isinstance(focused_combo, QComboBox) and focused_combo.view().isVisible():
                super().keyPressEvent(event)
                return

            if key == Qt.Key_R:
                self._set_channel_view(
                    ViewerChannel.RGB.value
                    if self._channel_view == ViewerChannel.RED.value
                    else ViewerChannel.RED.value
                )
            elif key == Qt.Key_G:
                self._set_channel_view(
                    ViewerChannel.RGB.value
                    if self._channel_view == ViewerChannel.GREEN.value
                    else ViewerChannel.GREEN.value
                )
            elif key == Qt.Key_B:
                self._set_channel_view(
                    ViewerChannel.RGB.value
                    if self._channel_view == ViewerChannel.BLUE.value
                    else ViewerChannel.BLUE.value
                )
            else:
                self._reset_viewer_adjustments()
            event.accept()
            return

        if key == Qt.Key_F and not event.isAutoRepeat():
            self._frame_view()
            event.accept()
            return

        if key == Qt.Key_H and not event.isAutoRepeat():
            self._show_hotkeys_dialog()
            event.accept()
            return

        if key == Qt.Key_E and not event.isAutoRepeat():
            self._context_variables_dock.setVisible(not self._context_variables_dock.isVisible())
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key-release events.

        Args:
            event: Qt key-release event.
        """
        super().keyReleaseEvent(event)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        """Intercept global key events for shared viewer shortcuts.

        Args:
            watched: Qt object currently receiving the event.
            event: Qt event instance.

        Returns:
            ``True`` when the event is handled and consumed, otherwise the
            superclass filter result.
        """
        if isinstance(watched, QWidget) and watched.window() is not self:
            return super().eventFilter(watched, event)

        if event.type() == QEvent.KeyPress and self._is_text_input_focus_active():
            return super().eventFilter(watched, event)

        if event.type() == QEvent.KeyPress and event.modifiers() == Qt.NoModifier:
            key = event.key()
            if key == Qt.Key_F and not event.isAutoRepeat():
                self._frame_view()
                event.accept()
                return True

            if key == Qt.Key_H and not event.isAutoRepeat():
                self._show_hotkeys_dialog()
                event.accept()
                return True

            if key == Qt.Key_E and not event.isAutoRepeat():
                self._context_variables_dock.setVisible(
                    not self._context_variables_dock.isVisible()
                )
                event.accept()
                return True

            if key in {Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5} and not event.isAutoRepeat():
                mode_by_key: dict[int, CompareModeData] = {
                    Qt.Key_1: CompareModeData.SPLIT,
                    Qt.Key_2: CompareModeData.WIPE,
                    Qt.Key_3: CompareModeData.FULL_LEFT,
                    Qt.Key_4: CompareModeData.FULL_RIGHT,
                    Qt.Key_5: CompareModeData.DIFF,
                }
                mode_data = mode_by_key[key].value
                index = self.compare_mode_combo.findData(mode_data)
                if index >= 0:
                    self.compare_mode_combo.setCurrentIndex(index)
                event.accept()
                return True

            if key in {Qt.Key_R, Qt.Key_G, Qt.Key_B, Qt.Key_Z} and not event.isAutoRepeat():
                focused_combo = self.focusWidget()
                if isinstance(focused_combo, QComboBox) and focused_combo.view().isVisible():
                    return super().eventFilter(watched, event)

                if key == Qt.Key_R:
                    self._set_channel_view(
                        ViewerChannel.RGB.value
                        if self._channel_view == ViewerChannel.RED.value
                        else ViewerChannel.RED.value
                    )
                elif key == Qt.Key_G:
                    self._set_channel_view(
                        ViewerChannel.RGB.value
                        if self._channel_view == ViewerChannel.GREEN.value
                        else ViewerChannel.GREEN.value
                    )
                elif key == Qt.Key_B:
                    self._set_channel_view(
                        ViewerChannel.RGB.value
                        if self._channel_view == ViewerChannel.BLUE.value
                        else ViewerChannel.BLUE.value
                    )
                else:
                    self._reset_viewer_adjustments()
                event.accept()
                return True

            # Prevent accidental mode changes from combo type-to-select (W/S/D).
            if key in {Qt.Key_W, Qt.Key_S, Qt.Key_D}:
                focused = self.focusWidget()
                if focused is self.compare_mode_combo and not self.compare_mode_combo.view().isVisible():
                    event.accept()
                    return True

        if event.type() == QEvent.KeyPress and event.key() in {Qt.Key_Left, Qt.Key_Right}:
            if event.isAutoRepeat() or event.modifiers() != Qt.NoModifier:
                event.accept()
                return True
            focused = self.focusWidget()
            if isinstance(focused, QComboBox) and focused.view().isVisible():
                return super().eventFilter(watched, event)
            delta = -1 if event.key() == Qt.Key_Left else 1
            self._step_active_source_frame(delta)
            event.accept()
            return True

        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Tab:
            if event.isAutoRepeat() or event.modifiers() != Qt.NoModifier:
                event.accept()
                return True
            focused = self.focusWidget()
            if isinstance(focused, QComboBox) and focused.view().isVisible():
                return super().eventFilter(watched, event)
            next_side: ViewerSide = (
                "right" if self._compare_state.active_side == "left" else "left"
            )
            self._set_active_side(next_side)
            event.accept()
            return True

        return super().eventFilter(watched, event)

    def _load_config(self, file_path: str) -> None:
        try:
            ocio_config = load_ocio_config(file_path)
            colorspaces = list_colorspaces(ocio_config)
            looks = list_looks(ocio_config)
        except Exception as exc:
            self._show_temporary_status(f"Config load failed: {exc}")
            return

        if not colorspaces:
            self._show_temporary_status("Config load failed: no colorspaces found")
            return

        self._loaded_ocio_config = ocio_config
        self._loaded_ocio_config_path = file_path
        self._ocio_context_values = list_context_variables(ocio_config)
        self._set_colorspace_items(colorspaces)
        self._set_look_items(looks)
        self._reconcile_panel_transform_settings(colorspaces, looks)
        self._sync_context_variables_dock()
        self._refresh_processed_view()

    def _sync_context_variables_dock(self) -> None:
        if self._loaded_ocio_config is None:
            self._context_variables_dock.clear()
            return
        if not self._ocio_context_values:
            self._context_variables_dock.show_no_variables_message()
            return
        self._context_variables_dock.set_variables(self._ocio_context_values)

    def _about_identity_items(self) -> list[tuple[str, str]]:
        return [
            ("Application", "Prism Viewer"),
            ("Version", prism_version),
            ("Purpose", "OCIO still-image viewer for A/B comparison"),
            ("License", "MIT"),
        ]

    def _diagnostics_items(self) -> list[tuple[str, str]]:
        return [
            ("Prism Version", prism_version),
            ("Python", platform.python_version()),
            ("PySide6", pyside_version),
            ("Qt", qVersion()),
            ("NumPy", np.__version__),
            ("OpenImageIO", self._open_image_io_version()),
            ("OpenColorIO", self._open_color_io_version()),
            ("OCIO Config", self._active_ocio_config_label()),
            ("Platform", platform.platform()),
        ]

    def _diagnostics_text(self) -> str:
        lines = ["Prism Viewer Diagnostics"]
        for key, value in self._diagnostics_items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _active_ocio_config_label(self) -> str:
        if not self._loaded_ocio_config_path:
            return "None loaded"
        return self._loaded_ocio_config_path

    def _open_color_io_version(self) -> str:
        try:
            import PyOpenColorIO as ocio
        except Exception:
            return "Unavailable"

        try:
            version = ocio.GetVersion()
        except Exception:
            return "Unknown"
        return str(version) if version else "Unknown"

    def _open_image_io_version(self) -> str:
        try:
            import OpenImageIO as oiio
        except Exception:
            return "Unavailable"

        for attr in ("VERSION_STRING", "__version__"):
            value = getattr(oiio, attr, None)
            if value:
                return str(value)

        try:
            version = oiio.get_string_attribute("library_version")
        except Exception:
            return "Unknown"
        return str(version) if version else "Unknown"

    def _load_dropped_image(self, image_path: str, requested_side: ViewerSide | None) -> None:
        side = self._resolve_drop_target_side(requested_side)
        panel = self._compare_state.panel(side)
        was_loaded_before = panel.loaded_image_data is not None
        path = Path(image_path)
        try:
            source = create_source_from_path(path)
        except Exception as exc:
            self._show_temporary_status(f"{side_label(side)} image load failed: {exc}")
            return

        try:
            frame_count = source.frame_count
        except Exception as exc:
            self._show_temporary_status(f"{side_label(side)} image load failed: {exc}")
            return

        panel.image_path = image_path
        panel.loaded_source = source
        panel.frame_count = frame_count
        panel.canvas_policy = "native"
        panel.scale_policy = "fit"
        panel.offset_x = 0
        panel.offset_y = 0
        loaded = self._load_panel_frame(side, panel, 0)
        if not loaded:
            return
        if self._compare_view_state.mode == "full":
            if not was_loaded_before:
                self._compare_state.active_side = side
        else:
            self._compare_state.active_side = side
        self._compare_state.alignment_anchor = AlignmentAnchor.VIEWPORT.value
        self._sync_controls_from_panel(side)
        self._update_active_side_ui()
        self._sync_compare_mode_combo_from_state()
        self._refresh_processed_view()
        self._split_view_transforms["left"] = (1.0, 0.0, 0.0)
        self._split_view_transforms["right"] = (1.0, 0.0, 0.0)
        self._set_compare_view_transform(1.0, 0.0, 0.0)
        self._apply_split_view_transforms()
        self._update_frame_controls()
        if self._compare_view_state.mode == "full":
            self.compare_mode_combo.setFocus(Qt.TabFocusReason)

    def _load_panel_frame(self, side: ViewerSide, panel: PanelState, frame_index: int) -> bool:
        source = panel.loaded_source
        if source is None:
            return False
        try:
            request_token = self._frame_service.next_request_token(source.source_id)
            frame_result = self._frame_service.get_frame(
                source, frame_index=frame_index, token=request_token
            )
        except Exception as exc:
            self._show_temporary_status(
                f"{side_label(side)} source frame load failed: {exc}"
            )
            return False
        if frame_result.stale:
            self._show_temporary_status(
                f"{side_label(side)} frame load ignored (stale request)"
            )
            return False
        panel.current_frame_index = frame_result.frame_index
        panel.frame_count = source.frame_count
        panel.loaded_image_data = qimage_to_float_rgb(frame_result.image)
        return True

    def _handle_dropped_files(
        self, file_paths: list[str], requested_side: ViewerSide | None = None
    ) -> None:
        first_side = requested_side
        for index, file_path in enumerate(file_paths):
            self._handle_dropped_file(file_path, first_side if index == 0 else None)

    def _handle_dropped_file(
        self, file_path: str, requested_side: ViewerSide | None = None
    ) -> None:
        path = Path(file_path)
        if not path.is_file():
            self._show_temporary_status(f"Drop failed: not a file: {file_path}")
            return
        if path.suffix.lower() == ".ocio":
            self._load_config(file_path)
            return
        self._load_dropped_image(file_path, requested_side)

    def _set_colorspace_items(self, colorspaces: list[str]) -> None:
        combos = (
            self.left_input_colorspace_combo,
            self.left_output_colorspace_combo,
            self.right_input_colorspace_combo,
            self.right_output_colorspace_combo,
        )
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            if colorspaces:
                combo.addItems(colorspaces)
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._update_side_combo_widths()

    def _set_look_items(self, looks: list[str]) -> None:
        for combo in (self.left_looks_combo, self.right_looks_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(None)")
            combo.addItems(looks)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._update_side_combo_widths()

    def _reconcile_panel_transform_settings(
        self, colorspaces: list[str], looks: list[str]
    ) -> None:
        if not colorspaces:
            return

        first_colorspace = colorspaces[0]
        valid_colorspaces = set(colorspaces)
        valid_looks = set(looks)
        for panel in (self._compare_state.left, self._compare_state.right):
            if panel.input_colorspace not in valid_colorspaces:
                panel.input_colorspace = first_colorspace
            if panel.output_colorspace not in valid_colorspaces:
                panel.output_colorspace = first_colorspace
            if panel.look and panel.look not in valid_looks:
                panel.look = None

        self._sync_controls_from_panel("left")
        self._sync_controls_from_panel("right")

    def _sync_controls_from_panel(self, side: ViewerSide) -> None:
        panel = self._compare_state.panel(side)
        if side == "left":
            input_combo = self.left_input_colorspace_combo
            output_combo = self.left_output_colorspace_combo
            looks_combo = self.left_looks_combo
            bypass_checkbox = self.left_bypass_checkbox
        else:
            input_combo = self.right_input_colorspace_combo
            output_combo = self.right_output_colorspace_combo
            looks_combo = self.right_looks_combo
            bypass_checkbox = self.right_bypass_checkbox

        input_combo.blockSignals(True)
        output_combo.blockSignals(True)
        looks_combo.blockSignals(True)
        bypass_checkbox.blockSignals(True)
        try:
            input_index = input_combo.findText(panel.input_colorspace)
            input_combo.setCurrentIndex(max(input_index, 0))

            output_index = output_combo.findText(panel.output_colorspace)
            output_combo.setCurrentIndex(max(output_index, 0))

            look_text = panel.look or "(None)"
            look_index = looks_combo.findText(look_text)
            looks_combo.setCurrentIndex(max(look_index, 0))

            bypass_checkbox.setChecked(panel.bypass)
        finally:
            input_combo.blockSignals(False)
            output_combo.blockSignals(False)
            looks_combo.blockSignals(False)
            bypass_checkbox.blockSignals(False)

    def _set_active_side(self, side: str) -> None:
        if side not in {"left", "right"}:
            return
        new_side: ViewerSide = side
        self._compare_state.active_side = new_side
        self._clear_inspector_status()
        self.compare_view.set_active_side(new_side)
        self._update_placeholder_texts()
        self._update_active_side_ui()
        self._sync_compare_mode_combo_from_state()
        self._refresh_side_with_active_status(new_side)
        self._update_frame_controls()

    def _update_active_side_ui(self) -> None:
        active_side = self._compare_state.active_side
        active_style = "border: 2px solid #4da3ff; background: #1f1f1f; color: #f0f0f0;"
        inactive_style = "border: 1px solid #666; background: #2a2a2a; color: #f0f0f0;"
        self.left_split_view.setStyleSheet(
            active_style if active_side == "left" else inactive_style
        )
        self.right_split_view.setStyleSheet(
            active_style if active_side == "right" else inactive_style
        )

    def _loaded_panel_count(self) -> int:
        count = 0
        if self._compare_state.left.loaded_image_data is not None:
            count += 1
        if self._compare_state.right.loaded_image_data is not None:
            count += 1
        return count

    def _choose_drop_target_side(self) -> ViewerSide:
        active_side = self._compare_state.active_side
        active_panel = self._compare_state.panel(active_side)
        if active_panel.loaded_image_data is None:
            return active_side

        other_side: ViewerSide = "right" if active_side == "left" else "left"
        other_panel = self._compare_state.panel(other_side)
        if other_panel.loaded_image_data is None:
            return other_side

        return active_side

    def _resolve_drop_target_side(self, requested_side: ViewerSide | None) -> ViewerSide:
        if requested_side is None:
            return self._choose_drop_target_side()

        requested_panel = self._compare_state.panel(requested_side)
        if requested_panel.loaded_image_data is None:
            return requested_side

        other_side: ViewerSide = "right" if requested_side == "left" else "left"
        other_panel = self._compare_state.panel(other_side)
        if other_panel.loaded_image_data is None:
            return other_side

        return requested_side

    def _side_for_drop_position(self, window_pos) -> ViewerSide | None:
        if self.compare_view.isVisible():
            compare_pos = self.compare_view.mapFrom(self, window_pos.toPoint())
            if self.compare_view.rect().contains(compare_pos):
                side = self.compare_view.side_for_position(compare_pos)
                if side is not None:
                    return side
                return self._compare_state.active_side

        viewer_pos = self.split_container.mapFrom(self, window_pos.toPoint())
        if not self.split_container.rect().contains(viewer_pos):
            return None

        if self.left_split_view.isVisible():
            left_pos = self.left_split_view.mapFrom(self.split_container, viewer_pos)
            if self.left_split_view.rect().contains(left_pos):
                return "left"

        if self.right_split_view.isVisible():
            right_pos = self.right_split_view.mapFrom(self.split_container, viewer_pos)
            if self.right_split_view.rect().contains(right_pos):
                return "right"

        return None

    def _update_view_mode(self) -> None:
        """Apply current compare mode visibility and sync settings to view widgets."""
        self._update_placeholder_texts()
        view_mode = self._compare_view_state.mode
        self.compare_view.set_wipe_unsynced_nav_enabled(
            view_mode == "wipe" and not self._sync_nav_enabled
        )
        use_compare_view = view_mode in {"wipe", "full", "diff"}
        self.compare_view.setVisible(use_compare_view)
        self.split_container.setVisible(not use_compare_view)

        compare_mode: CompareMode
        if view_mode == "wipe":
            compare_mode = "wipe"
        elif view_mode == "diff":
            compare_mode = "diff"
        else:
            compare_mode = "full"
        self.compare_view.set_mode(compare_mode)
        self.compare_view.set_active_side(self._compare_state.active_side)
        self.compare_view.set_wipe_top_side(self._compare_view_state.wipe_top_side)
        self.compare_view.set_wipe_position(self._compare_view_state.wipe_position)
        self.compare_view.set_wipe_angle(self._compare_view_state.wipe_angle)
        self._apply_compare_view_transform()
        self._apply_split_view_transforms()

        if use_compare_view:
            return

        self.left_split_view.setVisible(True)
        self.right_split_view.setVisible(True)

    def _update_placeholder_texts(self) -> None:
        active_side = self._compare_state.active_side
        active_label = side_label(active_side)
        self.compare_view.set_placeholder_text(f"Drop image {active_label} here")
        self.left_split_view.set_placeholder_text("Drop image A here")
        self.right_split_view.set_placeholder_text("Drop image B here")

    def _sync_compare_mode_combo_from_state(self) -> None:
        """Sync compare-mode combo selection from internal compare state."""
        mode_data = self._compare_mode_data_from_state(self._compare_view_state.mode)
        index = self.compare_mode_combo.findData(mode_data)
        if index < 0:
            return
        self.compare_mode_combo.blockSignals(True)
        self.compare_mode_combo.setCurrentIndex(index)
        self.compare_mode_combo.blockSignals(False)

    def _update_sync_nav_visibility(self) -> None:
        self.sync_nav_checkbox.setVisible(True)

    def _set_panel_display_image(self, side: ViewerSide, image: QImage) -> None:
        self._display_images[side] = image
        self._sync_split_view_images()

    def _clear_panel_display(self, side: ViewerSide, text: str) -> None:
        self._display_images[side] = None
        self._sync_split_view_images()

    def _rescale_visible_pixmaps(self) -> None:
        self.compare_view.update()
        self.left_split_view.update()
        self.right_split_view.update()

    def _default_config_directory(self) -> Path:
        project_root = Path(__file__).resolve().parents[3]
        sample_dir = project_root / "samples" / "ocio_config"
        if sample_dir.exists():
            return sample_dir
        return project_root

    def _update_combo_width(self, combo: QComboBox) -> None:
        """Limit combo max width to its longest entry text."""
        font_metrics = combo.fontMetrics()
        texts = [combo.itemText(i) for i in range(combo.count())]
        if combo.placeholderText():
            texts.append(combo.placeholderText())
        if not texts:
            texts = [" "]

        longest_text_width = max(font_metrics.horizontalAdvance(text) for text in texts)
        horizontal_padding = 48  # arrow + frame + margins
        combo.setMaximumWidth(longest_text_width + horizontal_padding)

    def _update_global_tone_control_labels(self) -> None:
        self.luminance_value_label.setText(f"{self._global_luminance:.2f}x")
        self.exposure_value_label.setText(f"{self._global_exposure_stops:+.2f}")

    def _install_spacebar_filters(self) -> None:
        for widget in (
            self,
            self.compare_mode_combo,
            self.sync_nav_checkbox,
            self.left_input_colorspace_combo,
            self.left_output_colorspace_combo,
            self.left_looks_combo,
            self.left_bypass_checkbox,
            self.right_input_colorspace_combo,
            self.right_output_colorspace_combo,
            self.right_looks_combo,
            self.right_bypass_checkbox,
            self.compare_view,
            self.left_split_view,
            self.right_split_view,
        ):
            widget.installEventFilter(self)

    def _focused_colorspace_combo(self) -> QComboBox | None:
        """Return focused input/output colorspace combo, if any."""
        focused = self.focusWidget()
        if isinstance(focused, QComboBox) and focused in {
            self.left_input_colorspace_combo,
            self.left_output_colorspace_combo,
            self.right_input_colorspace_combo,
            self.right_output_colorspace_combo,
        }:
            return focused
        return None

    def _is_text_input_focus_active(self) -> bool:
        """Return whether focus is inside a text-editable widget subtree."""
        focused = self.focusWidget()
        if focused is None:
            return False

        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return True

        parent = focused.parentWidget()
        while parent is not None:
            if isinstance(parent, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
                return True
            parent = parent.parentWidget()
        return False

    def _step_combo_selection(self, combo: QComboBox, delta: int) -> None:
        if combo.count() <= 0:
            return
        current_index = combo.currentIndex()
        if current_index < 0:
            current_index = 0
        next_index = max(0, min(combo.count() - 1, current_index + delta))
        if next_index != current_index:
            combo.setCurrentIndex(next_index)

    def _set_channel_view(self, mode: str) -> None:
        """Set channel display mode and refresh rendered output when changed."""
        if mode == self._channel_view:
            return
        self._channel_view = mode
        self._clear_inspector_status()
        self._refresh_processed_view()

    def _activate_compare_mode(self, mode: CompareMode) -> None:
        mode_data = self._compare_mode_data_from_state(mode)
        index = self.compare_mode_combo.findData(mode_data)
        if index < 0:
            return
        if self.compare_mode_combo.currentIndex() == index:
            self._on_compare_mode_changed(index)
            return
        self.compare_mode_combo.setCurrentIndex(index)

    def _compare_mode_data_from_state(self, mode: CompareMode) -> str:
        """Map runtime compare mode to combo payload mode data."""
        if mode != "full":
            return mode
        return (
            CompareModeData.FULL_LEFT.value
            if self._compare_state.active_side == "left"
            else CompareModeData.FULL_RIGHT.value
        )

    def _apply_compare_mode_data(self, mode_data: str) -> bool:
        """Apply combo payload mode data to internal compare mode/active side."""
        if mode_data not in {mode.value for mode in CompareModeData}:
            return False
        if mode_data == CompareModeData.FULL_LEFT.value:
            self._compare_view_state.mode = "full"
            self._compare_state.active_side = "left"
            self.compare_view.set_active_side("left")
            self._update_active_side_ui()
            return True
        if mode_data == CompareModeData.FULL_RIGHT.value:
            self._compare_view_state.mode = "full"
            self._compare_state.active_side = "right"
            self.compare_view.set_active_side("right")
            self._update_active_side_ui()
            return True
        self._compare_view_state.mode = mode_data
        return True

    def _frame_view(self) -> None:
        """Reset zoom/pan framing, targeting side under cursor when relevant."""
        framed_transform = (1.0, 0.0, 0.0)
        active_side = self._compare_state.active_side
        cursor_side = self._side_under_cursor()
        if self._compare_view_state.mode in COMPARE_MODES_WITH_SIDE_CONTEXT and cursor_side is not None:
            active_side = cursor_side
            self._compare_state.active_side = active_side
            self.compare_view.set_active_side(active_side)
            self._update_active_side_ui()
            self._sync_compare_mode_combo_from_state()
            self._update_frame_controls()
        self._clear_inspector_status()

        if self._compare_view_state.mode in COMPARE_MODES_WITH_SIDE_CONTEXT and not self._sync_nav_enabled:
            self._split_view_transforms[active_side] = framed_transform
            self._apply_split_view_transforms()
            self._set_compare_view_transform(*framed_transform)
            return

        self._set_compare_view_transform(*framed_transform)
        self._split_view_transforms["left"] = framed_transform
        self._split_view_transforms["right"] = framed_transform
        self._apply_split_view_transforms()

    def _side_under_cursor(self) -> ViewerSide | None:
        """Resolve compare side currently under global cursor position."""
        window_pos = self.mapFromGlobal(QCursor.pos())
        return self._side_for_drop_position(QPointF(window_pos))

    def _refresh_processed_view(self) -> None:
        self._apply_geometry_settings_to_views()
        self._update_view_mode()
        self._refresh_panel("left", self._compare_state.left)
        self._refresh_panel("right", self._compare_state.right)
        self._update_side_transform_controls_enabled()
        self._update_compare_view_images()
        self._set_active_side_status(self._compare_state.active_side)
        self._update_frame_controls()
        self._update_persistent_status_message()

    def _refresh_side_with_active_status(self, side: ViewerSide) -> None:
        self._apply_geometry_settings_to_views()
        self._update_view_mode()
        panel = self._compare_state.panel(side)
        self._refresh_panel(side, panel)
        self._update_side_transform_controls_enabled()
        self._update_compare_view_images()
        self._set_active_side_status(self._compare_state.active_side)
        self._update_frame_controls()
        self._update_persistent_status_message()

    def _set_active_side_status(self, side: ViewerSide) -> None:
        del side
        # Footer HUD is inspection-only; workflow/status text is routed to temporary status-bar
        # messages instead of persistent footer strings.
        return

    def _update_compare_view_images(self) -> None:
        image_a = self._display_images["left"]
        image_b = self._display_images["right"]
        self.compare_view.set_images(image_a, image_b)
        self._sync_split_view_images()
        self._diff_display_image = self._build_diff_image(image_a, image_b)
        self.compare_view.set_diff_image(self._diff_display_image)

    def _refresh_display_from_cached_buffers(self) -> None:
        for side in ("left", "right"):
            buffer = self._processed_display_buffers[side]
            if buffer is None:
                continue
            self._set_panel_display_image(side, self._display_qimage_from_rgb(buffer))
        self._update_compare_view_images()

    def _schedule_tone_refresh(self) -> None:
        if self._tone_refresh_timer.isActive():
            return
        self._tone_refresh_timer.start()

    def _sync_split_view_images(self) -> None:
        self.left_split_view.set_images(self._display_images["left"], None)
        self.right_split_view.set_images(None, self._display_images["right"])

    def _apply_geometry_settings_to_views(self) -> None:
        self.compare_view.set_alignment_anchor(self._compare_state.alignment_anchor)
        self.left_split_view.set_alignment_anchor(self._compare_state.alignment_anchor)
        self.right_split_view.set_alignment_anchor(self._compare_state.alignment_anchor)

        left = self._compare_state.left
        right = self._compare_state.right
        for view in (self.compare_view, self.left_split_view, self.right_split_view):
            view.set_side_geometry(
                "left",
                canvas_policy=left.canvas_policy,
                scale_policy=left.scale_policy,
                offset_x=left.offset_x,
                offset_y=left.offset_y,
            )
            view.set_side_geometry(
                "right",
                canvas_policy=right.canvas_policy,
                scale_policy=right.scale_policy,
                offset_x=right.offset_x,
                offset_y=right.offset_y,
            )

    def _set_space_pan_enabled_all(self, enabled: bool) -> None:
        self.compare_view.set_space_pan_enabled(enabled)
        self.left_split_view.set_space_pan_enabled(enabled)
        self.right_split_view.set_space_pan_enabled(enabled)

    def _set_compare_view_transform(self, zoom: float, pan_x: float, pan_y: float) -> None:
        self._compare_view_state.zoom = zoom
        self._compare_view_state.pan_x = pan_x
        self._compare_view_state.pan_y = pan_y
        self._compare_view_state.fit_mode = "manual"
        self._apply_compare_view_transform()

    def _apply_compare_view_transform(self) -> None:
        zoom = self._compare_view_state.zoom
        pan_x = self._compare_view_state.pan_x
        pan_y = self._compare_view_state.pan_y
        self.compare_view.set_view_transform(zoom, pan_x, pan_y)

    def _apply_split_view_transforms(self) -> None:
        left_zoom, left_pan_x, left_pan_y = self._split_view_transforms["left"]
        right_zoom, right_pan_x, right_pan_y = self._split_view_transforms["right"]
        self.left_split_view.set_view_transform(left_zoom, left_pan_x, left_pan_y)
        self.right_split_view.set_view_transform(right_zoom, right_pan_x, right_pan_y)
        self.compare_view.set_wipe_side_transform("left", left_zoom, left_pan_x, left_pan_y)
        self.compare_view.set_wipe_side_transform("right", right_zoom, right_pan_x, right_pan_y)

    def _build_diff_image(
        self, image_a: QImage | None, image_b: QImage | None
    ) -> QImage | None:
        if image_a is None or image_b is None:
            self._diff_status_message = "Load both A and B for Diff mode"
            return None

        data_a = qimage_to_float_rgb(image_a)
        data_b = qimage_to_float_rgb(image_b)

        height = min(data_a.shape[0], data_b.shape[0])
        width = min(data_a.shape[1], data_b.shape[1])
        if height <= 0 or width <= 0:
            self._diff_status_message = "Diff unavailable: invalid image dimensions"
            return None

        cropped_a = data_a[:height, :width, :]
        cropped_b = data_b[:height, :width, :]
        diff_rgb = np.abs(cropped_a - cropped_b)
        # Grayscale diff intensity for clearer black/white inspection.
        diff_gray = diff_rgb.mean(axis=2, keepdims=True)
        diff = np.repeat(diff_gray, 3, axis=2)

        if data_a.shape[:2] != data_b.shape[:2]:
            self._diff_status_message = (
                f"Diff: grayscale |A-B| on overlap {width}x{height} (size mismatch)"
            )
        else:
            self._diff_status_message = "Diff: grayscale absolute RGB difference |A-B|"

        return self._display_qimage_from_rgb(diff)

    def _refresh_panel(self, side: ViewerSide, panel: PanelState) -> None:
        if panel.loaded_image_data is None:
            placeholder = "Drop A image here" if side == "left" else "Drop B image here"
            self._clear_panel_processed_buffer(side)
            self._clear_panel_display(side, placeholder)
            self._panel_status_messages[side] = "Waiting for image"
            return

        if self._loaded_ocio_config is None:
            self._set_panel_processed_buffer(side, panel.loaded_image_data)
            self._set_panel_display_image(
                side, self._display_qimage_from_rgb(self._processed_display_buffers[side])
            )
            self._panel_status_messages[side] = "Loaded image (waiting for OCIO config)"
            return

        if panel.bypass:
            self._set_panel_processed_buffer(side, panel.loaded_image_data)
            self._set_panel_display_image(
                side, self._display_qimage_from_rgb(self._processed_display_buffers[side])
            )
            self._panel_status_messages[side] = "Bypass enabled (no transform)"
            return

        input_colorspace = panel.input_colorspace
        output_colorspace = panel.output_colorspace
        if not input_colorspace or not output_colorspace:
            self._set_panel_processed_buffer(side, panel.loaded_image_data)
            self._set_panel_display_image(
                side, self._display_qimage_from_rgb(self._processed_display_buffers[side])
            )
            self._panel_status_messages[side] = (
                "Loaded image (waiting for colorspace selection)"
            )
            return

        look = panel.look

        try:
            processor = build_ocio_processor(
                self._loaded_ocio_config,
                input_colorspace,
                output_colorspace,
                look=look,
                context_values=self._ocio_context_values,
            )
            processed_data = apply_ocio_transform(panel.loaded_image_data, processor)
            self._set_panel_processed_buffer(side, processed_data)
            processed_image = self._display_qimage_from_rgb(self._processed_display_buffers[side])
        except Exception as exc:
            self._set_panel_processed_buffer(side, panel.loaded_image_data)
            self._set_panel_display_image(
                side, self._display_qimage_from_rgb(self._processed_display_buffers[side])
            )
            self._panel_status_messages[side] = str(exc)
            return

        self._set_panel_display_image(side, processed_image)
        frame_suffix = panel_frame_suffix(panel)
        if look:
            self._panel_status_messages[side] = (
                f"Processed: {input_colorspace} -> {output_colorspace} (look: {look}){frame_suffix}"
            )
            return
        self._panel_status_messages[side] = (
            f"Processed: {input_colorspace} -> {output_colorspace}{frame_suffix}"
        )

    def _set_panel_processed_buffer(self, side: ViewerSide, data: np.ndarray) -> None:
        self._processed_display_buffers[side] = np.asarray(data, dtype=np.float32)

    def _clear_panel_processed_buffer(self, side: ViewerSide) -> None:
        self._processed_display_buffers[side] = None

    def _display_qimage_from_rgb(self, image_rgb: np.ndarray) -> QImage:
        tone_adjusted = self._apply_global_tone_controls(image_rgb)
        return float_rgb_to_qimage(self._apply_channel_view_to_rgb(tone_adjusted))

    def _apply_global_tone_controls(self, image_rgb: np.ndarray) -> np.ndarray:
        final_gain = self._global_luminance * (2.0 ** self._global_exposure_stops)
        if abs(final_gain - 1.0) < 1e-6:
            return image_rgb
        adjusted = image_rgb * np.float32(final_gain)
        return np.clip(adjusted, 0.0, None)

    def _apply_channel_view_to_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        if self._channel_view == ViewerChannel.RGB.value:
            return image_rgb

        if self._channel_view == ViewerChannel.RED.value:
            channel = image_rgb[:, :, 0:1]
            return np.repeat(channel, 3, axis=2)
        if self._channel_view == ViewerChannel.GREEN.value:
            channel = image_rgb[:, :, 1:2]
            return np.repeat(channel, 3, axis=2)
        if self._channel_view == ViewerChannel.BLUE.value:
            channel = image_rgb[:, :, 2:3]
            return np.repeat(channel, 3, axis=2)
        return image_rgb

    def _sample_qimage_at_uv(
        self, image: QImage | None, u: float, v: float
    ) -> tuple[int, int, tuple[float, float, float]] | None:
        if image is None or image.isNull():
            return None
        width = image.width()
        height = image.height()
        if width <= 0 or height <= 0:
            return None

        x = int(round(max(0.0, min(1.0, u)) * max(width - 1, 0)))
        y = int(round(max(0.0, min(1.0, v)) * max(height - 1, 0)))
        color = image.pixelColor(x, y)
        return (x, y, (color.redF(), color.greenF(), color.blueF()))

    def _format_sample(
        self, sample: tuple[int, int, tuple[float, float, float]] | None
    ) -> str:
        if sample is None:
            return "n/a"
        r, g, b = sample[2]
        return f"({r:.3f},{g:.3f},{b:.3f})"

    def _set_hud_sample(
        self,
        *,
        width: int,
        height: int,
        x: int,
        y: int,
        r: float,
        g: float,
        b: float,
    ) -> None:
        self._hud_resolution_label.setText(f"{width}x{height}")
        self._hud_coord_label.setText(f"x={x} y={y}")
        self._hud_r_label.setText(f"{r:.4f}")
        self._hud_g_label.setText(f"{g:.4f}")
        self._hud_b_label.setText(f"{b:.4f}")

    def _clear_hud_sample(self) -> None:
        self._hud_coord_label.setText("x=- y=-")
        self._hud_r_label.setText("----")
        self._hud_g_label.setText("----")
        self._hud_b_label.setText("----")

    def _mode_requirement_hint(self) -> str | None:
        mode = self._compare_view_state.mode
        loaded_count = self._loaded_panel_count()
        if loaded_count >= 2 or mode not in COMPARE_MODES_REQUIRING_BOTH_IMAGES:
            return None
        if mode == "split":
            return "Split compare needs A and B loaded"
        if mode == "wipe":
            return "Wipe compare needs A and B loaded"
        if mode == "diff":
            return "Diff needs A and B loaded"
        return None

    def _clear_inspector_status(self) -> None:
        return

    def _show_temporary_status(self, message: str, timeout_ms: int = 3000) -> None:
        self.statusBar().showMessage(message, timeout_ms)
        QTimer.singleShot(
            timeout_ms, lambda: self.statusBar().showMessage(self._persistent_status_message)
        )

    def _update_persistent_status_message(self) -> None:
        left_panel = self._compare_state.left
        right_panel = self._compare_state.right
        self._persistent_status_message = persistent_status_message(
            left_panel,
            right_panel,
            self._loaded_ocio_config_path,
        )
        self.statusBar().showMessage(self._persistent_status_message)

    def _set_viewer_background(self, color: QColor) -> None:
        self.compare_view.set_background_color(color)
        self.left_split_view.set_background_color(color)
        self.right_split_view.set_background_color(color)

    def _update_drop_target_highlight(self, side: ViewerSide | None) -> None:
        self.compare_view.set_drop_target_side(side)
        self.left_split_view.set_drop_target_side("left" if side == "left" else None)
        self.right_split_view.set_drop_target_side("right" if side == "right" else None)

    def _clear_drop_target_highlight(self) -> None:
        self._update_drop_target_highlight(None)

    def _update_side_transform_controls_enabled(self) -> None:
        left_enabled = self._compare_state.left.loaded_image_data is not None
        self.left_input_colorspace_combo.setEnabled(left_enabled)
        self.left_output_colorspace_combo.setEnabled(left_enabled)
        self.left_looks_combo.setEnabled(left_enabled)
        self.left_bypass_checkbox.setEnabled(left_enabled)

        right_enabled = self._compare_state.right.loaded_image_data is not None
        self.right_input_colorspace_combo.setEnabled(right_enabled)
        self.right_output_colorspace_combo.setEnabled(right_enabled)
        self.right_looks_combo.setEnabled(right_enabled)
        self.right_bypass_checkbox.setEnabled(right_enabled)

    def _load_ui_pixmap(self, relative_path: str) -> QPixmap | None:
        """Load a UI asset pixmap from prism.ui package resources."""
        try:
            resource = resources.files("prism.ui").joinpath(relative_path)
            if not resource.is_file():
                return None
            pixmap = QPixmap(str(resource))
            if pixmap.isNull():
                return None
            return pixmap
        except Exception:
            return None

    def _load_ui_icon(self, relative_path: str) -> QIcon | None:
        """Load a UI asset icon from prism.ui package resources."""
        try:
            resource = resources.files("prism.ui").joinpath(relative_path)
            if not resource.is_file():
                return None
            icon = QIcon(str(resource))
            if icon.isNull():
                return None
            return icon
        except Exception:
            return None

    def _update_frame_controls(self) -> None:
        self._update_frame_controls_for_side("left")
        self._update_frame_controls_for_side("right")

    def _update_frame_controls_for_side(self, side: ViewerSide) -> None:
        panel = self._compare_state.panel(side)
        has_source = panel.loaded_source is not None
        frame_count = max(1, panel.frame_count)
        frame_text = "-"
        if has_source and panel.loaded_source is not None:
            frame_info = panel.loaded_source.get_frame_info(panel.current_frame_index)
            frame_text = str(frame_info.display_number)

        frame_edit = self.left_frame_index_edit if side == "left" else self.right_frame_index_edit
        prev_button = self.left_prev_frame_button if side == "left" else self.right_prev_frame_button
        next_button = self.left_next_frame_button if side == "left" else self.right_next_frame_button

        frame_edit.blockSignals(True)
        frame_edit.setText(frame_text)
        frame_edit.blockSignals(False)
        can_step = has_source and frame_count > 1
        prev_button.setEnabled(can_step)
        next_button.setEnabled(can_step)
        frame_edit.setEnabled(has_source)

    def _resolve_frame_number_to_index(self, panel: PanelState, requested_number: int) -> int:
        if panel.loaded_source is None:
            return 0
        if requested_number <= 0:
            requested_number = 1

        source = panel.loaded_source
        frame_count = max(1, panel.frame_count)
        for index in range(frame_count):
            if source.get_frame_info(index).display_number == requested_number:
                return index

        return max(0, min(frame_count - 1, requested_number - 1))
