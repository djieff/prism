# UI

## Purpose

`ui` composes Prism widgets, routes user interaction, and presents processed images and inspection data.
It consumes structured/core state and should not embed OCIO processing logic.

## Key Modules

- `main_window.py`
- `compare_view.py`
- `view_math.py`
- `context_variables_panel.py`
- `context_variables_dock.py`

## Responsibilities

### `main_window.py`
- Owns top-level app window composition and menu actions.
- Owns canonical UI-facing state orchestration:
  - compare mode selection (`Split`, `Wipe`, `Full (A)`, `Full (B)`, `Diff`)
  - per-side (A/B) source + transform state
  - frame stepping controls
  - OCIO config selection and transform parameter controls
  - OCIO context variable state updates from dock panel
- Triggers refresh/reprocess flow after relevant state edits.
- Maintains inspection HUD fields (resolution/coords/source/channel values).

### `compare_view.py`
- Renders compare modes (`split` routing in main window, plus `wipe`, `full_a`, `full_b`, `diff` draw paths).
- Handles interaction:
  - wipe divider dragging
  - wipe divider rotation handles
  - pan/zoom interaction
  - side-aware interaction routing from cursor position
  - hover coordinate emission for inspector/HUD
- Applies configurable viewport background color.

### `view_math.py`
- Shared geometry utilities for zoom/pan and coordinate conversion.
- Keeps mapping logic reusable/testable outside widget handlers.

### `context_variables_panel.py`
- UI editor for OCIO context variables (`QLabel` + `QLineEdit` rows).
- Emits `context_values_changed` on `editingFinished`.
- Supports empty states for no-config and no-variable cases.

### `context_variables_dock.py`
- `QDockWidget` wrapper for panel integration into `View` menu.
- Provides floating/toggleable panel behavior.

## Signal/Data Flow

### Image + transform update flow

1. User changes source/frame/colorspace/look/context/bypass state.
2. Main window updates canonical panel/context state.
3. Main window refreshes side display images through OCIO processing path.
4. Compare/split widgets receive updated `QImage` buffers.

### Hover/HUD flow

1. Compare widgets emit normalized hover coordinates + side hint.
2. Main window samples display buffers at hover UV.
3. HUD labels update with resolution, pixel coords, selected source label, and channel values.

### Compare interaction flow

- `CompareView` emits:
  - `wipe_changed`
  - `view_changed`
  - `hover_changed`
- Main window consumes these signals and synchronizes compare/view state.

## Boundaries

- UI should call `core.ocio_processor` through main-window orchestration, not inside lower-level widgets.
- UI widgets should avoid owning canonical OCIO/config/source state.
- Complex logic should be delegated to reusable helpers (`view_math`, `core`, `io`), keeping event handlers thin.
