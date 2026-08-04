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
- `lut_inspector/window.py`
- `lut_inspector/plot_widget.py`
- `lut_inspector/volume_widget.py`
- `scopes/cie_xy_window.py`
- `scopes/cie_xy_plot_widget.py`
- `scopes/vectorscope_window.py`
- `scopes/vectorscope_plot_widget.py`
- `scopes/waveform_window.py`
- `scopes/waveform_plot_widget.py`

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

### `lut_inspector/window.py`
- Modeless LUT utility window opened from `View -> LUT Inspection`.
- Handles LUT file drag/drop and optional browse-file selection.
- Orchestrates LUT loading via `io.lut.loader` and routes parsed data to the
  curve and volume widgets.
- Presents `Curves` and `Volume` tabs.
- Hosts Volume controls for projection mode, point-position mode, and
  neutral-axis visibility.
- Formats summary metrics produced by `core.lut.analysis`, including sample
  count, channel count, output min/max, out-of-range state, monotonicity, and
  CSP shaper state when present.
- Reports load status/error feedback to the user.
- For `.cube` files, expects scalar-equivalent RGB domains (`DOMAIN_MIN` and `DOMAIN_MAX` components must match per row); mixed per-channel domain components are treated as unsupported and surfaced as load errors.
- Does not present Colour/Delta E metrics until the Inspector has an explicit
  colourspace/comparison assumption to show to the user.

### `lut_inspector/plot_widget.py`
- Renders scalable X/Y LUT transfer curves.
- Draws axes/grid and per-channel curve overlays (RGB when present).
- Renders only the `LutPlotData` curves supplied by the loader; it does not own
  parsing, summary analysis, interpolation, or colour science decisions.
- Resizes with its parent inspection window.

### `lut_inspector/volume_widget.py`
- Renders projected 3D LUT volume previews as a QPainter point cloud.
- Supports projection modes:
  - `RGB isometric`
  - `RG plane`
  - `RB plane`
  - `GB plane`
- Supports point positions from transformed output RGB values or the original
  input lattice.
- Draws plane-axis letters for direct channel projections.
- Draws an optional neutral-axis overlay based on the input grayscale diagonal.
- Keeps rendering bounded through the projection sample limit reported in the
  widget status text.
- Consumes projected data from `core.lut.volume_projection`; it does not parse
  LUT files or own sampling rules.

### `scopes/cie_xy_window.py`
- Modeless CIE xy chromaticity utility window opened from
  `View -> Monitoring -> CIE xy Chromaticity`.
- Supports local scope modes: `A`, `B`, `A|B` (side-by-side).
- Analyzes the current viewer RGB signal using Prism's display-viewer
  interpretation. The window does not expose a separate input-colorspace
  selector.
- Hosts exactly three global gamut overlay selectors. Each selector can be
  `None`, duplicate overlays are de-duplicated by the plot widget, and selected
  overlays are shared across A, B, and both `Split` panes.
- Hosts a `White Point` selector for `None`, `D50`, `D55`, `D60`, `D65`,
  `D75`, and `DCI-P3` marker overlays without changing image chromaticity
  analysis.
- Hosts a `Trace Color` selector, defaulting to `Normal`, for normal or boosted
  trace-color rendering without changing image chromaticity analysis.
- Consumes per-side float analysis buffers from main window and routes CIE xy
  view data to plot widgets. Buffers are post-OCIO when a transform is active,
  but may be untransformed in bypass/incomplete-config paths.
- Analysis occurs before global exposure/luminance and channel-view presentation
  controls.
- Mirrors waveform/vectorscope lifecycle and mode sync: `Full (A)`, `Full (B)`,
  and `Split` are synchronized with the main viewer; `Wipe` and `Diff` show an
  explicit unsupported state.

### `scopes/cie_xy_plot_widget.py`
- Renders a CIE 1931 xy graph with fixed bounds, grid, axis labels, numerical
  spectral locus, optional white point marker, selected gamut triangles, and image
  density.
- Rebuilds a cached density `QImage` from `CieXyTrace` data and validates
  density shape, finite values, and non-negative values.
- Supports normal source-color rendering and boosted source-color rendering for
  display readability; both modes use the same xy density data.
- Draws empty graph context even when no trace is available, so gamut overlays
  and the CIE reference remain visible.
- Keeps overlay state display-only; it does not alter image chromaticity
  computation.

### `scopes/vectorscope_window.py`
- Modeless vectorscope utility window opened from `View -> Monitoring -> Vectorscope`.
- Supports local scope modes: `A`, `B`, `A|B` (side-by-side).
- Supports explicit `BT.709` and `BT.2020` standards for component chroma
  calculation.
- Hosts a `Color` selector, defaulting to `Normal`, for `Teal`, `Normal`, or
  `Boosted` trace rendering without changing component chroma analysis.
- Consumes per-side float analysis buffers from main window and routes
  vectorscope data to plot widgets.
- Mirrors waveform mode sync behavior for `Full (A)`, `Full (B)`, and `Split`;
  `Wipe` and `Diff` show an explicit unsupported state.

### `scopes/vectorscope_plot_widget.py`
- Renders normalized chroma density with circular graticule, target boxes, and
  optional normal or boosted source-color density.
- Keeps the component-chroma presentation separate from colourimetric CIE xy
  plotting.

### `scopes/waveform_window.py`
- Modeless waveform utility window opened from `View -> Monitoring -> Waveform Monitor`.
- Supports local scope modes: `A`, `B`, `A|B` (side-by-side).
- Supports explicit `BT.709` and `BT.2020` standards for the encoded `Y'` trace.
- Defaults to BT.709 and does not infer standards from arbitrary OCIO names.
- Does not auto-follow active side changes from compare mode.
- Consumes per-side float analysis buffers from main window and routes waveform
  data to plot widgets. Buffers are post-OCIO when a transform is active, but
  may be untransformed in bypass/incomplete-config paths.
- Analysis occurs before global exposure/luminance and channel-view presentation
  controls.

### `scopes/waveform_plot_widget.py`
- Renders waveform density heatmaps for RGB channel overlays and encoded `Y'`.
- Uses SciPy Gaussian filtering with `sigma=(0.5, 0.5)` on rendering copies.
- Normalizes filtered R, G, B, and Y' together with one shared factor.
- Caches prepared densities per trace; signal-mode changes reuse the cache.
- Never mutates the raw density arrays in `WaveformTrace`.
- Displays empty-state text when no waveform trace is available.
- Resizes with parent window layout.

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
