# Core

## Purpose

`core` owns viewer state, source abstractions, frame orchestration, and OCIO transform application contracts.

## Key Modules

- `viewer_state.py`
- `frame_source.py`
- `source_models.py`
- `frame_cache.py`
- `frame_service.py`
- `ocio_processor.py`
- `lut/analysis.py`
- `lut/interpolation.py`
- `lut/volume_projection.py`
- `scopes/cie_xy.py`
- `scopes/vectorscope.py`
- `scopes/waveform.py`
- `scopes/waveform_science.py`

## Responsibilities

### `viewer_state.py`
- Defines UI-facing state dataclasses (`PanelState`, `CompareState`, `CompareViewState`).
- Holds compare mode, wipe state, zoom/pan state, and per-side colorspace/look/bypass settings.

### `frame_source.py`
- Defines shared contracts/protocols (`FrameSource`) and frame metadata (`FrameInfo`).

### `source_models.py`
- Provides concrete source models for still images, sequences, and movie files.
- Normalizes frame indexing and frame-info access.
- Exposes `create_source_from_path(...)` for input classification handoff.

### `frame_cache.py`
- Small bounded LRU-like cache keyed by `(source_id, frame_index)`.

### `frame_service.py`
- Orchestrates frame retrieval through cache + source decode.
- Tracks request tokens for stale-result detection when rapid frame changes occur.

### `ocio_processor.py`
- Builds OCIO processors from selected input/output colorspace, optional look, and context values.
- Applies processor transforms to float RGB buffers.

### `lut/analysis.py`
- Computes UI-agnostic LUT Inspector summary metrics from plotted sample data.
- Reports sample count, effective channel count, per-channel output min/max,
  values outside `[0, 1]`, and per-channel monotonicity.
- Keeps numerical summary behavior out of Qt widgets.

### `lut/interpolation.py`
- Provides reusable LUT interpolation helpers for inspection workflows.
- Evaluates 1D piecewise-linear prelut mappings with explicit endpoint
  clamping.
- Normalizes shaped prelut outputs to unit cube coordinates with explicit
  clamping.
- Uses SciPy `RegularGridInterpolator` behind a Prism helper for 3D LUT
  trilinear sampling.
- Public sampling coordinates are normalized `(x, y, z)` while stored cube data
  remains `(z, y, x, channels)`.
- Direct neutral-axis extraction remains preferred when exact lattice samples
  are available and interpolation is unnecessary.

### `lut/volume_projection.py`
- Builds deterministic 3D LUT point-cloud samples from stored `(z, y, x, 3)`
  volume data.
- Projects sampled RGB points into normalized 2D coordinates for the UI Volume
  view.
- Supports `RGB isometric`, `RG plane`, `RB plane`, and `GB plane` projections.
- Supports output-cloud positions and input-lattice positions.
- Applies a deterministic sample cap for large LUT previews so UI rendering
  stays responsive.
- Selects neutral-axis samples from input lattice diagonal indices for the UI
  overlay; this keeps the reference stable even when output RGB is warped.

### `scopes/cie_xy.py`
- Builds deterministic CIE 1931 xy chromaticity traces from float RGB analysis
  buffers.
- Converts viewer RGB values to XYZ before computing xy chromaticity. The UI
  uses Prism's viewer/display RGB interpretation; the core still exposes
  Colour Science-backed RGB colourspace data for deterministic conversion and
  tests.
- Clamps sampled RGB values to `[0, 1]` for the first display-normalized monitor
  contract.
- Ignores black/zero samples where `X + Y + Z <= 0` because chromaticity is
  undefined.
- Builds normalized density and color-density grids over fixed xy bounds
  (`x: 0.0..0.8`, `y: 0.0..0.9`).
- Exposes CIE 1931 spectral locus xy coordinates from numerical Colour Science
  CMF data, without using plotting APIs.
- Provides reference gamut overlay contracts and de-duplicates selected overlay
  names while preserving selection order.

### `scopes/vectorscope.py`
- Builds deterministic component chroma traces from float RGB analysis buffers.
- Uses explicit BT.709/BT.2020 encoded-signal coefficients for the Cb/Cr-style
  chroma calculation.
- Records signal-standard, coefficients, plot scale, normalized density, and
  source-color density in `VectorscopeTrace`.
- Keeps vectorscope math separate from colourimetric CIE xy conversion.

### `scopes/waveform.py`
- Builds deterministic raw R, G, B, and encoded Y' density grids from float RGB
  analysis buffers.
- Defaults to BT.709 and accepts explicit BT.2020 selection.
- Records signal-standard and coefficient provenance in `WaveformTrace`.
- Keeps the legacy `density_luma` field name for compatibility; its documented
  meaning is encoded Y' density, not scene-linear luminance.

### `scopes/waveform_science.py`
- Obtains BT.709/BT.2020 encoded-signal weights from Colour's
  `WEIGHTS_YCBCR` registry.
- Returns defensive, read-only coefficient arrays.
- Uses SciPy Gaussian filtering with the locked `(0.5, 0.5)` kernel for
  presentation copies only.
- Normalizes filtered R, G, B, and Y' channels with one shared maximum so
  channel relationships are preserved.
- Keeps raw waveform density arrays unchanged and UI-independent.

## Colour Metrics Deferral

Colour Science is available as a Prism dependency, but LUT Inspector does not
currently expose Delta E or other perceptual LUT metrics. Arbitrary LUT files do
not reliably declare source colourspace, target colourspace, transfer encoding,
viewing condition, or creative/technical intent. Any Colour-backed LUT metric
must therefore be introduced by a separate plan with an explicit colourspace and
comparison contract.

## Extension Points

- Add new source kinds by extending source detection + concrete model implementing `FrameSource` behavior.
- Extend transform parameterization in `viewer_state` first, then consume in `ocio_processor`.
- Keep extension logic explicit and side-effect conscious; avoid hidden routing abstractions.
