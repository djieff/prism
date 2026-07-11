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
- `lut_analysis.py`
- `lut_interpolation.py`
- `scope_waveform.py`
- `scope_waveform_science.py`

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

### `lut_analysis.py`
- Computes UI-agnostic LUT Inspector summary metrics from plotted sample data.
- Reports sample count, effective channel count, per-channel output min/max,
  values outside `[0, 1]`, and per-channel monotonicity.
- Keeps numerical summary behavior out of Qt widgets.

### `lut_interpolation.py`
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

### `scope_waveform.py`
- Builds deterministic raw R, G, B, and encoded Y' density grids from float RGB
  analysis buffers.
- Defaults to BT.709 and accepts explicit BT.2020 selection.
- Records signal-standard and coefficient provenance in `WaveformTrace`.
- Keeps the legacy `density_luma` field name for compatibility; its documented
  meaning is encoded Y' density, not scene-linear luminance.

### `scope_waveform_science.py`
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
