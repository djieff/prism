# I/O

## Purpose

`io` handles source/config discovery and data loading from disk into formats consumed by `core` and `ui`.

## Key Modules

- `source_detect.py`
- `sequence_loader.py`
- `image_loader.py`
- `movie_loader.py`
- `ocio_config.py`
- `lut_loader.py`

## Responsibilities

### `source_detect.py`
- Classifies input path into still/sequence/movie categories.

### `sequence_loader.py`
- Groups deterministic numbered frame sequences from a seed file.

### `image_loader.py`
- Loads still images through Qt codecs.
- Uses OpenImageIO fallback for EXR/DPX where Qt codec support may be missing.
- Converts between `QImage` and normalized float RGB numpy buffers.

### `movie_loader.py`
- Loads movie frames via optional backends (`cv2`, `imageio.v3`).
- Provides frame count and per-frame decode with backend fallback attempts.

### `ocio_config.py`
- Loads OCIO config files.
- Enumerates colorspaces, looks, and config-declared context variables/defaults.

### `lut_loader.py`
- Loads LUT plot data for the LUT Inspector.
- Supports `.cube`, `.csp`, `.spi1d`, `.spi3d`, `.3dl`, and Houdini `.lut`
  files.
- Returns `LutPlotData` for UI plotting and core summary analysis.
- Returns `LutVolumeData` for supported 3D LUT volume previews.
- Keeps parsing and format validation in `io`; reusable analysis and
  interpolation helpers live in `core`.
- Uses direct red-fastest lattice indexing for exact neutral-axis extraction
  where interpolation is unnecessary.
- Delegates shaped CSP-style 3D sampling to `core.lut_interpolation`.
- Keeps volume data stored as `(z, y, x, 3)` arrays for downstream projection
  helpers.

## Environment Sensitivity

- OCIO requires Python bindings (`PyOpenColorIO` import name, pip package `OpenColorIO`).
- Movie decoding depends on optional backend availability (`opencv-python`, `imageio[ffmpeg]`).
- EXR/DPX still-image support depends on Qt build codecs or OpenImageIO fallback.
- Windows development is a primary target; avoid Linux-only assumptions in tooling/workflows.
