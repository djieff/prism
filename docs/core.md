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

## Extension Points

- Add new source kinds by extending source detection + concrete model implementing `FrameSource` behavior.
- Extend transform parameterization in `viewer_state` first, then consume in `ocio_processor`.
- Keep extension logic explicit and side-effect conscious; avoid hidden routing abstractions.
