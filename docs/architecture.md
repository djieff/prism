# Architecture

## Purpose

Prism is a desktop OCIO viewer with a simple directional architecture:

`input -> decode/load -> OCIO process -> viewer presentation`

The project prioritizes explicit state and predictable flow over abstraction-heavy designs.

## Subsystems

- `core`: state models and processing contracts
- `io`: source/config/image/movie loading and detection
- `ui`: widgets, event flow, and presentation logic

## Dependency Direction

- `ui` depends on `core` and `io`
- `core` should not depend on `ui`
- `io` should not depend on `ui`

## Data Flow

Primary runtime flow:

1. User loads media/config from UI.
2. `io` resolves source kind and decodes frames/images.
3. `core` state is updated (per-side A/B source + transform state, frame index, compare/view state).
4. `core.ocio_processor` builds processor using selected colorspaces/look/context.
5. Processed display buffers are pushed to compare/split views.
6. UI hover/navigation reads display buffers for HUD and interaction feedback.
7. An open waveform window reads the per-side float analysis buffers, builds
   raw density data in `core.scope_waveform`, and prepares filtered rendering
   copies through `core.scope_waveform_science`.
8. An open LUT Inspector loads LUT files through `io.lut_loader`, summarizes
   plot data through `core.lut_analysis`, uses `core.lut_interpolation` for
   shaped 3D sampling where needed, and renders curves in UI widgets.

Waveform boundary notes:
- the analysis buffer is post-OCIO only when a transform is active;
- bypass/incomplete-config paths may provide untransformed source data;
- global exposure/luminance and channel-view controls are applied later during
  viewer image presentation and are not waveform inputs;
- OCIO names are not used to infer a Colour signal standard.

LUT Inspector boundary notes:
- file parsing and format validation stay in `io.lut_loader`;
- numerical summary and interpolation helpers stay in `core`;
- curve rendering and status text stay in `ui`;
- 3D LUT display is currently a neutral-axis projection, not a volumetric
  preview;
- Colour/Delta E LUT metrics are deferred until a dedicated workflow can state
  source/target colourspace and comparison assumptions explicitly.

Frame flow (multi-frame sources):

1. Source model (`core.source_models`) receives frame request.
2. `core.frame_service` requests cache lookup (`core.frame_cache`).
3. Cache hit returns immediately; miss decodes via source backend.
4. Result includes staleness metadata for request-token safety.

## Constraints

- Keep OCIO business logic outside widgets.
- Keep state explicit and serializable where practical (dataclasses, plain containers).
- Avoid speculative framework layers (managers/registries/pipelines) in MVP scope.
- Preserve deterministic behavior for processing and metadata returned to UI.
