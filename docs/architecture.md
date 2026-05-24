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
