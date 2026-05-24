# OCIO Context

## Purpose

Describe how Prism handles OCIO context variables for context-dependent transforms.

## Current Model

- Canonical context state is owned by `MainWindow`:
  - `self._ocio_context_values: dict[str, str]`
- Variable names/defaults come from loaded OCIO config declarations
  (config environment variable names + defaults), not arbitrary OS env variable listing.
- Context UI is presentation/editor only:
  - `ContextVariablesPanel` and `ContextVariablesDock` do not own authoritative state.

Update lifecycle:

1. User loads OCIO config.
2. Main window extracts context variables/defaults.
3. Main window stores canonical context mapping.
4. Main window syncs dock/panel rows.
5. User edits value in panel (`editingFinished`).
6. Panel emits `context_values_changed`.
7. Main window updates canonical mapping and refreshes processing.

## UI Integration

- `View -> OCIO Context Variables` toggles dock visibility.
- Dock starts hidden by default.
- Empty states:
  - no config loaded: `No OCIO config loaded.`
  - config with no variables: `No OCIO context variables detected.`
- Hiding/closing dock does not discard canonical context state.

## Processor Integration

- `core.ocio_processor.build_ocio_processor(...)` accepts `context_values`.
- When provided, context string variables are set on OCIO context before processor creation.
- Processing behavior is then re-evaluated with the updated context mapping.

## Constraints

- Context handling must stay explicit and deterministic.
- OCIO/context logic should remain outside widget internals.
- UI edits should emit bounded events (on edit-finish) to avoid excessive rebuild churn.
