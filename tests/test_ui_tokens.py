"""Regression tests for shared UI token sets."""

from prism.core.ui_tokens import (
    COMPARE_MODES_REQUIRING_BOTH_IMAGES,
    COMPARE_MODES_WITH_SIDE_CONTEXT,
)


def test_compare_modes_with_side_context_stays_expected() -> None:
    assert COMPARE_MODES_WITH_SIDE_CONTEXT == {"split", "wipe", "full"}


def test_compare_modes_requiring_both_images_stays_expected() -> None:
    assert COMPARE_MODES_REQUIRING_BOTH_IMAGES == {"split", "wipe", "diff"}
