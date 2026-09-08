"""GCC-PaDIS reuses the formal mask7 split validated by Holdout Scan-TTA."""

from eval_tta_mri_holdout.kspace_split import (
    ACS_LINES,
    ACS_START,
    ACS_STOP,
    EXPECTED_ACQUIRED_LINES,
    EXPECTED_HOLDOUT_LINES,
    HOLDOUT_FRACTION,
    IMAGE_SIZE,
    HoldoutSplit,
    deterministic_refinement_seed,
    split_mask7_columns,
)

__all__ = [
    "ACS_LINES",
    "ACS_START",
    "ACS_STOP",
    "EXPECTED_ACQUIRED_LINES",
    "EXPECTED_HOLDOUT_LINES",
    "HOLDOUT_FRACTION",
    "IMAGE_SIZE",
    "HoldoutSplit",
    "deterministic_refinement_seed",
    "split_mask7_columns",
]
