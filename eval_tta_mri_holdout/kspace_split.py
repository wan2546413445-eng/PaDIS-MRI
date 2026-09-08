"""Fixed Cartesian-column holdout split for the 384x384 R=7 formal mask."""

import random
from dataclasses import dataclass
from typing import Tuple

import torch


IMAGE_SIZE = 384
ACS_LINES = 24
ACS_START = (IMAGE_SIZE - ACS_LINES) // 2
ACS_STOP = ACS_START + ACS_LINES
EXPECTED_ACQUIRED_LINES = IMAGE_SIZE // 7
HOLDOUT_FRACTION = 0.20
EXPECTED_HOLDOUT_LINES = round(HOLDOUT_FRACTION * EXPECTED_ACQUIRED_LINES)


@dataclass(frozen=True)
class HoldoutSplit:
    mask_cond: torch.Tensor
    mask_hold: torch.Tensor
    holdout_columns: Tuple[int, ...]
    num_acquired_lines: int


def deterministic_refinement_seed(
    subject_seed: int,
    event: int,
    refinement: int,
    stream: int = 0,
) -> int:
    """Stable seed mixer that never reads or changes a global RNG."""
    modulus = 2**63 - 1
    return int(
        (
            int(subject_seed) * 1_000_003
            + int(event) * 10_007
            + int(refinement) * 101
            + int(stream) * 1_000_000_007
        )
        % modulus
    )


def split_mask7_columns(mask_full: torch.Tensor, *, seed: int) -> HoldoutSplit:
    """Hold out exactly 11 non-ACS acquired columns from the formal mask7."""
    if tuple(mask_full.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Formal mask must end in 384x384, got {tuple(mask_full.shape)}"
        )

    mask_2d = mask_full.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)[0]
    mask_bool = mask_2d != 0
    acquired_columns = torch.all(mask_bool, dim=0)
    if torch.any(mask_bool != acquired_columns.unsqueeze(0)):
        raise ValueError("Formal mask7 must contain complete Cartesian columns")

    acquired = torch.nonzero(acquired_columns, as_tuple=False).flatten().tolist()
    if len(acquired) != EXPECTED_ACQUIRED_LINES:
        raise ValueError(
            f"Formal mask7 must have 54 acquired columns, found {len(acquired)}"
        )
    center_columns = tuple(range(ACS_START, ACS_STOP))
    if not all(bool(acquired_columns[column]) for column in center_columns):
        raise ValueError("All center 24 ACS columns must be acquired")

    center_set = set(center_columns)
    outer_acquired = [column for column in acquired if column not in center_set]
    if len(outer_acquired) != EXPECTED_ACQUIRED_LINES - ACS_LINES:
        raise ValueError("Formal mask7 must contain 30 acquired non-ACS columns")

    local_rng = random.Random(int(seed))
    holdout_columns = tuple(
        sorted(local_rng.sample(outer_acquired, EXPECTED_HOLDOUT_LINES))
    )
    mask_hold = torch.zeros_like(mask_full)
    mask_hold[..., :, list(holdout_columns)] = mask_full[
        ..., :, list(holdout_columns)
    ]
    mask_cond = mask_full - mask_hold
    return HoldoutSplit(
        mask_cond=mask_cond,
        mask_hold=mask_hold,
        holdout_columns=holdout_columns,
        num_acquired_lines=len(acquired),
    )
