"""
LMPC-specific patch partition helpers.

This file does not replace the original denoise_padding.py. It reuses the
original denoisedFromPatches implementation and adds deterministic/complementary
offset support for Late-Stage Multi-Partition Consensus (LMPC).
"""

import random
from typing import Optional, Sequence, Tuple

from denoise_padding import denoisedFromPatches


Offset = Tuple[int, int]


def get_indices_lmpc(
    spaced,
    patches: int,
    pad: int,
    psize: int,
    freezeindex: bool = False,
    offset: Optional[Sequence[int]] = None,
):
    """
    Build one non-overlapping PaDIS partition.

    Parameters
    ----------
    spaced:
        Original PaDIS grid start coordinates.
    patches:
        Number of patch starts per spatial dimension.
    pad:
        Image padding. Valid offsets are in [0, pad - 1].
    psize:
        Patch size.
    freezeindex:
        Preserve the original helper's fixed-offset debug behavior.
    offset:
        None -> use the original random-offset behavior.
        (a,b) -> use an explicitly specified offset.
    """
    if offset is None:
        a, b = 0, 0
        if pad > 0:
            a = random.randint(0, pad - 1)
            b = random.randint(0, pad - 1)
        if freezeindex:
            a, b = 0, 0
    else:
        if len(offset) != 2:
            raise ValueError(f"offset must contain two integers, got {offset}")
        a, b = int(offset[0]), int(offset[1])

        if pad == 0:
            if (a, b) != (0, 0):
                raise ValueError(
                    f"pad=0 only supports offset=(0,0), got {(a, b)}"
                )
        elif not (0 <= a < pad and 0 <= b < pad):
            raise ValueError(
                f"offset must satisfy 0 <= a,b < pad={pad}, got {(a, b)}"
            )

    indices = []
    for p in range(patches):
        for q in range(patches):
            indices.append(
                [
                    int(spaced[p]) + a,
                    int(spaced[p]) + a + psize,
                    int(spaced[q]) + b,
                    int(spaced[q]) + b + psize,
                ]
            )
    return indices
