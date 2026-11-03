"""Fixed PaDIS settings and the small set of LGFC experiment parameters."""

from dataclasses import dataclass


IMAGE_SIZE = 384
PAD = 64
PATCH_SIZE = 64
STEPS = 78
INNER_LOOPS = 10
ZETA = 3.0
SEED = 123
SIGMA_MIN = 0.003
SIGMA_MAX = 10.0
RHO = 7.0
EPS = 1e-8


@dataclass(frozen=True)
class LGFCConfig:
    """LGFC parameters that may change between experiments."""

    reference_mode: str = "ema"
    ema_beta: float = 0.8
    start_outer: int = 60
    correction_every: int = 1
    unsampled_weight: float = 0.6
    max_relative_update: float = 0.08
