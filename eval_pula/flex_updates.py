"""Dispatch P/D single-step kernels, with explicit FOV/padding bridge."""
import torch.nn.functional as F
from eval_pula.flex_pula import pula_initialize, pula_step
from eval_pula.flex_dps import dps_step

class UpdateDispatcher:
    """Shared FOV, persistent DPS padding only during consecutive D updates.

    P->D: zero-pad current FOV, exactly as legacy hybrid.
    D->P: crop current FOV, discard padding, identity on FOV (new mixed kernel).
    D->D: retain padding; P->P: normal pULA path.
    """
    def __init__(self, state, system, oracle, raw_operator, raw_measurement):
        self.state = state
        self.system = system
        self.oracle = oracle
        self.raw_operator = raw_operator
        self.raw_measurement = raw_measurement
        self.padded = None
        self.fov = slice(oracle.pad, oracle.pad+oracle.image_size)

    def step(self, method, sigma, shift, gamma, zeta, final_level):
        if method == 'P':
            self.padded = None
            self.state, info = pula_step(self.state, self.system, self.oracle, sigma, shift, gamma)
        elif method == 'D':
            if self.padded is None:
                self.padded = F.pad(self.state, (self.oracle.pad,)*4)
            self.padded, info = dps_step(self.padded, self.oracle, self.raw_operator,
                                        self.raw_measurement, self.system.generator,
                                        sigma, shift, zeta, final_level)
            self.state = self.padded[:, :, self.fov, self.fov].detach()
        else:
            raise ValueError(method)
        return info
