"""
PaDIS patch score oracle.

This module isolates the *prior interface* from any posterior sampler.

Why this exists
---------------
In the current PaDIS-MRI repo, `eval/recon.py::dps2()` interleaves:
    1) PaDIS patch denoising,
    2) score construction,
    3) DPS likelihood gradient through the denoiser,
    4) DPS diffusion updates.

That structure is correct for the original DPS baseline, but it is not a clean
interface for plugging PaDIS into another posterior sampler such as pULA.

This file extracts only the prior-side computation:
    score_oracle(x, sigma) -> full-image complex score.

Modes
-----
- mode="direct":
      query = x
      score = (D(query, sigma) - query) / sigma^2
  This is the mathematically clean "score oracle" interface expected by
  sampler-level methods such as ULA/pULA.

- mode="legacy_dps2":
      query = x + sigma * eps
      score = (D(query, sigma) - query) / sigma^2
  This reproduces the score construction used inside the current
  `eval/recon.py::dps2()` path for equivalence checks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"
for _p in (str(EVAL_DIR), str(PROJECT_ROOT), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from denoise_padding import denoisedFromPatches, getIndices

Tensor = torch.Tensor


@dataclass
class ScoreOracleOutput:
    score: Tensor
    denoised_complex: Tensor
    denoised_real: Tensor
    query: Tensor
    indices: List[List[int]]


@dataclass
class PaDISScoreOracleConfig:
    image_size: int = 384
    pad: int = 64
    psize: int = 64
    mode: str = "direct"          # "direct" or "legacy_dps2"
    freeze_patch_indices: bool = False


class PaDISPatchScoreOracle:
    def __init__(self, net: Any, config: PaDISScoreOracleConfig):
        self.net = net
        self.cfg = config
        self.net.eval()

        if self.cfg.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.cfg.psize <= 0:
            raise ValueError("psize must be positive")
        if self.cfg.pad < 0:
            raise ValueError("pad must be non-negative")

        self._patches = self.cfg.image_size // self.cfg.psize + 1
        self._spaced = np.linspace(
            0,
            (self._patches - 1) * self.cfg.psize,
            self._patches,
            dtype=int,
        )

    def build_indices(self) -> List[List[int]]:
        return getIndices(
            self._spaced,
            self._patches,
            self.cfg.pad,
            self.cfg.psize,
            freezeindex=self.cfg.freeze_patch_indices,
        )

    @staticmethod
    def _to_two_channel_real(x: Tensor) -> Tensor:
        if not torch.is_complex(x):
            raise TypeError("PaDIS score oracle expects complex tensors.")
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W] complex tensor, got {tuple(x.shape)}")
        return torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _to_complex(x_real: Tensor) -> Tensor:
        if x_real.ndim != 4 or x_real.shape[1] != 2:
            raise ValueError(f"Expected [B,2,H,W] real tensor, got {tuple(x_real.shape)}")
        return torch.complex(x_real[:, 0], x_real[:, 1]).unsqueeze(1)

    @torch.no_grad()
    def __call__(
        self,
        x: Tensor,
        sigma: Tensor | float,
        latents_pos: Tensor,
        *,
        mode: Optional[str] = None,
        indices: Optional[List[List[int]]] = None,
        randn_like=torch.randn_like,
        eps: Optional[Tensor] = None,
    ) -> ScoreOracleOutput:
        """
        Compute the PaDIS patch score.

        Args:
            x: padded complex state [B,1,H_pad,W_pad].
            sigma: scalar tensor/float.
            latents_pos: positional grid expected by the PaDIS network.
            mode: override config mode.
            indices: optional patch indices, useful for deterministic tests.
            randn_like: random generator compatible with torch.randn_like.
            eps: optional externally supplied complex noise, used by
                 legacy_dps2 equivalence tests.
        """
        mode = (mode or self.cfg.mode).lower()
        sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma), device=x.device)
        sigma_t = sigma_t.to(device=x.device, dtype=x.real.dtype)

        if mode == "direct":
            query = x
        elif mode == "legacy_dps2":
            if eps is None:
                eps = randn_like(x)
            query = x + sigma_t * eps
        else:
            raise ValueError(f"Unsupported score oracle mode: {mode}")

        if indices is None:
            indices = self.build_indices()

        query_real = self._to_two_channel_real(query)
        den_real = denoisedFromPatches(
            self.net,
            query_real,
            sigma_t,
            latents_pos,
            None,
            indices,
            t_goal=0,
            wrong=False,
        )
        den_cplx = self._to_complex(den_real)
        score = (den_cplx - query) / (sigma_t ** 2)

        return ScoreOracleOutput(
            score=score,
            denoised_complex=den_cplx,
            denoised_real=den_real,
            query=query,
            indices=indices,
        )
