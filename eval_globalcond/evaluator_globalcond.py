import os
import re
import sys
import pickle
import random
import glob
import threading
import numpy as np
import torch
import tqdm
import time
import argparse
import json
from pathlib import Path
from PIL import Image
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from typing import List, Tuple
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(REPO_DIR, "eval")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, EVAL_DIR)

import dnnlib

from dnnlib.util import configure_bart
configure_bart()
from bart import bart

from inverse_operators import *
from eval.utils import fftmod, makeFigures
from recon_globalcond import dps2_globalcond
from evaluator import DPSHyperEvaluator as BaseDPSHyperEvaluator
random.seed(123)
torch.manual_seed(123)
np.random.seed(123)
torch.set_printoptions(profile="full")


class DPSHyperEvaluator(BaseDPSHyperEvaluator):
    def __init__(self, *args, global_context_size=96, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_context_size = int(global_context_size)

    def dps2_wrapper(
        self,
        inverse_op,
        measurement,
        clean,
        zeta,
        pad,
        psize,
        num_steps,
        save_dir=None,
        tag=None,
        save_intermediate: bool = False,
        intermediate_every: int = 10,
        inner_loops: int = 10,
    ):
        if measurement is None:
            raise NotImplementedError(
                "GC-PaDIS eval currently supports reconstruction only. "
                "Unconditional dps_uncond is not adapted to global context."
            )

        recon, a, b, c, d, e, f = dps2_globalcond(
            net=self.model,
            latents=self.latents,
            latents_pos=self.latents_pos,
            inverseop=inverse_op,
            measurement=measurement,
            clean=clean,
            pad=pad,
            psize=psize,
            zeta=zeta,
            num_steps=num_steps,
            inner_loops=inner_loops,
            save_dir=save_dir,
            tag=tag,
            save_intermediate=save_intermediate,
            intermediate_every=intermediate_every,
            global_context_size=self.global_context_size,
        )
        return recon, a, b, c, d, e, f