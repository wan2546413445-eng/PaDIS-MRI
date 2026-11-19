"""Whole-image EDM denoising with eval/recon.py:dps_edm tensor conventions."""

import hashlib
import sys
from pathlib import Path

import torch


def checkpoint_environment(net):
    """Report real import origins, including embedded NVIDIA persistent source."""
    module = sys.modules[type(net).__module__]
    persistence = sys.modules.get("torch_utils.persistence")
    source = getattr(persistence, "_module_to_src_dict", {}).get(module)
    source_file = getattr(module, "__file__", None)
    if source is None and source_file and Path(source_file).is_file():
        source = Path(source_file).read_text()
    return {
        "model_class": f"{type(net).__module__}.{type(net).__qualname__}",
        "model_class_name": type(net).__name__,
        "model_module_file": source_file,
        "model_source_sha256": None if source is None else hashlib.sha256(source.encode()).hexdigest(),
        "training_module_files": {
            name: getattr(sys.modules[name], "__file__", None)
            for name in ("training", "training.networks") if name in sys.modules
        },
        "sys_path": list(sys.path),
    }


class FullImageScoreOracle:
    def __init__(self, net):
        self.net = net
        self.first_forward = None

    @torch.no_grad()
    def denoise(self, state, sigma, shift=None):
        """Return D_sigma(state); the sampler's optional shift slot is unused."""
        sigma_arg = torch.as_tensor(sigma, device=state.device).float()
        x_real = torch.view_as_real(state.squeeze(1)).permute(0, 3, 1, 2)
        den_real = self.net(x_real, sigma_arg)
        den_cplx = torch.complex(den_real[:, 0], den_real[:, 1]).unsqueeze(1)
        if self.first_forward is None:
            self.first_forward = {
                "state_input_shape": list(state.shape),
                "network_input_shape": list(x_real.shape),
                "network_output_shape": list(den_real.shape),
                "state_dtype": str(state.dtype), "network_input_dtype": str(x_real.dtype),
                "network_output_dtype": str(den_real.dtype),
                "sigma_shape": list(sigma_arg.shape), "sigma_dtype": str(sigma_arg.dtype),
                "device": str(state.device),
            }
        return den_cplx
