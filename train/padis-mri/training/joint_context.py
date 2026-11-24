"""Joint patch / full-field noisy conditioning for PaDIS-MRI.

Reference implementation of the *input construction* used by 3D CT's joint
local-global patch prior.  This is NOT a claimed exact full-image score.

Runtime information:
 - x: [B,2,P,P] locally noisy real/imag patch
 - context: [B,2,P,P] downsample of the SAME noisy full image (P=64)
 - x_pos: [B,2,P,P] absolute patch coordinates

For 16/32 auxiliary patch training only, context defaults to zeros. It is
not computed, nor is there an extra global denoiser at inference.
"""
import torch

from torch_utils import persistence
from training.networks import Patch_EDMPrecond

# In repo this parent is persistence-decorated; inherit its underlying class
# to avoid persistence double-wrapping (same mechanism as existing code).
_Base = Patch_EDMPrecond.__mro__[1]


@persistence.persistent_class
class JointContextEDMPrecond(_Base):
    """A single standard SongUNet accepting local, position and full global.

    The training loop supplies img_channels=4 (2 MRI + 2 position channels).
    The actual backbone expects 6 (the other 2 are real/imag global context).
    """
    def __init__(self, img_channels, out_channels=2, **kwargs):
        if int(img_channels) != 4 or int(out_channels) != 2:
            raise ValueError('JointContextEDMPrecond expects 2 MRI + 2 position channels, output 2')
        super().__init__(img_channels=int(img_channels)+2,
                         out_channels=int(out_channels), **kwargs)
        self.use_joint_global = True
        self.joint_context_size = 64

    def forward(self, x, sigma, x_pos=None, class_labels=None,
                force_fp32=False, global_context=None, **model_kwargs):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError('x must have shape [B,2,P,P]')
        x = x.to(torch.float32)
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
        if x_pos is None or x_pos.shape != x.shape:
            raise ValueError('x_pos must be [B,2,P,P], matching x')
        if global_context is None:
            # Existing multi-scale training: 16/32 original EDM, no global.
            # The model summary may also call with a large dummy x.
            if x.shape[-1] == self.joint_context_size:
                raise ValueError('64x64 JointContext requires global_context; missing train/eval integration')
            global_context = torch.zeros_like(x)
        elif global_context.shape != x.shape:
            raise ValueError(f'global_context {global_context.shape} must match x {x.shape}')
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros((x.shape[0], self.label_dim), device=x.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        c_skip = self.sigma_data ** 2 / (sigma.square() + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma.square() + self.sigma_data ** 2).sqrt()
        c_in = (sigma.square() + self.sigma_data ** 2).rsqrt()
        c_noise = sigma.log() / 4

        # Local & global are the SAME stochastic realization at the SAME sigma.
        # Global downsampling changes the noise variance; it is not a second
        # sigma-scheduled denoising process.  The raw noisy local patch alone
        # remains the EDM c_skip residual input.
        x_in = torch.cat((c_in * x, x_pos.to(torch.float32),
                          c_in * global_context.to(torch.float32)), dim=1)
        pred = self.model(x_in.to(dtype), c_noise.flatten(),
                          class_labels=class_labels, **model_kwargs)
        if pred.shape != x.shape:
            raise RuntimeError(f'Expected denoising prediction {x.shape}, got {pred.shape}')
        return c_skip * x + c_out * pred.to(torch.float32)
