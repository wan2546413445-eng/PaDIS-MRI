# ---------------------------------------------------------------
# Global-context conditioned PaDIS patch EDM loss.
#
# Inspired by "Local Patches Meet Global Context" for 3D CT, adapted
# to 2D MRI magnitude/complex patch priors:
#   local 2D patch + absolute position + downsampled global image context.
#
# This is still a patch prior training loss, not a physical k-space loss.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from training.patch_loss import Patch_EDMLoss
from torch_utils import persistence


@persistence.persistent_class
class GlobalContextPatch_EDMLoss(Patch_EDMLoss):
    """Patch EDM loss conditioned on a low-resolution global context.

    Training input to Patch_EDMPrecond becomes:
        noisy patch x_t^p,
        x_pos = cat([absolute position channels, low-res global context patch], dim=1)

    Default context:
        context = upsample(downsample(magnitude(c_in * x_t_full), S), full_size)
        then crop context with the same random patch coordinates.

    This follows the "local patches + downsampled global context" idea, but
    adapts it to 2D MRI by using magnitude low-resolution context by default.
    """

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        global_context_mode="magnitude_lowres",  # "magnitude_lowres", "complex_lowres", "none"
        global_context_size=96,
        global_context_dropout=0.0,
        global_context_from="noisy",             # "noisy" is train/eval-consistent; "clean" is diagnostic only.
        global_context_eps=1e-8,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.global_context_mode = str(global_context_mode)
        self.global_context_size = int(global_context_size)
        self.global_context_dropout = float(global_context_dropout)
        self.global_context_from = str(global_context_from)
        self.global_context_eps = float(global_context_eps)
        self._config_printed = False

    def _sample_patch_indices(self, images, patch_size):
        device = images.device
        batch_size, resolution = images.size(0), images.size(2)
        h, w = images.size(2), images.size(3)
        th, tw = patch_size, patch_size

        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]

        x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0
        images_pos = torch.cat((x_pos, y_pos), dim=1)
        return rows, columns, images_pos

    def _crop_with_indices(self, tensor, rows, columns):
        # Same advanced-indexing pattern as PaDIS Patch_EDMLoss.pachify().
        batch_size = tensor.size(0)
        th = rows.size(1)
        tensor = tensor.permute(1, 0, 2, 3)
        patch = tensor[
            :,
            torch.arange(batch_size, device=tensor.device)[:, None, None],
            rows[:, torch.arange(th, device=tensor.device)[:, None]],
            columns[:, None],
        ]
        return patch.permute(1, 0, 2, 3)

    def _normalize_context(self, context):
        # Keep the condition numerically comparable across images/sigmas.
        # Use per-sample standardization, but avoid destroying zero-context dropout.
        dims = (1, 2, 3)
        mean = context.mean(dim=dims, keepdim=True)
        std = context.std(dim=dims, keepdim=True).clamp_min(self.global_context_eps)
        return (context - mean) / std

    def _build_global_context(self, clean_full, noisy_full, sigma):
        mode = self.global_context_mode.lower()
        if mode == "none":
            return None

        if self.global_context_from.lower() == "clean":
            source = clean_full
        else:
            # Match Patch_EDMPrecond's input scale. The denoiser sees c_in * noisy_patch.
            c_in = 1.0 / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)
            source = c_in * noisy_full

        if mode == "magnitude_lowres":
            context = torch.sqrt(torch.sum(source ** 2, dim=1, keepdim=True) + self.global_context_eps)
        elif mode == "complex_lowres":
            context = source
        else:
            raise ValueError(f"Unknown global_context_mode={self.global_context_mode}")

        h, w = context.shape[-2:]
        s = min(self.global_context_size, h, w)

        # 2D MRI adaptation of downsampled global volume context:
        # downsample full 2D slice to SxS and upsample back, preserving global structure
        # while removing local high-frequency detail.
        if s < h or s < w:
            low = F.interpolate(context, size=(s, s), mode="area")
            context = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)

        context = self._normalize_context(context)

        if self.global_context_dropout > 0:
            keep = (torch.rand([context.shape[0], 1, 1, 1], device=context.device) >= self.global_context_dropout).to(context.dtype)
            context = context * keep

        return context

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        del resolution

        if not self._config_printed:
            print(
                "[GlobalContextPatch_EDMLoss] "
                f"mode={self.global_context_mode}, size={self.global_context_size}, "
                f"dropout={self.global_context_dropout}, from={self.global_context_from}"
            )
            self._config_printed = True

        # Apply augmentation on full images before patch sampling so that local patch
        # and global context remain geometrically consistent.
        y_full, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        rnd_normal = torch.randn([y_full.shape[0], 1, 1, 1], device=y_full.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        noise_full = torch.randn_like(y_full) * sigma
        yn_full = y_full + noise_full

        rows, columns, images_pos = self._sample_patch_indices(y_full, patch_size)
        y_patch = self._crop_with_indices(y_full, rows, columns)
        yn_patch = self._crop_with_indices(yn_full, rows, columns)

        context_full = self._build_global_context(y_full, yn_full, sigma)
        if context_full is not None:
            context_patch = self._crop_with_indices(context_full, rows, columns)
            x_pos = torch.cat([images_pos, context_patch], dim=1)
        else:
            x_pos = images_pos

        D_yn = net(yn_patch, sigma, x_pos=x_pos, class_labels=labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y_patch) ** 2)
        return loss
