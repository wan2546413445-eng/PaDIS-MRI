# ---------------------------------------------------------------
# Extended from PaDIS Patch_EDMLoss with Focal Frequency Loss.
# ---------------------------------------------------------------

import torch

from training.patch_loss import Patch_EDMLoss
from torch_utils import persistence


@persistence.persistent_class
class FocalFrequencyPatch_EDMLoss(Patch_EDMLoss):
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        ffl_weight=0.05,
        ffl_alpha=1.0,
        ffl_log_matrix=False,
        ffl_batch_matrix=False,
        ffl_eps=1e-8,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.ffl_weight = ffl_weight
        self.ffl_alpha = ffl_alpha
        self.ffl_log_matrix = ffl_log_matrix
        self.ffl_batch_matrix = ffl_batch_matrix
        self.ffl_eps = ffl_eps
        self._ffl_config_printed = False

    def _focal_frequency_loss_map(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        freq_dist = torch.abs(pred_fft - target_fft) ** 2
        focal_weight = freq_dist ** self.ffl_alpha

        if self.ffl_log_matrix:
            focal_weight = torch.log(focal_weight + 1.0)

        if self.ffl_batch_matrix:
            norm = focal_weight.mean()
        else:
            norm = focal_weight.mean(dim=(1, 2, 3), keepdim=True)

        focal_weight = focal_weight / (norm + self.ffl_eps)
        focal_weight = focal_weight.detach()

        ffl_map = focal_weight * freq_dist
        return ffl_map.real

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        del resolution
        images, images_pos = self.pachify(images, patch_size)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        yn = y + torch.randn_like(y) * sigma

        D_yn = net(yn, sigma, x_pos=images_pos, class_labels=labels, augment_labels=augment_labels)
        base_loss = weight * ((D_yn - y) ** 2)

        if self.ffl_weight <= 0:
            return base_loss

        if not self._ffl_config_printed:
            print(
                f"[FocalFrequencyPatch_EDMLoss] ffl_weight={self.ffl_weight}, "
                f"ffl_alpha={self.ffl_alpha}, log_matrix={self.ffl_log_matrix}, "
                f"batch_matrix={self.ffl_batch_matrix}"
            )
            self._ffl_config_printed = True

        ffl_map = self._focal_frequency_loss_map(D_yn, y)
        loss = base_loss + self.ffl_weight * ffl_map
        return loss
