"""Patch-level foreground weighting for the identical, symmetric OPA experiment."""
import torch

from torch_utils import persistence, training_stats
from training.dataset import ImageFolderDatasetX
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import _extract_patches, _position_maps, _sample_overlap_geometry


class IndexedDataset(ImageFolderDatasetX):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label, self._raw_idx[idx], bool(self._xflip[idx])


def load_integral(path, dataset):
    cache = torch.load(path, map_location="cpu")
    integral = cache["integral"]
    expected = (dataset.num_examples, dataset.height + 1, dataset.width + 1)
    if integral.dtype != torch.int32 or tuple(integral.shape) != expected or cache["padding"] != dataset.pad:
        raise ValueError(f"Foreground cache must match dataset: int32 {expected}, padding={dataset.pad}")
    return integral


def fractions(integral, raw_idx, flipped, top, left, size):
    top, left = torch.stack((top, left)).cpu().unbind()
    left = torch.where(flipped, integral.shape[2] - 1 - left - size, left)
    bottom, right = top + size, left + size
    count = (integral[raw_idx, bottom, right] - integral[raw_idx, top, right]
             - integral[raw_idx, bottom, left] + integral[raw_idx, top, left])
    return count.float() / (size * size)


@persistence.persistent_class
class OPAPFWLoss(Patch_EDMLoss):
    def __init__(self, w_max, lambda_overlap=1.25, **edm_kwargs):
        super().__init__(**edm_kwargs)
        self.w_max = w_max
        self.lambda_overlap = lambda_overlap

    def __call__(self, net, images, patch_size, resolution, labels, augment_pipe,
                 integral, raw_idx, flipped):
        if augment_pipe is not None:
            raise ValueError("OPA-PFW uses augment=0")
        b, _, h, w = images.shape
        device = images.device
        paired = patch_size == 64
        if paired:
            top, left, *_ = _sample_overlap_geometry(images, patch_size)
        elif h == w == patch_size:
            top = left = torch.zeros(b, dtype=torch.long, device=device)
        else:
            # Same draws and order as Patch_EDMLoss.pachify.
            top = torch.randint(h - patch_size + 1, (b,), device=device)
            left = torch.randint(w - patch_size + 1, (b,), device=device)

        clean = _extract_patches(images, top, left, patch_size)
        pos = _position_maps(top, left, b, patch_size, resolution if paired else h, device)
        sigma = (torch.randn((b, 1, 1, 1), device=device) * self.P_std + self.P_mean).exp()
        weight = (sigma.square() + self.sigma_data ** 2) / (sigma * self.sigma_data).square()
        noisy = clean + sigma * torch.randn_like(clean)
        if paired:
            noisy2 = clean + sigma * torch.randn_like(clean)
            d1, d2 = net(torch.cat((noisy, noisy2)), sigma.repeat(2, 1, 1, 1),
                         x_pos=torch.cat((pos, pos)), class_labels=None if labels is None else
                         torch.cat((labels, labels)), augment_labels=None).chunk(2)
            edm = 0.5 * (weight * (d1-clean).square() + weight * (d2-clean).square())
            opa = (weight * (d2-d1).square()).mean((1, 2, 3), keepdim=True)
        else:
            denoised = net(noisy, sigma, x_pos=pos, class_labels=labels, augment_labels=None)
            edm = weight * (denoised-clean).square()
            opa = edm.new_zeros((b, 1, 1, 1))
        original = edm + self.lambda_overlap * opa
        ratio = fractions(integral, raw_idx, flipped, top, left, patch_size).to(device)
        patch_weight = 1 + (self.w_max - 1) * ratio
        total = original * patch_weight.view(b, 1, 1, 1)
        for name, value in (("foreground_fraction", ratio), ("weight", patch_weight),
                            ("edm", edm), ("opa_raw", opa), ("weighted_total", total)):
            training_stats.report("PFW/" + name, value.detach())
        return total
