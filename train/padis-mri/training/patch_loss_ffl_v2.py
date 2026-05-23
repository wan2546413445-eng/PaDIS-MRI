# ---------------------------------------------------------------
# Extended from PaDIS Patch_EDMLoss with robust patch-local frequency losses.
# ---------------------------------------------------------------

import torch

from training.patch_loss import Patch_EDMLoss
from torch_utils import persistence


@persistence.persistent_class
class FocalFrequencyPatch_EDMLoss(Patch_EDMLoss):
    """Patch EDM loss + controlled patch-local frequency regularization.

    Notes:
        This loss is still a PaDIS patch prior loss:
            full image -> random patch -> Patch_EDMPrecond -> patch loss.

        The default mode is no longer the original full FFT FFL. It uses:
            1) magnitude-domain representation, to avoid directly forcing real/imag phase channels;
            2) Tukey/Hann/Hamming window before FFT, to reduce rectangular crop leakage;
            3) smooth high-pass mask in patch frequency, to focus on local details;
            4) percentile clipping, to prevent a few extreme frequency bins from dominating;
            5) sigma gate, to apply frequency pressure mainly at lower noise levels.

        It is a patch-local spectral regularizer, not a physical MRI k-space loss.
    """

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
        # v2 controls
        ffl_mode="robust_hf",          # "v1" or "robust_hf"
        ffl_use_magnitude=True,        # use sqrt(real^2+imag^2) before FFT
        ffl_window="tukey",            # "rect", "hann", "hamming", "tukey"
        ffl_tukey_alpha=0.5,
        ffl_hp_radius=0.08,            # normalized patch-frequency radius; <=0 disables high-pass
        ffl_hp_softness=0.02,          # smooth transition width; <=0 gives hard high-pass
        ffl_clip_low=0.10,             # percentile lower bound; <=0 disables lower clipping
        ffl_clip_high=0.98,            # percentile upper bound; >=1 disables upper clipping
        ffl_sigma_min=0.0,             # lower sigma gate; <=0 disables lower gate
        ffl_sigma_max=0.5,             # upper sigma gate; <=0 disables upper gate
        ffl_sigma_softness=0.05,       # smooth gate width; <=0 gives hard sigma gate
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)

        self.ffl_weight = float(ffl_weight)
        self.ffl_alpha = float(ffl_alpha)
        self.ffl_log_matrix = bool(ffl_log_matrix)
        self.ffl_batch_matrix = bool(ffl_batch_matrix)
        self.ffl_eps = float(ffl_eps)

        self.ffl_mode = str(ffl_mode)
        self.ffl_use_magnitude = bool(ffl_use_magnitude)
        self.ffl_window = str(ffl_window)
        self.ffl_tukey_alpha = float(ffl_tukey_alpha)
        self.ffl_hp_radius = float(ffl_hp_radius)
        self.ffl_hp_softness = float(ffl_hp_softness)
        self.ffl_clip_low = float(ffl_clip_low)
        self.ffl_clip_high = float(ffl_clip_high)
        self.ffl_sigma_min = float(ffl_sigma_min)
        self.ffl_sigma_max = float(ffl_sigma_max)
        self.ffl_sigma_softness = float(ffl_sigma_softness)

        self._ffl_config_printed = False
        self._window_cache = {}

    def _tukey_window_1d(self, n, device, dtype):
        alpha = self.ffl_tukey_alpha
        if alpha <= 0:
            return torch.ones(n, device=device, dtype=dtype)
        if alpha >= 1:
            return torch.hann_window(n, periodic=False, device=device, dtype=dtype)

        x = torch.linspace(0, 1, n, device=device, dtype=dtype)
        w = torch.ones_like(x)

        left = x < alpha / 2
        right = x >= 1 - alpha / 2

        w[left] = 0.5 * (1.0 + torch.cos(torch.pi * (2.0 * x[left] / alpha - 1.0)))
        w[right] = 0.5 * (1.0 + torch.cos(torch.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0)))
        return w

    def _make_window(self, h, w, device, dtype):
        key = (self.ffl_window, h, w, str(device), str(dtype), self.ffl_tukey_alpha)
        if key in self._window_cache:
            return self._window_cache[key]

        name = self.ffl_window.lower()
        if name == "rect":
            wy = torch.ones(h, device=device, dtype=dtype)
            wx = torch.ones(w, device=device, dtype=dtype)
        elif name == "hann":
            wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
            wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
        elif name == "hamming":
            wy = torch.hamming_window(h, periodic=False, device=device, dtype=dtype)
            wx = torch.hamming_window(w, periodic=False, device=device, dtype=dtype)
        elif name == "tukey":
            wy = self._tukey_window_1d(h, device, dtype)
            wx = self._tukey_window_1d(w, device, dtype)
        else:
            raise ValueError(f"Unknown ffl_window={self.ffl_window}")

        win = wy[:, None] * wx[None, :]
        win = win[None, None, :, :]

        # Keep frequency-loss scale comparable across different windows.
        win = win / torch.sqrt(torch.mean(win ** 2) + self.ffl_eps)
        self._window_cache[key] = win
        return win

    def _make_highpass_mask(self, h, w, device, dtype):
        if self.ffl_hp_radius <= 0:
            return 1.0

        fy = torch.fft.fftfreq(h, d=1.0, device=device).to(dtype)
        fx = torch.fft.fftfreq(w, d=1.0, device=device).to(dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        radius = self.ffl_hp_radius

        if self.ffl_hp_softness > 0:
            hp = torch.sigmoid((rr - radius) / self.ffl_hp_softness)
        else:
            hp = (rr >= radius).to(dtype)

        return hp[None, None, :, :]

    def _robust_percentile_mask(self, freq_dist):
        # freq_dist shape: [B,C,H,W] or [B,1,H,W]
        qlow = self.ffl_clip_low
        qhigh = self.ffl_clip_high

        if qlow <= 0 and qhigh >= 1:
            return torch.ones_like(freq_dist)

        flat = freq_dist.detach().reshape(freq_dist.shape[0], -1)
        mask = torch.ones_like(freq_dist, dtype=torch.bool)

        if qlow > 0:
            low = torch.quantile(flat, qlow, dim=1).view(-1, 1, 1, 1)
            mask = mask & (freq_dist >= low)

        if qhigh < 1:
            high = torch.quantile(flat, qhigh, dim=1).view(-1, 1, 1, 1)
            mask = mask & (freq_dist <= high)

        return mask.to(freq_dist.dtype)

    def _sigma_gate(self, sigma):
        gate = torch.ones_like(sigma)

        if self.ffl_sigma_max > 0:
            if self.ffl_sigma_softness > 0:
                gate = gate * torch.sigmoid((self.ffl_sigma_max - sigma) / self.ffl_sigma_softness)
            else:
                gate = gate * (sigma <= self.ffl_sigma_max).to(sigma.dtype)

        if self.ffl_sigma_min > 0:
            if self.ffl_sigma_softness > 0:
                gate = gate * torch.sigmoid((sigma - self.ffl_sigma_min) / self.ffl_sigma_softness)
            else:
                gate = gate * (sigma >= self.ffl_sigma_min).to(sigma.dtype)

        return gate

    def _legacy_focal_frequency_loss_map(self, pred, target):
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

        return (focal_weight * freq_dist).real

    def _robust_high_frequency_loss_map(self, pred, target):
        # Real/imag MRI channels are often not visually aligned with magnitude texture.
        # Default: regularize local magnitude spectrum.
        if self.ffl_use_magnitude:
            pred_x = torch.sqrt(torch.sum(pred ** 2, dim=1, keepdim=True) + self.ffl_eps)
            target_x = torch.sqrt(torch.sum(target ** 2, dim=1, keepdim=True) + self.ffl_eps)
        else:
            pred_x = pred
            target_x = target

        _, _, h, w = pred_x.shape
        win = self._make_window(h, w, pred_x.device, pred_x.dtype)
        hp = self._make_highpass_mask(h, w, pred_x.device, pred_x.dtype)

        pred_fft = torch.fft.fft2(pred_x * win, norm="ortho")
        target_fft = torch.fft.fft2(target_x * win, norm="ortho")

        freq_dist = torch.abs(pred_fft - target_fft) ** 2
        freq_dist = freq_dist * hp

        valid = self._robust_percentile_mask(freq_dist)

        focal_weight = freq_dist ** self.ffl_alpha
        if self.ffl_log_matrix:
            focal_weight = torch.log(focal_weight + 1.0)

        focal_weight = focal_weight * valid

        if self.ffl_batch_matrix:
            norm = focal_weight.mean()
        else:
            norm = focal_weight.mean(dim=(1, 2, 3), keepdim=True)

        focal_weight = (focal_weight / (norm + self.ffl_eps)).detach()
        ffl_map = focal_weight * freq_dist * valid

        # If using magnitude spectrum, broadcast the scalar map to real/imag loss channels.
        if ffl_map.shape[1] == 1 and pred.shape[1] != 1:
            ffl_map = ffl_map.expand(-1, pred.shape[1], -1, -1)

        return ffl_map.real

    def _focal_frequency_loss_map(self, pred, target):
        mode = self.ffl_mode.lower()
        if mode in ["v1", "legacy", "full"]:
            return self._legacy_focal_frequency_loss_map(pred, target)
        if mode in ["robust_hf", "hf", "v2"]:
            return self._robust_high_frequency_loss_map(pred, target)
        raise ValueError(f"Unknown ffl_mode={self.ffl_mode}")

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
                "[FocalFrequencyPatch_EDMLoss] "
                f"mode={self.ffl_mode}, weight={self.ffl_weight}, alpha={self.ffl_alpha}, "
                f"log_matrix={self.ffl_log_matrix}, batch_matrix={self.ffl_batch_matrix}, "
                f"use_magnitude={self.ffl_use_magnitude}, window={self.ffl_window}, "
                f"tukey_alpha={self.ffl_tukey_alpha}, hp_radius={self.ffl_hp_radius}, "
                f"hp_softness={self.ffl_hp_softness}, clip=({self.ffl_clip_low},{self.ffl_clip_high}), "
                f"sigma_gate=({self.ffl_sigma_min},{self.ffl_sigma_max}), "
                f"sigma_softness={self.ffl_sigma_softness}"
            )
            self._ffl_config_printed = True

        ffl_map = self._focal_frequency_loss_map(D_yn, y)
        gate = self._sigma_gate(sigma)

        return base_loss + self.ffl_weight * gate * ffl_map
