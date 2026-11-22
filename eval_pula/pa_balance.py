"""PA-Balanced-pULA coefficients and lagged EDM innovation statistics.

All variance maps are per-real-channel covariances in padis_real units.
This module neither calls the denoiser/MRI operator nor draws random numbers.
"""

import math

import torch
import torch.nn.functional as F


VARIANTS = ("baseline", "noise08", "balanced_k0", "balanced_full")


def edm_sigma_data(net):
    """Read the loaded checkpoint, never substitute a training default."""
    if not hasattr(net, "sigma_data") and hasattr(net, "module"):
        net = net.module
    if not hasattr(net, "sigma_data"):
        raise ValueError("Balanced pULA requires sigma_data from the loaded EDM checkpoint.")
    value = float(net.sigma_data)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Checkpoint sigma_data must be finite and positive.")
    return value


def edm_coefficients(sigma, sigma_data):
    sigma, sigma_data = float(sigma), float(sigma_data)
    if not (math.isfinite(sigma) and sigma > 0 and math.isfinite(sigma_data) and sigma_data > 0):
        raise ValueError("EDM sigma and sigma_data must be finite and positive.")
    radius = math.hypot(sigma, sigma_data)
    c_skip = (sigma_data / radius) ** 2
    c_out = sigma * (sigma_data / radius)
    return c_skip, c_out, 1.0 / radius


def balanced_coefficients(template, sigma_net, sigma_data, gamma, convention, variance=None):
    """J_code = (1+gamma/2) B_code + diagonal, with a shared real/imag map."""
    if convention not in ("padis_real", "paper_complex"):
        raise ValueError(f"Unknown pULA convention: {convention}")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive.")
    _, _, c_in = edm_coefficients(sigma_net, sigma_data)
    kappa = 2 if convention == "paper_complex" else 1
    # h0=(1-c_skip)/sigma_sys^2, evaluated without subtractive cancellation.
    h0_code = c_in ** 2 / kappa
    try:
        base = (1.0 / sigma_net) ** 2 / kappa + gamma * h0_code / 2
    except OverflowError as error:
        raise ValueError("sigma is too small for the CG coefficient dtype.") from error
    limit = torch.finfo(template.real.dtype).max
    if not math.isfinite(base) or base > limit or 1.0 + gamma / 2 > limit:
        raise ValueError("sigma/gamma exceed the CG coefficient dtype range.")
    diagonal = template.real.new_tensor(base)
    if variance is not None:
        if (variance.shape[-2:] != template.shape[-2:] or variance.numel() != template.numel()
                or variance.is_complex() or variance.device != template.device):
            raise ValueError("V_real must be a real FOV map on the state device, with one value per FOV pixel.")
        if not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError("V_real must be finite and nonnegative.")
        diagonal = (base + gamma * variance.reshape_as(template).to(torch.float64) / (2 * kappa)).to(template.real.dtype)
    if not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise FloatingPointError("Balanced diagonal must remain finite and positive in the state dtype.")
    return 1.0 + gamma / 2, diagonal


def region_mean(value, block_size):
    """Non-overlapping means, including correctly weighted partial edge blocks."""
    return F.avg_pool2d(value, block_size, stride=block_size, ceil_mode=True, count_include_pad=False)


def region_expand(value, shape, block_size):
    return value.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)[..., :shape[-2], :shape[-1]]


def squared_modulus(value):
    # Accumulate statistics in double precision; leave state/network dtype alone.
    return value.real.double().square() + value.imag.double().square()


def interior_window(template, shift, pad, psize):
    """Sine-squared patch coordinates of actual shifted padded-image grids."""
    row, col = shift
    height, width = template.shape[-2:]
    rows = torch.arange(height, device=template.device, dtype=torch.float64) + pad
    cols = torch.arange(width, device=template.device, dtype=torch.float64) + pad
    rows = torch.sin(math.pi * ((rows - row).remainder(psize) + .5) / psize).square()
    cols = torch.sin(math.pi * ((cols - col).remainder(psize) + .5) / psize).square()
    return rows[:, None] * cols[None, :]


class PABalanceStats:
    """One sample/trajectory, frozen layer metric and causal cross-layer history."""

    def __init__(self, sigma_data, pad, psize, block_size=8, epsilon=1e-12):
        if block_size < 1 or not math.isfinite(epsilon) or epsilon <= 0 or psize < 1:
            raise ValueError("PA block size, patch size and epsilon must be positive.")
        self.sigma_data = float(sigma_data)
        self.pad, self.psize = pad, psize
        self.block_size, self.epsilon = block_size, epsilon
        self.m = self.weight = None
        self.slope = 0.0
        self.last_sigma = None

    def begin_level(self, sigma, template):
        self.sigma = float(sigma)
        self.coefficients = edm_coefficients(sigma, self.sigma_data)
        if not template.is_complex() or template.ndim != 4 or template.shape[:2] != (1, 1):
            raise ValueError("PA statistics require one [1,1,H,W] complex state per trajectory.")
        coarse = region_mean(template.real.double(), self.block_size)
        if self.m is None:
            self.m = torch.zeros_like(coarse)
            self.weight = torch.zeros_like(coarse)
        elif self.m.shape != coarse.shape or self.m.device != template.device:
            raise ValueError("Create independent PA statistics for a new FOV/device.")
        elif self.last_sigma is not None:
            decay = self.sigma / self.last_sigma
            self.m = self.m * decay
            self.weight = self.weight * decay
        vf = self.m / (self.weight + self.epsilon)
        # c_out/sigma^2 = sigma_data/(sigma*hypot(sigma,sigma_data)).
        score_scale = (self.sigma_data / math.hypot(self.sigma, self.sigma_data)) / self.sigma
        if not math.isfinite(score_scale) or score_scale > math.sqrt(torch.finfo(torch.float64).max):
            raise ValueError("sigma is too small for PA variance accumulation.")
        self.frozen_variance = region_expand(vf * score_scale ** 2, template.shape, self.block_size).clone()
        if not torch.isfinite(self.frozen_variance).all():
            raise FloatingPointError("PA variance is not finite; inspect sigma and innovation magnitude.")
        self.previous_f = self.previous_z = self.previous_window = None
        self.fit_num = template.real.new_zeros((), dtype=torch.float64)
        self.fit_den = torch.zeros_like(self.fit_num)
        self.innovation_sum = torch.zeros_like(self.fit_num)
        self.innovation_max = torch.zeros_like(self.fit_num)
        self.reliability_sum = torch.zeros_like(self.fit_num)
        self.pairs = 0
        return self.frozen_variance

    @torch.no_grad()
    def observe(self, state, denoised, shift):
        c_skip, c_out, c_in = self.coefficients
        f = (denoised.detach() - c_skip * state.detach()) / c_out
        z = c_in * state.detach()
        if not torch.isfinite(f).all() or not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite EDM residual/state in PA statistics.")
        window = interior_window(state, shift, self.pad, self.psize)
        if self.previous_f is not None:
            df = f - self.previous_f
            dz = z - self.previous_z
            residual = df - self.slope * dz
            nu = region_mean(squared_modulus(residual), self.block_size) / 4
            q = region_mean(squared_modulus(dz), self.block_size) / 2
            old_energy = region_mean(squared_modulus(self.previous_f), self.block_size) / 2
            if not torch.isfinite(nu + q + old_energy).all():
                raise FloatingPointError("PA innovation statistics exceed the accumulation dtype range.")
            reliability = 1 / (1 + q / (old_energy + self.epsilon))
            self.m = .5 * self.m + .5 * reliability * nu
            self.weight = .5 * self.weight + .5 * reliability
            fit_weight = window * self.previous_window
            self.fit_num += (fit_weight * (dz.conj().to(torch.complex128) * df).real).sum()
            self.fit_den += (fit_weight * squared_modulus(dz)).sum()
            self.innovation_sum += nu.mean()
            self.innovation_max = torch.maximum(self.innovation_max, nu.max())
            self.reliability_sum += reliability.mean()
            self.pairs += 1
        self.previous_f, self.previous_z, self.previous_window = f, z, window

    def end_level(self, gamma):
        frozen_slope = self.slope
        if self.pairs:
            fitted = (self.fit_num / (self.fit_den + self.epsilon)).item()
            self.slope = .5 * self.slope + .5 * fitted
        if not math.isfinite(self.slope):
            raise FloatingPointError("Nonfinite fitted PA state slope.")
        self.last_sigma = self.sigma
        variance = self.frozen_variance
        ratio = gamma * self.sigma ** 2 * variance / 2
        return {
            "sigma": self.sigma, "gamma": gamma,
            "v_mean": variance.mean().item(), "v_max": variance.max().item(),
            "relative_noise_ratio_mean": ratio.mean().item(), "relative_noise_ratio_max": ratio.max().item(),
            "state_slope": frozen_slope, "next_state_slope": self.slope,
            "innovation_mean": (self.innovation_sum / max(1, self.pairs)).item(),
            "innovation_max": self.innovation_max.item(), "innovation_pairs": self.pairs,
            "reliability_mean": (self.reliability_sum / max(1, self.pairs)).item(),
        }
