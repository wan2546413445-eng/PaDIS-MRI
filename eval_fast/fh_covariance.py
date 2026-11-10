"""Free Hunch precision updates in the real, unpadded PaDIS image space."""

import math

import numpy as np
import torch


CACHE_FORMAT = "padis-fh-dct-v3"
NORMALIZATION = "padis_native_bart_acs99"
CHANNEL_ORDER = ("real", "imag")
NOISE_ESTIMATOR = "training_coil_acquired_fov_background_kspace_trace"
VARIANCE_FLOOR = 1e-8


def load_fh_cache(path, image_size):
    with np.load(path, allow_pickle=False) as cache:
        if cache["format"].item() != CACHE_FORMAT:
            raise ValueError(f"Rebuild {path} from PaDIS training noisy.pt and ksp/ with the v3 script.")
        variance = cache["variance"]
        noise_variance = float(cache["noise_variance"].item())
        if (
            tuple(cache["channel_order"]) != CHANNEL_ORDER
            or cache["normalization"].item() != NORMALIZATION
            or cache["noise_estimator"].item() != NOISE_ESTIMATOR
            or tuple(cache["fov_shape"]) != (image_size, image_size)
            or variance.shape != (2, image_size, image_size)
            or cache["sample_count"].item() < 2
            or not np.isfinite(variance).all()
            or np.any(variance < 0)
            or not math.isfinite(noise_variance)
            or noise_variance <= 0
        ):
            raise ValueError(f"Incompatible PaDIS FH training covariance cache: {path}")
        source_files = cache["source_files"].tolist()
    return torch.from_numpy(variance), noise_variance, source_files


class OrthoDCT:
    """Separable orthonormal DCT-II and its inverse using real FFTs."""

    def __init__(self, reference):
        self.axes = []
        for length in reference.shape[-2:]:
            k = torch.arange(length, dtype=reference.dtype, device=reference.device)
            phase = torch.polar(torch.ones_like(k), -math.pi / (2 * length) * k)
            scale = torch.full_like(k, 1 / math.sqrt(2 * length))
            scale[0] /= math.sqrt(2)
            self.axes.append((phase, scale))

    def forward(self, image):
        for dim, (phase, scale) in zip((-2, -1), self.axes):
            rows = image.transpose(dim, -1)
            even = torch.cat((rows, rows.flip(-1)), dim=-1)
            spectrum = torch.fft.rfft(even)[..., :-1]
            image = ((spectrum * phase).real * scale).transpose(dim, -1)
        return image

    def inverse(self, coefficients):
        for dim, (phase, scale) in reversed(list(zip((-2, -1), self.axes))):
            rows = coefficients.transpose(dim, -1)
            spectrum = (rows / scale) * phase.conj()
            spectrum = torch.cat((spectrum, torch.zeros_like(spectrum[..., :1])), dim=-1)
            coefficients = torch.fft.irfft(spectrum)[..., :scale.numel()].transpose(dim, -1)
        return coefficients


class FHPrecision:
    """P = diag(diagonal) + modes @ diag(weights) @ modes.T in DCT coordinates."""

    def __init__(self, data_variance, noise_variance, max_rank=16):
        self.dct = OrthoDCT(data_variance)
        self.diagonal = data_variance.flatten().clamp_min(VARIANCE_FLOOR).reciprocal()
        self.diagonal.add_(1 / noise_variance)
        self.noise_variance = noise_variance
        self.max_rank = max_rank
        self.modes = self.diagonal.new_empty((self.diagonal.numel(), 0))
        self.weights = self.diagonal.new_empty(0)

    def transfer(self, noise_variance):
        # A decreasing VE schedule adds a positive multiple of the identity.
        self.diagonal.add_(1 / noise_variance - 1 / self.noise_variance)
        self.noise_variance = noise_variance

    def _apply_dct(self, vector):
        result = self.diagonal * vector
        if self.weights.numel():
            result = result + self.modes @ (self.weights * (self.modes.T @ vector))
        return result

    def apply(self, image):
        coefficients = self.dct.forward(image)
        result = self._apply_dct(coefficients.flatten())
        return self.dct.inverse(result.view_as(coefficients))

    def update(self, displacement, response):
        """Apply inverse covariance BFGS, or return the reason it was skipped."""
        s, q = self.dct.forward(torch.stack((displacement, response))).flatten(1)
        sq = torch.dot(s, q)
        threshold = 1e-6 * torch.linalg.vector_norm(s) * torch.linalg.vector_norm(q)
        curvature, minimum_curvature = torch.stack((sq, threshold)).tolist()
        if not (math.isfinite(curvature) and math.isfinite(minimum_curvature)):
            return "nonfinite_secant"
        if curvature <= 0:
            return "nonpositive_curvature"
        if curvature <= minimum_curvature:
            return "weak_positive_curvature"

        pq = self._apply_dct(q)
        new_modes = torch.cat((self.modes, torch.stack((s, pq), dim=1) / sq.sqrt()), dim=1)
        curvature_ratio = (1 + torch.dot(q, pq) / sq).view(1)
        small = torch.diag(torch.cat((self.weights, curvature_ratio, sq.new_zeros(1))))
        small[-2, -1] = small[-1, -2] = -1

        # Compress the correction relative to the positive diagonal. Discarding
        # modes replaces their relative eigenvalues by 1. Clamp retained
        # eigenvalues of the full relative precision away from zero.
        sqrt_diagonal = self.diagonal.sqrt().unsqueeze(1)
        basis, factor = torch.linalg.qr(new_modes / sqrt_diagonal, mode="reduced")
        # Higher precision here costs only rank-sized arithmetic and reduces
        # cancellation between the two terms of the inverse-BFGS correction.
        factor = factor.double()
        reduced = factor @ small.double() @ factor.T
        weights, rotation = torch.linalg.eigh((reduced + reduced.T) * 0.5)
        keep = weights.abs().argsort(descending=True)[:self.max_rank]
        self.modes = ((basis @ rotation[:, keep].to(basis.dtype)) * sqrt_diagonal).contiguous()
        self.weights = weights[keep].clamp_min(-1 + 1e-4).to(basis.dtype)
        return None


class FHProfile:
    """CUDA intervals, read only after the enclosing reconstruction synchronizes."""

    def __init__(self):
        self.events = {name: [] for name in (
            "probe_forward", "covariance_update", "cg_total", "cg_precision",
        )}

    def begin(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(self, name, start):
        self.events[name].append((start, self.begin()))

    def summary(self):
        return {
            name: sum((start.elapsed_time(end) for start, end in pairs), 0.0) / 1000
            for name, pairs in self.events.items()
        }
