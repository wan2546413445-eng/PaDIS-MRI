"""Two-level CPU flow check with a deterministic patch denoiser."""

from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "eval")]

import denoise_padding
from eval_fast import recon_fast
from eval_fast.recon_fast import _channels
from inverse_operators import MRI_utils
from recon import _ve_base_schedule


def main():
    width, pad = 4, 2
    padding = (pad,) * 4
    fov = (..., slice(pad, pad + width), slice(pad, pad + width))
    shape = (1, 1, width, width)
    truth = torch.full(shape, 0.7 + 0.3j)
    maps = torch.cat((torch.ones_like(truth), 0.5j * torch.ones_like(truth)), dim=1)
    op = MRI_utils(torch.ones_like(truth.real), maps)
    measurement = op.forward(_channels(truth))
    adjoint = op.adjoint(measurement)
    net = SimpleNamespace(round_sigma=torch.as_tensor)
    sigmas = _ve_base_schedule(net, 2, 0.003, 10.0, 7.0, truth.device).float()
    padded = F.pad(adjoint, padding)
    predictions = [F.pad(truth * fraction, padding) for fraction in (0.2, 0.4, 0.5, 0.6)]
    events, inputs, sigmas_seen = [], [], []
    eta, cg_lambda, cg_gain = 0.8, 0.1, 0.75
    counts = dict.fromkeys(("main_denoise", "cg_solves", "cg_iterations", "dps_backprops"), 0)

    def denoise(_net, channels, sigma, *_args, **_kwargs):
        assert not torch.is_grad_enabled()
        events.append("denoise")
        sigmas_seen.append(sigma)
        inputs.append(torch.complex(channels[:, 0], channels[:, 1]).unsqueeze(1))
        return _channels(predictions[len(inputs) - 1])

    original_cg = recon_fast._cg_correction

    def cg(op, adjoint, v, *args, **kwargs):
        events.append("cg")
        sweep = len(inputs)
        torch.testing.assert_close(v, (predictions[sweep - 2][fov] + predictions[sweep - 1][fov]) / 2)
        return original_cg(op, adjoint, v, *args, **kwargs)

    generator = torch.Generator().manual_seed(123)
    noises = [torch.randn(padded.shape, dtype=padded.dtype, generator=generator) for _ in range(4)]
    torch.manual_seed(123)
    with (
        patch.object(denoise_padding, "denoisedFromPatches", denoise),
        patch.object(denoise_padding, "getIndices", wraps=denoise_padding.getIndices) as get_indices,
        patch.object(recon_fast, "_cg_correction", cg),
    ):
        result = recon_fast.reconstruct_ct_cg_ddim(
            net, None, op, measurement, num_steps=2, inner_loops=2, pad=pad, psize=2,
            ddim_eta=eta, cg_lambda=cg_lambda, cg_iters=5, cg_gain=cg_gain, counts=counts,
        )
        assert get_indices.call_count == 4

    assert events == ["denoise", "denoise", "cg"] * 2
    assert counts == {"main_denoise": 4, "cg_solves": 2, "cg_iterations": 10, "dps_backprops": 0}
    torch.testing.assert_close(torch.stack(sigmas_seen), sigmas[:2].repeat_interleave(2))
    torch.testing.assert_close(inputs[0], padded + sigmas[0] * noises[0])
    torch.testing.assert_close(inputs[1], predictions[0] + sigmas[0] * noises[1])

    def corrected(first, second):
        v = (predictions[first][fov] + predictions[second][fov]) / 2
        normal = maps.abs().square().sum(dim=1, keepdim=True)
        return v + cg_gain * (adjoint - normal * v) / (normal + cg_lambda)

    fused = (inputs[0] - predictions[0] + inputs[1] - predictions[1]) / (sigmas[0] * sqrt(2))
    expected_next = F.pad(corrected(0, 1), padding) + sigmas[1] * (sqrt(1 - eta**2) * fused + eta * noises[2])
    torch.testing.assert_close(inputs[2], expected_next)
    torch.testing.assert_close(inputs[3], predictions[2] + sigmas[1] * noises[3])
    torch.testing.assert_close(result, corrected(2, 3).squeeze(1))
    torch.testing.assert_close(torch.get_rng_state(), generator.get_state())
    print("ct_cg_ddim 2×2 flow passed (4 denoise / 2 CG / 0 DPS)")


if __name__ == "__main__":
    main()
