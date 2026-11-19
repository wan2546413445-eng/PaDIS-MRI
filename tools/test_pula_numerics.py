"""CPU numerical checks for the two PaDIS pULA variance conventions."""

from math import sqrt
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pula import recon_pula
from eval_pula.cg import solve_cg
from eval_pula.recon_pula import PULASystem, native_complex_noise, real_unit_noise, reconstruct, whiten_measurement
from eval_pula.score_opa import real_channels


SEED = 123
GAMMA = 0.5
SIGMA = 0.73


class MaskedCoilOperator:
    def __init__(self, mask, maps):
        self.mask = mask
        self.maps = maps

    def forward(self, channels):
        image = torch.complex(channels[:, :1], channels[:, 1:])
        return self.mask * self.maps * image

    def adjoint(self, data):
        return (self.maps.conj() * self.mask * data).sum(dim=1, keepdim=True)


class ToyOracle:
    pad = 1
    image_size = 2

    def __init__(self):
        self.net = torch.nn.Identity()
        self.sigmas = []

    def denoise(self, state, sigma, shift):
        self.sigmas.append(sigma)
        return self.net(state * (1 - 0.25 * sigma) + 0.02 * shift[0])

    denoise_padded = denoise


def check(label, actual, expected, *, rtol=2e-6, atol=2e-7):
    difference = (actual - expected).to(torch.complex128)
    reference = expected.to(torch.complex128).norm().item()
    error = difference.norm().item()
    relative = error / reference if reference else (float("inf") if error else 0.0)
    print(f"{label}: max error={difference.abs().max().item():.3e}, relative error={relative:.3e}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def check_random_draws(measurement, image):
    real_rng = torch.Generator().manual_seed(SEED)
    complex_rng = torch.Generator().manual_seed(SEED)
    for label, template in (("measurement", measurement), ("image", image)):
        real = real_unit_noise(template, real_rng)
        complex_ = native_complex_noise(template, complex_rng)
        check(f"{label} RNG scale", real, sqrt(2.0) * complex_, rtol=0, atol=0)
        assert torch.equal(real_rng.get_state(), complex_rng.get_state())


def check_legacy(raw_operator, raw_measurement, covariance, real, image):
    if isinstance(covariance, float):
        maps = sqrt(2.0 / covariance) * raw_operator.maps
        measurement = sqrt(2.0 / covariance) * raw_measurement
    else:
        factor = torch.linalg.cholesky(covariance.to(torch.complex128) / 2).to(raw_measurement.dtype)
        coils = raw_measurement.shape[1]
        maps, measurement = (
            torch.linalg.solve_triangular(factor, value.reshape(coils, -1), upper=False).reshape_as(value)
            for value in (raw_operator.maps, raw_measurement)
        )
    check("legacy whitening maps", real.operator.maps, maps, rtol=0, atol=0)
    check("legacy whitening measurement", real.y, measurement, rtol=0, atol=0)
    operator = MaskedCoilOperator(raw_operator.mask, maps)
    rng = torch.Generator().manual_seed(SEED)

    def noise(template):
        y_noise, x_noise = (sqrt(2.0) * torch.randn(t.shape, dtype=t.dtype, generator=rng)
                            for t in (measurement, template))
        return operator.adjoint(y_noise) + x_noise / SIGMA, x_noise

    def normal(v):
        return operator.adjoint(operator.forward(real_channels(v))) + v / SIGMA ** 2

    mean_rhs = operator.adjoint(measurement)
    random_rhs, image_noise = noise(mean_rhs)
    init_rhs = mean_rhs + random_rhs
    warm = SIGMA * image_noise
    legacy_state, _ = solve_cg(normal, init_rhs, warm, 10, 1e-6)
    state, _ = real.solve(*real.initialization_rhs(SIGMA), SIGMA)
    check("legacy initialization", state, legacy_state, rtol=0, atol=0)

    prior = (ToyOracle().denoise(image, SIGMA, (0, 0)) - image) / SIGMA ** 2
    likelihood = operator.adjoint(measurement - operator.forward(real_channels(image)))
    random_rhs, image_noise = noise(image)
    diffusion = sqrt(2.0 * GAMMA)
    rhs = GAMMA * (likelihood + prior) + diffusion * random_rhs
    warm = GAMMA * SIGMA ** 2 * prior + diffusion * SIGMA * image_noise
    actual_rhs, actual_warm = real.update_rhs(image, prior, SIGMA, GAMMA)
    check("legacy update RHS", actual_rhs, rhs, rtol=0, atol=0)
    check("legacy update warm start", actual_warm, warm, rtol=0, atol=0)
    check("legacy delta", real.solve(actual_rhs, actual_warm, SIGMA)[0],
          solve_cg(normal, rhs, warm, 10, 1e-6)[0], rtol=0, atol=0)
    assert torch.equal(rng.get_state(), real.generator.get_state())


def check_conventions(raw_operator, raw_measurement, covariance, image):
    real_op, b = whiten_measurement(raw_operator, raw_measurement, covariance)
    paper_op, y = whiten_measurement(raw_operator, raw_measurement, covariance, "paper_complex")
    real = PULASystem(real_op, b, torch.Generator().manual_seed(SEED), 10, 1e-6)
    paper = PULASystem(paper_op, y, torch.Generator().manual_seed(SEED), 10, 1e-6, "paper_complex")
    assert real.convention == "padis_real"
    check_legacy(raw_operator, raw_measurement, covariance, real, image)
    check("whitened maps B vs sqrt(2) A", real_op.maps, sqrt(2.0) * paper_op.maps)
    check("whitened measurement b vs sqrt(2) y", b, sqrt(2.0) * y)

    normal_images = []

    def capture_normal(normal, *args):
        normal_images.append(normal(image))
        return solve_cg(normal, *args)

    for system in (real, paper):
        system.generator.manual_seed(SEED)
    real_rhs, real_warm = real.initialization_rhs(SIGMA)
    paper_rhs, paper_warm = paper.initialization_rhs(SIGMA)
    check("initialization RHS", real_rhs, 2 * paper_rhs)
    check("initialization warm start", real_warm, paper_warm)
    with patch.object(recon_pula, "solve_cg", side_effect=capture_normal):
        real_state, _ = real.solve(real_rhs, real_warm, SIGMA)
        paper_state, _ = paper.solve(paper_rhs, paper_warm, SIGMA)
    check("CG operator H_real vs 2 H_paper", normal_images[0], 2 * normal_images[1])
    check("initialized state", real_state, paper_state)
    assert torch.equal(real.generator.get_state(), paper.generator.get_state())

    denoised = ToyOracle().denoise(image, SIGMA, (0, 0))
    real_score = (denoised - image) / real.system_sigma(SIGMA) ** 2
    paper_score = (denoised - image) / paper.system_sigma(SIGMA) ** 2
    real_rhs, real_warm = real.update_rhs(image, real_score, SIGMA, GAMMA)
    paper_rhs, paper_warm = paper.update_rhs(image, paper_score, SIGMA, GAMMA)
    check("update RHS", real_rhs, 2 * paper_rhs)
    check("update warm start", real_warm, paper_warm)
    real_delta, _ = real.solve(real_rhs, real_warm, SIGMA)
    paper_delta, _ = paper.solve(paper_rhs, paper_warm, SIGMA)
    check("CG delta", real_delta, paper_delta)
    assert torch.equal(real.generator.get_state(), paper.generator.get_state())

    def run(system, save):
        system.generator.manual_seed(SEED)
        oracle, handoff = ToyOracle(), []
        result, _, timing, _ = reconstruct(
            system, oracle, (1.0, SIGMA), ((0, 0),), GAMMA,
            dps_operator=raw_operator, dps_measurement=raw_measurement,
            dps_sigmas=(0.1, 0.05), dps_shifts=((0, 0), (0, 0)),
            dps_inner_loops=1, report=lambda _: None, save_handoff=handoff.append if save else None,
        )
        assert oracle.sigmas == [SIGMA, 0.1, 0.05]
        assert timing["langevin_updates"] == 1 and timing["dps_updates"] == 2
        assert timing["pula_convention"] == system.convention
        return result, handoff, system.generator.get_state()

    real_final, real_handoff, real_rng = run(real, True)
    paper_final, paper_handoff, paper_rng = run(paper, True)
    check("full pULA update (handoff)", real_handoff[0], paper_handoff[0])
    check("synthetic DPS final state", real_final, paper_final)
    assert torch.equal(real_rng, paper_rng)
    no_save_final, _, no_save_rng = run(real, False)
    check("handoff callback leaves final state unchanged", real_final, no_save_final, rtol=0, atol=0)
    assert torch.equal(real_rng, no_save_rng)


def main():
    rng = torch.Generator().manual_seed(19)
    mask = torch.tensor([[[[True, False], [True, True]]]])
    image = torch.randn((1, 1, 2, 2), dtype=torch.complex64, generator=rng)
    maps = torch.randn((1, 2, 2, 2), dtype=torch.complex64, generator=rng)
    operator = MaskedCoilOperator(mask, maps)
    measurement = operator.forward(real_channels(image))
    check_random_draws(measurement, image)
    correlated = torch.tensor([[0.62, 0.12 - 0.08j], [0.12 + 0.08j, 0.44]], dtype=torch.complex64)
    for covariance in (0.27, correlated):
        print("Scalar covariance" if isinstance(covariance, float) else "Hermitian coil covariance")
        check_conventions(operator, measurement, covariance, image)


if __name__ == "__main__":
    main()
