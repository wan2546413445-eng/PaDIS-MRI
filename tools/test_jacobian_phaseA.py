"""CPU-only Phase A checks. Run from the repository root: pytest -q tools/test_jacobian_phaseA.py"""
from pathlib import Path
import json
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_pula.jacobian_diagnostics import (
    direction_cosine, observe, partition_masks, real_dot, write_outputs,
)


def to_2ch(x):
    return torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2).contiguous()


class ToyMRI:
    """Multi-coil orthonormal FFT model, consistent with MRI_utils.forward/adjoint."""
    def __init__(self, size, coils=2):
        self.maps = torch.randn(1, coils, size, size, dtype=torch.complex128)
        self.mask = (torch.rand(size, size) > .3).to(torch.complex128)
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        z = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())[:, None]
        return self.mask * torch.fft.fft2(self.maps * z, norm="ortho")

    def adjoint(self, y):
        self.adjoint_calls += 1
        x = torch.fft.ifft2(self.mask * y, norm="ortho")
        return (self.maps.conj() * x).sum(dim=1, keepdim=True)


def generated_step(size=16, pad=4, with_diagnostic=True, seed=123):
    """Toy version of unchanged DPS update; explicit noise draws are identical."""
    source = torch.Generator(device="cpu").manual_seed(seed)
    system = ToyMRI(size)
    # Fix independent setup randomness, then compare ON/OFF with identical setup.
    target = torch.full((1, 1, size, size), .15+.07j, dtype=torch.complex128)
    measurement = system.forward(to_2ch(target))
    current = F.pad(target, (pad,) * 4)
    records = []
    for idx in range(3):
        current = current.detach().requires_grad_(True)
        noise = torch.randn(current.shape, generator=source, dtype=current.dtype)
        u = current + .03 * noise
        d = 0.8 * u + .1 * u.conj()
        d_fov = d[..., pad:pad+size, pad:pad+size]
        residual = measurement - system.forward(to_2ch(d_fov))
        norm = torch.linalg.vector_norm(residual.reshape(1, -1), dim=-1)
        grad = torch.autograd.grad(norm.square().sum(), current)[0]
        if with_diagnostic and idx in (0, 2):
            records.append(observe(current, grad, residual, system,
                                   sigma=.03, outer=idx, inner=0, shift=(1, 2),
                                   pad=pad, image_size=size, psize=4,
                                   network_forwards=idx+1, network_backwards=idx+1,
                                   boundary_band=1))
        score = (d - u) / (.03 ** 2)
        alpha = .5 * (.03 ** 2)
        current = current - (3. / norm[:, None, None, None]) * grad
        current = current + (alpha / 2) * score
        if idx < 2:
            current = current + (alpha ** .5) * torch.randn(current.shape, generator=source, dtype=current.dtype)
    return current.detach(), source.get_state(), system, records


def test_complex_inner_product_and_direction():
    a = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex128)
    b = 2 * a
    assert real_dot(a, b).item() == 2 * (a.abs().square().sum().item())
    assert direction_cosine(a, b) == pytest.approx(1.)
    assert direction_cosine(a, 1j*a) == pytest.approx(0., abs=1e-12)
    assert direction_cosine(a, -a) == pytest.approx(-1.)


def test_mri_toy_adjoint():
    torch.manual_seed(42)
    op = ToyMRI(12, 3)
    x = torch.randn(1, 1, 12, 12, dtype=torch.complex128)
    y = torch.randn(1, 3, 12, 12, dtype=torch.complex128)
    lhs = real_dot(op.forward(to_2ch(x)), y)
    rhs = real_dot(x, op.adjoint(y))
    assert torch.allclose(lhs, rhs, atol=1e-11)


def test_actual_mri_utils_adjoint_if_repo_available():
    source = Path(__file__).resolve().parents[1] / "eval" / "inverse_operators.py"
    if not source.exists():
        pytest.skip("Real repository eval/inverse_operators.py unavailable in standalone ZIP")
    sys.path.insert(0, str(source.parent))
    from inverse_operators import MRI_utils
    torch.manual_seed(42)
    mask = (torch.rand(10, 10) > 0.5).to(torch.complex64)
    maps = torch.randn(1, 3, 10, 10, dtype=torch.complex64)
    op = MRI_utils(mask, maps)
    x = torch.randn(1, 1, 10, 10, dtype=torch.complex64)
    y = torch.randn(1, 3, 10, 10, dtype=torch.complex64)
    assert torch.allclose(real_dot(op.forward(to_2ch(x)), y),
                          real_dot(x, op.adjoint(y)), rtol=1e-5, atol=1e-5)


def test_crop_and_zero_padding_adjoint():
    x = torch.randn(1, 1, 12, 12, dtype=torch.complex128)
    y = torch.randn(1, 1, 8, 8, dtype=torch.complex128)
    assert torch.allclose(real_dot(x[..., 2:10, 2:10], y),
                          real_dot(x, F.pad(y, (2, 2, 2, 2))))
    assert torch.equal(F.pad(y, (2, 2, 2, 2))[..., 2:10, 2:10], y)


def test_linear_denoiser_jacobian_and_factor_two():
    size, pad = 8, 4
    torch.manual_seed(37)
    current = torch.randn(1, 1, 16, 16, dtype=torch.complex128).requires_grad_(True)
    op = ToyMRI(size)
    a = .65 + .45j
    d = a * current
    residual = -op.forward(to_2ch(d[..., pad:pad+size, pad:pad+size]))
    grad = torch.autograd.grad(residual.abs().square().sum(), current)[0]
    v = F.pad(-op.adjoint(residual), (pad, pad, pad, pad))
    expected = 2 * a.conjugate() * v
    assert torch.allclose(grad, expected, atol=1e-10, rtol=1e-10)
    row = observe(current, grad, residual, op, sigma=.1, outer=0, inner=0,
                  shift=(0, 0), pad=pad, image_size=size, psize=4,
                  network_forwards=1, network_backwards=1, boundary_band=1)
    assert row["jacobian_norm_ratio"] == pytest.approx(abs(a))
    assert row["jacobian_cosine"] == pytest.approx(a.real / abs(a), abs=1e-10)
    assert row["padding_g_energy_fraction"] == pytest.approx(0.)


def test_padding_energy_ratio_and_boundaries():
    mask = partition_masks(64, 64, pad=16, image_size=32, psize=16,
                           shift=(7, 13), band=2)
    assert mask["boundary"][23, 30] and mask["boundary"][22, 30]
    assert not mask["boundary"][25, 32]
    assert mask["boundary"][27, 29] and mask["boundary"][27, 28]
    assert not mask["boundary"][27, 31]
    assert mask["padding"][10, 10] and not mask["boundary"][10, 10]
    assert (mask["padding"] | mask["boundary"] | mask["interior"]).all()
    assert not (mask["padding"] & mask["boundary"]).any()
    assert not partition_masks(64, 64, 16, 32, 16, (0, 0), 2)["boundary"][23, 25]


def test_padding_energy_fraction_is_computed_on_g_not_v():
    size, pad = 8, 4
    op = ToyMRI(size)
    current = torch.zeros(1, 1, 16, 16, dtype=torch.complex128)
    grad = torch.ones_like(current) * 2
    residual = torch.zeros(1, 2, size, size, dtype=torch.complex128)
    record = observe(current, grad, residual, op, sigma=1, outer=0, inner=0,
                     shift=(0, 0), pad=pad, image_size=size, psize=4,
                     network_forwards=1, network_backwards=1, boundary_band=1)
    assert record["padding_g_energy_fraction"] == pytest.approx(1 - size**2/(size+2*pad)**2)
    assert record["zero_gradient"] is False
    assert record["jacobian_cosine"] is None


def test_rng_state_untouched():
    torch.manual_seed(123)
    global_before = torch.random.get_rng_state().clone()
    private = torch.Generator().manual_seed(987)
    private_before = private.get_state().clone()
    op = ToyMRI(8)
    # The operator constructor draws randomness; only check the diagnostic itself.
    global_before = torch.random.get_rng_state().clone()
    x = torch.zeros(1, 1, 16, 16, dtype=torch.complex128)
    observe(x, x, torch.zeros(1, 2, 8, 8, dtype=x.dtype), op,
            sigma=1., outer=0, inner=0, shift=(0, 0), pad=4,
            image_size=8, psize=4, network_forwards=0, network_backwards=0,
            boundary_band=1)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert torch.equal(private.get_state(), private_before)


def test_dps_reconstruction_bitwise_and_network_calls_unchanged():
    torch.manual_seed(42)
    xoff, rngoff, off, rows_off = generated_step(with_diagnostic=False)
    torch.manual_seed(42)
    xon, rngon, on, rows_on = generated_step(with_diagnostic=True)
    assert torch.equal(xoff, xon)
    assert torch.equal(rngoff, rngon)
    assert off.forward_calls == on.forward_calls
    assert on.adjoint_calls == off.adjoint_calls + len(rows_on)
    assert len(rows_off) == 0 and len(rows_on) == 2


def test_json_csv_finite_fields(tmp_path):
    torch.manual_seed(42)
    _, _, _, rows = generated_step(with_diagnostic=True)
    write_outputs(tmp_path, rows, {"model_path": "/synthetic/toy.pkl", "sample_index": 0})
    obj = json.loads((tmp_path / "jacobian_diagnostics.json").read_text())
    assert obj["metadata"]["sample_index"] == 0
    assert len(obj["records"]) == 2
    assert "outer_level,inner_step,sigma" in (tmp_path / "jacobian_diagnostics.csv").read_text()
    assert all(r["finite"] and r["diagnostic_wall_seconds"] >= 0 for r in rows)
    assert all(r["extra_network_forwards"] == 0 and r["extra_network_backwards"] == 0 for r in rows)


def test_nonfinite_diagnostic_serializes_as_null(tmp_path):
    op = ToyMRI(8)
    current = torch.zeros(1, 1, 16, 16, dtype=torch.complex128)
    grad = torch.full_like(current, complex(float("nan"), 0.0))
    residual = torch.zeros(1, 2, 8, 8, dtype=current.dtype)
    rec = observe(current, grad, residual, op, sigma=1., outer=0, inner=0,
                  shift=(0, 0), pad=4, image_size=8, psize=4,
                  network_forwards=1, network_backwards=1, boundary_band=1)
    assert rec["status"] == "nonfinite"
    write_outputs(tmp_path, [rec], {"sample_index": 0})
    obj = json.loads((tmp_path / "jacobian_diagnostics.json").read_text())
    assert obj["records"][0]["jacobian_cosine"] is None


def test_complex_autograd_real_directional_derivative():
    torch.manual_seed(19)
    op = ToyMRI(6)
    state = torch.randn(1, 1, 6, 6, dtype=torch.complex128).requires_grad_(True)
    direction = torch.randn_like(state)
    direction = direction / torch.linalg.vector_norm(direction)
    coefficient = .8 + .3j
    measurement = torch.zeros(1, 2, 6, 6, dtype=torch.complex128)

    def objective(z):
        return (op.forward(to_2ch(coefficient * z)) - measurement).abs().square().sum()

    grad = torch.autograd.grad(objective(state), state)[0]
    step = 1e-6
    finite_difference = ((objective(state.detach() + step * direction) -
                          objective(state.detach() - step * direction)) / (2 * step)).item()
    assert finite_difference == pytest.approx(real_dot(grad, direction).item(), rel=1e-8, abs=1e-8)

# Acceptance tests for the REAL output pair checker (synthetic saved runs; no GPU).
from copy import deepcopy
import numpy as np
from tools.compare_jacobian_phaseA_recons import COUNT_KEYS, compare


PAIR_CONFIG = {
    "model_path": "/model/E0_10k.pkl", "val_dir": "/data/T1FLAIR",
    "sample_index": 0, "mask_select": 7, "seed": 123, "grid_seed": 123,
    "score_mode": "patch", "grid_mode": "random_step", "pula_convention": "padis_real",
    "gamma": 0.5, "cg_iters": 10, "cg_rtol": 1e-6,
    "sigma_max": 1.0, "sigma_min": 0.01, "sigma_schedule": "exponential",
    "num_noise_levels": 48, "langevin_steps": 4,
    "dps_zeta": 3.0, "dps_base_steps": 78, "dps_rho": 7.0,
    "dps_sigma_max": 10.0, "dps_sigma_min": 0.003, "dps_tail_steps": 24,
    "dps_inner_loops": 10, "pad": 64, "psize": 64, "image_size": 384,
    "patch_batch": 49, "patch_checkpoint": "0",
    "noise_calibration": {"variance": 0.01, "source": "sample_0.pt"},
    "dps_shifts": [[1, 3], [5, 8]], "pula_shifts_by_level": [[[1, 1], [2, 2]]],
    "dps_tail_sigmas": [0.1, 0.01], "pula_levels": 23,
}


def make_saved_pair(tmp_path):
    on, off = tmp_path / "on", tmp_path / "off"
    for name, root in (("on", on), ("off", off)):
        dest = root / "random_step" / "sample_0" / "evaluate"
        (dest / "recons").mkdir(parents=True)
        np.save(dest / "recons" / "recon_patch_0.npy",
                np.array([[1+2j, 3-4j]], dtype=np.complex64))
        config = deepcopy(PAIR_CONFIG)
        if name == "on":
            config.update(diagnose_dps_jacobian=True, jacobian_levels=[0, 11, 23],
                          jacobian_inner_steps=[0])
        (dest / "config.json").write_text(json.dumps(config))
        (dest / "timing.json").write_text(json.dumps({key: i for i, key in enumerate(COUNT_KEYS)}))
    return on, off


def write_config_field(root, key, value):
    path = root / "random_step" / "sample_0" / "evaluate" / "config.json"
    obj = json.loads(path.read_text())
    obj[key] = value
    path.write_text(json.dumps(obj))


def test_paired_acceptance_excludes_only_diagnostic_options(tmp_path):
    on, off = make_saved_pair(tmp_path)
    report = compare(on, off)
    assert report["bitwise_equal"] and report["config_same"]
    assert report["network_and_cg_counts_same"]
    assert report["config_keys_compared"] == len(PAIR_CONFIG)
    assert report["config_mismatches"] == {}


@pytest.mark.parametrize("field", [
    "gamma", "cg_iters", "cg_rtol", "sigma_max", "sigma_min",
    "dps_zeta", "dps_base_steps", "dps_rho", "dps_sigma_max", "dps_sigma_min",
    "pad", "psize", "image_size", "val_dir", "patch_batch", "patch_checkpoint",
    "noise_calibration", "dps_shifts", "pula_shifts_by_level", "model_path",
])
def test_paired_acceptance_rejects_any_relevant_config_change(tmp_path, capsys, field):
    on, off = make_saved_pair(tmp_path)
    write_config_field(off, field, "mismatched_value")
    with pytest.raises(SystemExit, match="Phase A acceptance FAILED"):
        compare(on, off)
    assert f'"{field}"' in capsys.readouterr().out


def test_paired_acceptance_detects_extra_or_missing_config_field(tmp_path):
    on, off = make_saved_pair(tmp_path)
    write_config_field(off, "unexpected_new_parameter", 5)
    with pytest.raises(SystemExit, match="Phase A acceptance FAILED"):
        compare(on, off)


def test_paired_acceptance_detects_missing_or_changed_count(tmp_path):
    on, off = make_saved_pair(tmp_path)
    path = off / "random_step" / "sample_0" / "evaluate" / "timing.json"
    obj = json.loads(path.read_text())
    obj.pop("network_backprops")
    path.write_text(json.dumps(obj))
    with pytest.raises(SystemExit, match="Phase A acceptance FAILED"):
        compare(on, off)


def test_paired_acceptance_detects_reconstruction_bit_change(tmp_path):
    on, off = make_saved_pair(tmp_path)
    path = off / "random_step" / "sample_0" / "evaluate" / "recons" / "recon_patch_0.npy"
    image = np.load(path)
    image[0, 0] += 1.0
    np.save(path, image)
    with pytest.raises(SystemExit, match="Phase A acceptance FAILED"):
        compare(on, off)
