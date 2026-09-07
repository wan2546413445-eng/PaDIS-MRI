"""Minimal CPU smoke test for the MCPP mechanism."""

import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from prior_calibration import MCPPHeadAdapter
from recon_mcpp import dps2_mcpp


class SongUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 一个真实的 frozen backbone 参数，避免“冻结主干”测试变成空检查。
        self.stem = torch.nn.Conv2d(2, 2, 1)
        self.dec = torch.nn.ModuleDict({
            "8x8_aux_norm": torch.nn.GroupNorm(1, 2),
            "8x8_aux_conv": torch.nn.Conv2d(2, 2, 3, padding=1),
        })

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        del noise_labels, class_labels, augment_labels
        x = self.stem(x)
        tmp = self.dec["8x8_aux_norm"](x)
        return self.dec["8x8_aux_conv"](torch.nn.functional.silu(tmp))


class Patch_EDMPrecond(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SongUNet()

    def forward(self, x, sigma, x_pos=None, class_labels=None):
        del x_pos
        return self.model(x, sigma.reshape(-1), class_labels)

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)


class TinyMRI:
    @staticmethod
    def adjoint(measurement):
        return measurement.clone()

    @staticmethod
    def forward(x_real):
        return torch.complex(x_real[:, 0], x_real[:, 1]).unsqueeze(1)


def main() -> None:
    torch.manual_seed(7)
    net = Patch_EDMPrecond().eval()
    adapter = MCPPHeadAdapter(net, print_info=True)

    selected = set(adapter.names)
    for name, parameter in net.named_parameters():
        if name in selected:
            assert parameter.requires_grad, f"adaptable parameter is frozen: {name}"
        else:
            assert not parameter.requires_grad, f"backbone parameter is unexpectedly trainable: {name}"

    # 同一次 measurement loss 图同时产生 grad_x 和 grad_phi。
    x_complex = torch.randn(1, 1, 8, 8, dtype=torch.complex64).requires_grad_(True)
    x_real = torch.view_as_real(x_complex.squeeze(1)).permute(0, 3, 1, 2)
    sigma = torch.tensor(0.5)
    fixed_input = x_real.detach().clone()
    before = net(fixed_input, sigma).detach().clone()
    denoised = net(x_real, sigma)
    denoised_complex = torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)
    measurement = torch.full_like(denoised_complex, 0.25 + 0.1j)
    sse = torch.sum(torch.abs(denoised_complex - measurement) ** 2)
    grads = torch.autograd.grad(sse, [x_complex] + adapter.parameters, allow_unused=False)
    assert grads[0] is not None and all(gradient is not None for gradient in grads[1:])
    assert torch.isfinite(grads[0]).all()
    assert all(torch.isfinite(gradient).all() for gradient in grads[1:])

    normalized = [
        gradient / (torch.sum(torch.abs(measurement) ** 2) + 1e-12)
        for gradient in grads[1:]
    ]
    adapter.step(normalized, 1e-3)
    after = net(fixed_input, sigma).detach()
    output_change = torch.linalg.vector_norm(after - before).item()
    assert output_change > 0, "Measurement-driven phi update did not change the future prior"

    adapter.reset()
    assert adapter.is_exactly_reset(), "phi reset did not exactly restore phi0"
    restored = net(fixed_input, sigma).detach()
    assert torch.equal(before, restored), "Reset output differs from pretrained output"

    call_counter = {"count": 0}

    def denoise_once(net_arg, x_arg, sigma_arg, pos_arg, labels_arg, indices_arg,
                     t_goal=0, wrong=False):
        del labels_arg, indices_arg, t_goal, wrong
        call_counter["count"] += 1
        return net_arg(x_arg, sigma_arg, pos_arg, None)

    def fixed_indices(spaced, patches, pad, psize):
        del spaced, patches, pad
        return [[0, psize, 0, psize]]

    adapter.reset()
    measurement_small = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    recon, diagnostics, metadata = dps2_mcpp(
        net=net,
        latents=torch.zeros_like(measurement_small),
        latents_pos=torch.zeros(1, 2, 8, 8),
        inverseop=TinyMRI(),
        measurement=measurement_small,
        num_steps=2,
        inner_loops=2,
        sigma_min=0.01,
        sigma_max=0.5,
        zeta=0.1,
        pad=0,
        psize=8,
        mcpp_lr=1e-4,
        randn_like=lambda value: torch.zeros_like(value),
        device="cpu",
        adapter=adapter,
        collect_diagnostics=True,
        _denoise_fn=denoise_once,
        _indices_fn=fixed_indices,
    )

    expected_calls = 4
    assert metadata["updates"] == expected_calls
    assert metadata["denoiser_calls"] == expected_calls
    assert call_counter["count"] == expected_calls
    assert len(diagnostics) == expected_calls
    assert torch.isfinite(recon).all()
    adapter.reset()
    assert adapter.is_exactly_reset()

    print("[PASS] final aux_norm + aux_conv discovery")
    print("[PASS] frozen backbone and trainable output head")
    print("[PASS] same-graph grad_x and grad_phi")
    print(f"[PASS] future denoiser output changed: ||D_after-D_before||={output_change:.6e}")
    print("[PASS] exact phi reset")
    print("[PASS] no NaN/Inf in a small reconstruction run")
    print(f"[PASS] one denoiser forward per inner loop: {expected_calls}/{expected_calls}")


if __name__ == "__main__":
    main()
