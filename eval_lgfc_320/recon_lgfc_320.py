"""LGFC reconstruction with a 320 PaDIS target and unchanged 384 MRI measurements."""

import os
import sys
from typing import Optional

import numpy as np
import torch
import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
for _path in (_REPO_ROOT, _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_lgfc.frequency_correction import (
    LGFCConfig,
    apply_lgfc_frequency_correction,
    update_reference,
    validate_lgfc_config,
)
from eval_lgfc_320.denoise_padding_320 import (
    CANVAS_SIZE,
    MEAS_END,
    MEAS_START,
    MEASUREMENT_SIZE,
    TARGET_END,
    TARGET_SIZE,
    TARGET_START,
    denoised_from_patches_320,
    get_indices,
)


def _ve_base_schedule(net, num_sigmas, sigma_min, sigma_max, rho, device):
    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (
        sigma_max ** (1.0 / rho)
        + (idx / (num_sigmas - 1.0))
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t = net.round_sigma(t)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0)


def _should_correct(outer_step: int, config: LGFCConfig) -> bool:
    return (
        config.enabled
        and outer_step >= config.start_outer
        and (outer_step - config.start_outer) % config.correction_every == 0
    )


def _complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2).contiguous()


def _real2_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)


def dps2_lgfc_320(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: torch.Tensor,
    clean: Optional[torch.Tensor] = None,
    num_steps: int = 78,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    randn_like=torch.randn_like,
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    save_intermediate: bool = False,
    intermediate_every: int = 10,
    lgfc_config: Optional[LGFCConfig] = None,
    device: str = "cuda",
):
    """Run the isolated 320-ROI experiment.

    Geometry is fixed as follows:
      original MRI measurement FOV: 384x384 (unchanged)
      PaDIS target ROI:             central 320x320
      PaDIS inference canvas:       448x448
      target patch grid:            6x6 patches of 64x64

    The 384 measurement, mask, sensitivity maps and MRI forward operator are
    never cropped or regenerated.
    """
    del save_intermediate, intermediate_every

    if measurement is None:
        raise ValueError("320-ROI mode requires the original 384 MRI measurement")
    if pad != 64 or psize != 64:
        raise ValueError("320-ROI mode fixes pad=64 and psize=64")
    if num_steps != 78 or inner_loops != 10:
        raise ValueError("320-ROI pilot fixes 78 outer steps and 10 inner loops")
    if zeta != 3.0:
        raise ValueError("320-ROI pilot fixes zeta=3.0")
    if latents.shape[-2:] != (TARGET_SIZE, TARGET_SIZE):
        raise ValueError(f"latents must be {TARGET_SIZE}x{TARGET_SIZE}")
    if latents_pos.shape[-2:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(f"latents_pos must be {CANVAS_SIZE}x{CANVAS_SIZE}")

    config = validate_lgfc_config(lgfc_config or LGFCConfig(enabled=False))
    net.eval()

    patches = TARGET_SIZE // psize + 1
    if patches != 6:
        raise RuntimeError("320 target with psize64 must produce a 6x6 patch grid")
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    if x_init.ndim != 4 or x_init.shape[1] != 1 or not torch.is_complex(x_init):
        raise ValueError("inverseop.adjoint(measurement) must return complex [B,1,384,384]")
    if x_init.shape[-2:] != (MEASUREMENT_SIZE, MEASUREMENT_SIZE):
        raise ValueError(
            f"measurement adjoint must be {MEASUREMENT_SIZE}x{MEASUREMENT_SIZE}, "
            f"got {tuple(x_init.shape[-2:])}"
        )

    # 384 real MRI FOV + 32 zero pixels on each side = 448 PaDIS canvas.
    x = torch.nn.functional.pad(x_init, (32, 32, 32, 32), "constant", 0)
    t_steps = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
        desc="LGFC-320",
    ):
        del t_next
        outer_step = i + 1
        reference = None
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for _ in range(inner_loops):
            indices = get_indices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)
            x_noisy = x + t_cur * randn_like(x)

            x_real_noisy = _complex_to_2ch(x_noisy)
            x_real_state = _complex_to_2ch(x)
            d_real = denoised_from_patches_320(
                net=net,
                x_noisy_real=x_real_noisy,
                x_state_real=x_real_state,
                t_hat=t_cur,
                latents_pos=latents_pos,
                class_labels=None,
                indices=indices,
            )
            d_cplx = _real2_to_complex(d_real)

            # Full 384 x0 estimate used by both the original MRI likelihood
            # and LGFC.  Only its central 320 region came from PaDIS.
            d_measurement = d_cplx[
                :, :, MEAS_START:MEAS_END, MEAS_START:MEAS_END
            ]
            reference = update_reference(
                reference,
                d_measurement.detach(),
                mode=config.reference_mode,
                ema_beta=config.ema_beta,
            )

            score = (d_cplx - x_noisy) / (t_cur ** 2)

            x0hat_384 = d_real[
                :, :, MEAS_START:MEAS_END, MEAS_START:MEAS_END
            ]
            ax = inverseop.forward(x0hat_384)
            residual = measurement - ax
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)
            likelihood_grad = torch.autograd.grad(sse, x)[0]

            x = x - (
                zeta / torch.sqrt(sse_ind)[:, None, None, None]
            ) * likelihood_grad

            if i < num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score

        if _should_correct(outer_step, config):
            if reference is None:
                raise RuntimeError("LGFC reference was not constructed")
            x_measurement = x[
                :, :, MEAS_START:MEAS_END, MEAS_START:MEAS_END
            ].detach()
            with torch.no_grad():
                x_corrected, _ = apply_lgfc_frequency_correction(
                    x_fov=x_measurement,
                    reference=reference,
                    maps=inverseop.maps,
                    mask=inverseop.mask,
                    measurement=measurement,
                    unsampled_weight=config.unsampled_weight,
                    max_relative_update=config.max_relative_update,
                    dc_growth_limit=config.dc_growth_limit,
                    backtrack_factor=config.backtrack_factor,
                    max_backtracks=config.max_backtracks,
                    eps=config.eps,
                )
                x_next = x.detach().clone()
                x_next[
                    :, :, MEAS_START:MEAS_END, MEAS_START:MEAS_END
                ] = x_corrected
            x = x_next

    recon = x[
        :, :, TARGET_START:TARGET_END, TARGET_START:TARGET_END
    ].detach().squeeze(1)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = 0.0
    noisynrmse = denoisednrmse = 0.0
    if clean is not None:
        if clean.shape[-2:] != (MEASUREMENT_SIZE, MEASUREMENT_SIZE):
            raise ValueError("clean GT must remain the original 384x384 image")
        gt_target = clean[:, :, 32:352, 32:352].squeeze(1)
        init_target = x_init[:, :, 32:352, 32:352].squeeze(1)
        make_figures = __import__("utils").makeFigures
        (
            noisypsnr,
            denoisedpsnr,
            noisyssim,
            denoisedssim,
            noisynrmse,
            denoisednrmse,
        ) = make_figures(
            noisy2=init_target,
            denoised2=recon,
            orig2=gt_target,
            i=num_steps,
            out_dir=save_dir,
            tag=tag,
            plot=False,
        )

    return (
        recon,
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
