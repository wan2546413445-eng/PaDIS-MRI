import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os
import sys
import csv
import time
import atexit
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart
configure_bart()

from bart import bart

from eval.inverse_operators import *
from denoise_padding_fast import denoisedFromPatches, getIndices, getIndicesMultiScale
from eval.utils import fftmod, makeFigures
# ============================================================
# DPS2 posterior-update profiling utilities
# ------------------------------------------------------------
# 与 denoise_padding_fast.py 共用同一个环境变量：
#
#   export PADIS_FASTPATCH_PROFILE=1
#
# 这样一次运行会同时输出：
#   1) denoisedFromPatches 内部耗时
#   2) dps2 完整 posterior update 外层耗时
#
# 仅在 profiling 打开时生效，不改变重建计算逻辑。
# ============================================================

DPS2_PROFILE = os.environ.get("PADIS_FASTPATCH_PROFILE", "0") == "1"

_DPS2_PROFILE_STATS = defaultdict(float)
_DPS2_PROFILE_COUNTS = defaultdict(int)


def _dps2_cuda_sync_if_needed():
    """
    CUDA 是异步执行的。
    profiling 打开时同步，保证计时真实。
    """
    if DPS2_PROFILE and torch.cuda.is_available():
        torch.cuda.synchronize()


def _dps2_profile_start():
    """
    返回某段计时起点。
    """
    if not DPS2_PROFILE:
        return None
    _dps2_cuda_sync_if_needed()
    return time.perf_counter()


def _dps2_profile_end(name: str, start_time):
    """
    结束某段计时并累计。
    """
    if not DPS2_PROFILE or start_time is None:
        return
    _dps2_cuda_sync_if_needed()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    _DPS2_PROFILE_STATS[name] += float(elapsed_ms)
    _DPS2_PROFILE_COUNTS[name] += 1


def print_dps2_profile():
    """
    程序退出时打印 dps2 完整后验更新流程的累计耗时。
    """
    if not DPS2_PROFILE:
        return

    print("\n" + "=" * 84)
    print("[DPS2 Posterior Update Profiling Summary]")
    print("=" * 84)

    ordered_keys = [
        "dps2_init_ms",
        "prepare_indices_and_grad_ms",
        "noise_and_real_convert_ms",
        "patch_denoise_call_ms",
        "score_and_crop_ms",
        "mri_forward_residual_sse_ms",
        "likelihood_backward_ms",
        "measurement_update_ms",
        "diffusion_update_ms",
        "posterior_update_total_ms",
        "dps2_total_ms",
    ]

    total_updates = _DPS2_PROFILE_COUNTS.get(
        "posterior_update_total_ms", 0
    )

    print(f"posterior updates: {total_updates}")

    for key in ordered_keys:
        count = _DPS2_PROFILE_COUNTS.get(key, 0)
        total = _DPS2_PROFILE_STATS.get(key, 0.0)
        avg = total / count if count > 0 else 0.0
        print(
            f"{key:38s} | "
            f"total = {total:11.3f} ms | "
            f"avg = {avg:9.3f} ms | "
            f"n = {count}"
        )

    print("=" * 84 + "\n")


if DPS2_PROFILE:
    atexit.register(print_dps2_profile)

# ============================================================
# DPS2 dtype-flow debug utilities
# ------------------------------------------------------------
# 仅用于检查 PaDIS-MRI 后验链路中的 dtype 是否从
# float32 / complex64 被 D_real=float64 推升到 complex128。
#
# 开启方式：
#   export PADIS_FASTPATCH_DTYPE_DEBUG=1
#
# 默认关闭，不影响正常重建。
# 只在第 1 次 posterior update 打印一次，避免刷屏。
# ============================================================

DPS2_DTYPE_DEBUG = os.environ.get("PADIS_FASTPATCH_DTYPE_DEBUG", "0") == "1"
_DPS2_DTYPE_DEBUG_PRINTED = False

def _maybe_print_dps2_input_dtype_debug(
    *,
    measurement,
    inverseop,
    x_init,
    x_after_pad,
    t_steps,
):
    """
    只打印一次 dps2 入口张量 dtype，
    用于定位 complex128 / float64 从哪里进入。
    """
    if not DPS2_DTYPE_DEBUG:
        return

    print("\n" + "=" * 104)
    print("[DPS2 Input DType Debug | before posterior loop]")
    print("=" * 104)

    tensors_to_print = [
        ("measurement", measurement),
        ("inverseop.maps", inverseop.maps),
        ("inverseop.mask", inverseop.mask),
        ("x_init_from_adjoint", x_init),
        ("x_after_pad", x_after_pad),
        ("t_steps", t_steps),
    ]

    for name, tensor in tensors_to_print:
        print(_tensor_dtype_line(name, tensor))

    print("=" * 104 + "\n")

def _tensor_dtype_line(name: str, tensor):
    """
    生成单个 tensor 的 dtype / shape / device / requires_grad 信息。
    只读取元信息，不改 tensor，不做 detach，不影响计算图。
    """
    if tensor is None:
        return f"{name:34s}: None"

    return (
        f"{name:34s}: "
        f"dtype={str(tensor.dtype):18s} | "
        f"shape={str(tuple(tensor.shape)):24s} | "
        f"device={str(tensor.device):8s} | "
        f"requires_grad={tensor.requires_grad}"
    )


def _maybe_print_dps2_dtype_debug(
    *,
    x_before_update,
    x_noisy,
    x_real_noisy,
    D_real,
    D_cplx,
    score,
    cropped_x0hat,
    Ax,
    residual,
    sse_ind,
    sse,
    likelihood_grad,
    x_after_measurement_update,
    x_after_diffusion_update,
):
    """
    只在首次 posterior update 打印一次 dtype 流。
    """
    global _DPS2_DTYPE_DEBUG_PRINTED

    if not DPS2_DTYPE_DEBUG or _DPS2_DTYPE_DEBUG_PRINTED:
        return

    print("\n" + "=" * 104)
    print("[DPS2 DType Flow Debug | first posterior update only]")
    print("=" * 104)

    tensors_to_print = [
        ("x_before_update", x_before_update),
        ("x_noisy", x_noisy),
        ("x_real_noisy", x_real_noisy),
        ("D_real", D_real),
        ("D_cplx", D_cplx),
        ("score", score),
        ("cropped_x0hat", cropped_x0hat),
        ("Ax", Ax),
        ("residual", residual),
        ("sse_ind", sse_ind),
        ("sse", sse),
        ("likelihood_grad", likelihood_grad),
        ("x_after_measurement_update", x_after_measurement_update),
        ("x_after_diffusion_update", x_after_diffusion_update),
    ]

    for name, tensor in tensors_to_print:
        print(_tensor_dtype_line(name, tensor))

    print("=" * 104 + "\n")

    _DPS2_DTYPE_DEBUG_PRINTED = True

def _ve_base_schedule(
    net,
    num_sigmas: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a base VE schedule of length `num_sigmas` (ex. 104), and appends a terminal zero (so length becomes num_sigmas+1).
    """
    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (sigma_max ** (1.0 / rho) + (idx / (num_sigmas - 1.0)) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    t = net.round_sigma(t) 
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0)


def _normalize_patch_probs(patch_sizes, patch_probs):
    patch_sizes = list(patch_sizes)
    patch_probs = list(patch_probs)
    if len(patch_sizes) == 0:
        raise ValueError("multiscale_patch_sizes must be non-empty.")
    if len(patch_sizes) != len(patch_probs):
        raise ValueError("multiscale_patch_sizes and multiscale_patch_probs must have the same length.")
    if any(float(p) < 0 for p in patch_probs):
        raise ValueError("multiscale_patch_probs must be non-negative.")
    probs_sum = float(sum(float(p) for p in patch_probs))
    if probs_sum <= 0:
        raise ValueError("multiscale_patch_probs must sum to a value > 0.")
    return [float(p) / probs_sum for p in patch_probs]


def _select_multiscale_patch_size(
    patch_schedule,
    patch_sizes,
    patch_probs,
    outer_step,
    num_steps,
):
    if patch_schedule == "fixed":
        raise RuntimeError("_select_multiscale_patch_size should not be called when patch_schedule='fixed'.")

    if patch_schedule == "train_random":
        return int(np.random.choice(np.array(patch_sizes), p=np.array(patch_probs, dtype=np.float64)))

    if patch_schedule == "coarse_to_fine":
        sizes_desc = sorted([int(s) for s in patch_sizes], reverse=True)
        if len(sizes_desc) == 1:
            return sizes_desc[0]

        progress = float(outer_step) / float(max(1, num_steps))
        if len(sizes_desc) == 2:
            return sizes_desc[0] if progress < 0.5 else sizes_desc[-1]

        mid_idx = len(sizes_desc) // 2
        if progress < 0.4:
            return sizes_desc[0]
        if progress < 0.8:
            return sizes_desc[mid_idx]
        return sizes_desc[-1]

    raise ValueError(f"Unknown patch_schedule: {patch_schedule}")

def dps2(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: Optional[torch.Tensor],
    num_steps: int = 104,  # num noising steps
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    randn_like=torch.randn_like,
    verbose: bool = False,
    clean: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    save_intermediate: bool = False,
    intermediate_every: int = 10,
    patch_schedule: str = "fixed",
    multiscale_patch_sizes=None,
    multiscale_patch_probs=None,
    sigma_switch: float = 0.1,
    resume_enable: bool = False,
    resume_step: int = 52,
    resume_noise_std: float = 0.05,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    PaDIS (patch DPS) MRI reconstruction with 10 inner sub-steps per sigma
    (total ~1040 updates for num_steps=104).

    Optional diagnostic mode:
    - save_intermediate=True:
        Save author-style intermediate visualizations and metrics
        after selected outer diffusion steps.
    - intermediate_every:
        Save every N outer steps, while always saving step 1 and final step.
    """

    _t_dps2_total = _dps2_profile_start()
    _t_dps2_init = _dps2_profile_start()

    net.eval()
    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    if patch_schedule == "train_random":
        if not multiscale_patch_sizes:
            multiscale_patch_sizes = [16, 32, 64]
        if not multiscale_patch_probs:
            multiscale_patch_probs = [0.2, 0.3, 0.5]

        normalized_patch_probs = _normalize_patch_probs(
            multiscale_patch_sizes,
            multiscale_patch_probs,
        )

    elif patch_schedule == "coarse_to_fine":
        if not multiscale_patch_sizes:
            multiscale_patch_sizes = [16, 32, 64]

        if len(multiscale_patch_sizes) == 0:
            raise ValueError("multiscale_patch_sizes must be non-empty for coarse_to_fine.")

        normalized_patch_probs = None

    elif patch_schedule == "sigma_c2f":
        if not multiscale_patch_sizes:
            multiscale_patch_sizes = [32, 64]

        if len(multiscale_patch_sizes) != 2:
            raise ValueError(
                "sigma_c2f requires exactly two patch sizes, e.g. [32, 64]."
            )

        normalized_patch_probs = None
        sizes_desc = sorted([int(s) for s in multiscale_patch_sizes], reverse=True)
        sigma_large_psize = sizes_desc[0]
        sigma_small_psize = sizes_desc[-1]

    elif patch_schedule == "fixed":
        normalized_patch_probs = None

    else:
        raise ValueError(f"Unknown patch_schedule: {patch_schedule}")
    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)  # complex

    t_steps = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    resume_done = False
    if resume_enable:
        if int(resume_step) < 1 or int(resume_step) >= int(num_steps):
            raise ValueError(
                f"resume_step must be in [1, num_steps-1], got {resume_step}."
            )
        if float(resume_noise_std) < 0:
            raise ValueError(
                f"resume_noise_std must be non-negative, got {resume_noise_std}."
            )
        print(
            f"[Noise&Resume] pseudo enabled: "
            f"resume_step={resume_step}, resume_noise_std={resume_noise_std}"
        )

    _maybe_print_dps2_input_dtype_debug(
        measurement=measurement,
        inverseop=inverseop,
        x_init=x_init,
        x_after_pad=x,
        t_steps=t_steps,
    )

    _dps2_profile_end("dps2_init_ms", _t_dps2_init)
    if patch_schedule == "fixed":
        print(f"[Patch Sampling] fixed psize={psize}")
    elif patch_schedule == "train_random":
        print(
            f"[Patch Sampling] schedule=train_random, "
            f"sizes={multiscale_patch_sizes}, "
            f"probs={normalized_patch_probs}"
        )
    elif patch_schedule == "coarse_to_fine":
        print(
            f"[Patch Sampling] schedule=coarse_to_fine, "
            f"sizes={multiscale_patch_sizes}"
        )
    elif patch_schedule == "sigma_c2f":
        print(
            f"[Patch Sampling] schedule=sigma_c2f, "
            f"sizes={multiscale_patch_sizes}, "
            f"sigma_switch={sigma_switch}"
        )

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0

    # ------------------------------------------------------------------
    # Intermediate visualization / author-style metric diagnostics
    # ------------------------------------------------------------------
    intermediate_rows = []
    intermediate_dir = None
    safe_tag = tag if tag is not None else "sample"
    intermediate_every = max(1, int(intermediate_every))

    if save_intermediate and save_dir is not None:
        intermediate_dir = os.path.join(save_dir, "intermediate_vis", safe_tag)
        os.makedirs(intermediate_dir, exist_ok=True)
        print(f"[intermediate] Enabled: {intermediate_dir} | every={intermediate_every}")

    # ------------------------------------------------------------------
    # Main posterior sampling loop
    # ------------------------------------------------------------------
    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for j in range(inner_loops):
            _t_update_total = _dps2_profile_start()

            # ----------------------------------------------------------
            # 1. Random patch partition + prepare current x for autograd
            # ----------------------------------------------------------
            _t_prepare = _dps2_profile_start()

            if patch_schedule == "fixed":
                indices = getIndices(spaced, patches, pad, psize)
                crop_pad_arg = None
            elif patch_schedule == "sigma_c2f":
                cur_psize = sigma_large_psize if float(t_cur.item()) > float(sigma_switch) else sigma_small_psize
                indices = getIndicesMultiScale(
                    image_size=w,
                    pad=pad,
                    psize=cur_psize,
                    freezeindex=False,
                )
                crop_pad_arg = pad

            else:
                cur_psize = _select_multiscale_patch_size(
                    patch_schedule=patch_schedule,
                    patch_sizes=multiscale_patch_sizes,
                    patch_probs=normalized_patch_probs,
                    outer_step=i,
                    num_steps=num_steps,
                )
                indices = getIndicesMultiScale(
                    image_size=w,
                    pad=pad,
                    psize=cur_psize,
                    freezeindex=False,
                )
                crop_pad_arg = pad
            x = x.detach().requires_grad_(True)
            x_before_update = x

            _dps2_profile_end("prepare_indices_and_grad_ms", _t_prepare)

            # ----------------------------------------------------------
            # 2. VE noise injection + complex -> 2-channel real
            # ----------------------------------------------------------
            _t_noise = _dps2_profile_start()

            x_noisy = x + (t_cur * randn_like(x))
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)  # (B,2,H_pad,W_pad)

            _dps2_profile_end("noise_and_real_convert_ms", _t_noise)

            # ----------------------------------------------------------
            # 3. Patch denoise call
            #    其内部更细粒度耗时已由 denoise_padding_fast.py 统计
            # ----------------------------------------------------------
            _t_patch = _dps2_profile_start()

            D_real = denoisedFromPatches(
                net,
                x_real_noisy,
                t_cur,
                latents_pos,
                None,
                indices,
                t_goal=0,
                wrong=False,
                crop_pad=crop_pad_arg,
            )

            _dps2_profile_end("patch_denoise_call_ms", _t_patch)

            # ----------------------------------------------------------
            # 4. Score function calculation + crop
            # ----------------------------------------------------------
            _t_score = _dps2_profile_start()

            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
            score = (D_cplx - x_noisy) / (t_cur ** 2)

            cropped_x0hat = (
                D_real
                if pad == 0
                else D_real[:, :, pad:pad+w, pad:pad+w]
            )

            _dps2_profile_end("score_and_crop_ms", _t_score)

            # ----------------------------------------------------------
            # 5. MRI forward + residual + SSE
            # ----------------------------------------------------------
            _t_mri_forward = _dps2_profile_start()

            Ax = inverseop.forward(cropped_x0hat)
            residual = measurement - Ax
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)

            _dps2_profile_end("mri_forward_residual_sse_ms", _t_mri_forward)

            # ----------------------------------------------------------
            # 6. Likelihood gradient backward
            # ----------------------------------------------------------
            _t_backward = _dps2_profile_start()

            likelihood_grad = torch.autograd.grad(
                outputs=sse,
                inputs=x
            )[0]

            _dps2_profile_end("likelihood_backward_ms", _t_backward)

            # ----------------------------------------------------------
            # 7. Measurement-consistency update
            # ----------------------------------------------------------
            _t_measurement_update = _dps2_profile_start()

            x_after_measurement_update = x - (
                    zeta / torch.sqrt(sse_ind)[:, None, None, None]
            ) * likelihood_grad

            x = x_after_measurement_update

            _dps2_profile_end("measurement_update_ms", _t_measurement_update)

            # ----------------------------------------------------------
            # 8. Diffusion update
            # ----------------------------------------------------------
            _t_diffusion_update = _dps2_profile_start()

            if i < num_steps - 1:
                x_after_diffusion_update = (
                        x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
                )
            else:
                x_after_diffusion_update = x + (alpha / 2) * score

            x = x_after_diffusion_update

            _maybe_print_dps2_dtype_debug(
                x_before_update=x_before_update,
                x_noisy=x_noisy,
                x_real_noisy=x_real_noisy,
                D_real=D_real,
                D_cplx=D_cplx,
                score=score,
                cropped_x0hat=cropped_x0hat,
                Ax=Ax,
                residual=residual,
                sse_ind=sse_ind,
                sse=sse,
                likelihood_grad=likelihood_grad,
                x_after_measurement_update=x_after_measurement_update,
                x_after_diffusion_update=x_after_diffusion_update,
            )

            _dps2_profile_end("diffusion_update_ms", _t_diffusion_update)

            # ----------------------------------------------------------
            # 9. One complete posterior update
            # ----------------------------------------------------------
            _dps2_profile_end("posterior_update_total_ms", _t_update_total)

            if verbose:
                with torch.no_grad():
                    print(f"step {i+1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

        # ------------------------------------------------------------------
        # Pseudo Noise & Resume
        # ------------------------------------------------------------------
        # Inspired by multiscale patch diffusion N&R:
        # after a chosen outer step, perturb the current posterior state
        # with a small amount of noise, then continue the original schedule.
        #
        # This is intentionally conservative:
        # - does not change patch size
        # - does not change zeta
        # - does not rebuild t_steps
        # - does not touch denoise_padding_fast.py
        # ------------------------------------------------------------------
        if resume_enable and (not resume_done) and (i + 1 == int(resume_step)):
            with torch.no_grad():
                noise_std = torch.as_tensor(
                    float(resume_noise_std),
                    dtype=x.real.dtype,
                    device=x.device,
                )
                x = x.detach() + noise_std * randn_like(x.detach())
                resume_done = True

            print(
                f"[Noise&Resume] injected noise after outer_step={i+1}, "
                f"noise_std={resume_noise_std}"
            )

        # ------------------------------------------------------------------
        # Save intermediate visualization after the full inner loop of this
        # outer diffusion step.
        # ------------------------------------------------------------------
        should_save = (
            save_intermediate
            and intermediate_dir is not None
            and clean is not None
            and (
                i == 0
                or ((i + 1) % intermediate_every == 0)
                or (i == num_steps - 1)
            )
        )

        if should_save:
            current_recon = x[:, :, pad:pad+w, pad:pad+w].detach().squeeze(1)

            (
                noisy_psnr,
                recon_psnr,
                noisy_ssim,
                recon_ssim,
                noisy_nrmse,
                recon_nrmse,
            ) = makeFigures(
                noisy2=x_init.squeeze(1),
                denoised2=current_recon,
                orig2=clean.squeeze(1),
                i=i + 1,
                out_dir=intermediate_dir,
                tag=f"{safe_tag}_step{i+1:03d}",
                plot=True,
            )

            intermediate_rows.append({
                "step": int(i + 1),
                "num_steps": int(num_steps),
                "noisy_psnr": float(noisy_psnr),
                "recon_psnr": float(recon_psnr),
                "noisy_ssim": float(noisy_ssim),
                "recon_ssim": float(recon_ssim),
                "noisy_nrmse": float(noisy_nrmse),
                "recon_nrmse": float(recon_nrmse),
            })

    # ------------------------------------------------------------------
    # Save intermediate metric CSV
    # ------------------------------------------------------------------
    if intermediate_rows and intermediate_dir is not None:
        csv_path = os.path.join(intermediate_dir, "intermediate_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(intermediate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(intermediate_rows)

        print(f"[intermediate] Saved diagnostics to {intermediate_dir}")

    _dps2_profile_end("dps2_total_ms", _t_dps2_total)

    return (
        x[:, :, pad:pad + w, pad:pad + w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
@torch.no_grad()
def dps_uncond(
    net,
    batch_size=1,
    resolution=384,
    psize=96,
    pad=96,
    num_steps=50,
    sigma_min=0.003,
    sigma_max=10.0,
    rho=7,
    device='cuda',
    randn_like=torch.randn_like,
):
    net.eval()

    #--- init ---
    shape = (batch_size, 1, resolution, resolution)
    x = sigma_max * randn_like(torch.zeros(shape, dtype=torch.complex64, device=device))
    if pad>0:
        x = F.pad(x, (pad, pad, pad, pad), 'constant', 0)

    #--- schedule ---大到小noise非线性递减
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max**(1/rho) +
               idx/(num_steps-1)*(sigma_min**(1/rho)-sigma_max**(1/rho))
              )**rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros(1, dtype=torch.float64, device=device)])

    #--- positional grid (if your model needs it) ---
    R = resolution + 2*pad
    x_lin = torch.linspace(-1,1,R,device=device)
    y_lin = torch.linspace(-1,1,R,device=device)
    x_pos = x_lin.view(1,-1).repeat(R,1)
    y_pos = y_lin.view(-1,1).repeat(1,R)
    latents_pos = torch.stack([x_pos,y_pos],dim=0).unsqueeze(0)  # [1,2,R,R]

    patches = (resolution//psize)+1
    spaced = np.linspace(0,(patches-1)*psize,patches,dtype=int)

    #--- VE-DPS loop (no measurement term) ---
    for i,(t_cur,t_next) in enumerate(zip(t_steps[:-1],t_steps[1:])):
        print(i)
        alpha = 0.5 * t_cur**2
        for _ in range(4):     # same inner loops as dps2
            x = x.detach().requires_grad_(True)

            # 1) VE noise injection
            eps = randn_like(x)
            x_noisy = x + t_cur * eps

            # 2) denoise patches
            xr = torch.view_as_real(x_noisy.squeeze(1)).permute(0,3,1,2)
            D_real = denoisedFromPatches(net, xr, t_cur, latents_pos, None,
                                         getIndices(spaced,patches,pad,psize),
                                         t_goal=0, wrong=False)
            D_cplx = torch.complex(D_real[:,0], D_real[:,1]).unsqueeze(1)

            # 3) score
            score = (D_cplx - x) / (t_cur**2)

            # 4) diffusion update
            if i < num_steps-1:
                x = x + (alpha/2)*score + torch.sqrt(alpha)*randn_like(x)
            else:
                x = x + (alpha/2)*score

    if pad>0:
        x = x[:,:,pad:-pad,pad:-pad]
    return x.detach()


def dps_edm(
        net,
        measurement: torch.Tensor,
        clean: torch.Tensor,
        inverseop,
        num_steps: int = 104,
        repeats_per_sigma: int = 10,
        sigma_min: float = 0.003,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        zeta: float = 3.0,
        pad: int = 0,  # use SAME pad as PaDIS to align FOV
        device: str = 'cuda',
        randn_like=torch.randn_like,
        verbose: bool = False,
        save_dir: Optional[str] = None,
        tag: Optional[str] = None,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    EDM-style DPS reconstruction (whole-image denoising) with measurement consistency. Matches PaDIS-MRI noising schedule.
    """
    net.eval()

    x_init = inverseop.adjoint(measurement).to(device)
    x = x_init.clone()

    t_base = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)
    t_rep = torch.repeat_interleave(t_base[:-1], repeats_per_sigma)
    t_steps = torch.cat([t_rep, t_base[-1:]], dim=0)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0
    total_iters = t_steps.numel() - 1

    # 3) main Euler‐Maruyama loop
    for i in tqdm.tqdm(range(total_iters)):
        t_cur = t_steps[i].float()
        alpha = 0.5 * (t_cur ** 2)

        x = x.detach().requires_grad_(True)

        # VE noise injection
        x_noisy = x + t_cur * randn_like(x)

        # model call in 2ch real
        x_real = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
        den_real = net(x_real, t_cur)  # [B,2,H,W]

        # score calc
        den_cplx = torch.complex(den_real[:, 0], den_real[:, 1]).unsqueeze(1)
        score = (den_cplx - x_noisy) / (t_cur ** 2)

        # measurement‐consistency
        Ax = inverseop.forward(den_real)
        residual = measurement - Ax
        sse = torch.sum(torch.abs(residual).view(residual.shape[0], -1) ** 2, dim=1)  # [B]
        # gradient of SSE w.r.t. x_cplx
        grad_l = torch.autograd.grad(outputs=sse, inputs=x)[0]

        # measurement‐consistency update
        x_mid = x + (alpha / 2) * score - (zeta / torch.sqrt(sse)[:, None, None, None]) * grad_l

        # diffusion update
        if i < total_iters - 1:
            x = x_mid + torch.sqrt(alpha) * randn_like(x_mid)
        else:
            x = x_mid

        if verbose:
            with torch.no_grad():
                print(f"step {i + 1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

        # if save_dir is not None and i == num_steps-1:
        #     noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse = makeFigures(
        #                                                                                         x_init.squeeze(1),
        #                                                                                         x.squeeze(1).detach(),
        #                                                                                         clean.squeeze(1),
        #                                                                                         i,
        #                                                                                         save_dir,
        #                                                                                         tag)

    return x.detach(), noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse


@torch.no_grad()
def dps_uncond_edm(
        net,
        batch_size=1,
        resolution=384,
        num_steps=96,
        sigma_min=0.003,
        sigma_max=10.0,
        rho=7,
        device='cuda',
        randn_like=torch.randn_like,
):
    """
    Unconditional generation using a plain EDM model trained on full images (no patching).
    """

    # 1) initialize noise
    shape = (batch_size, net.img_channels, resolution, resolution)
    x = sigma_max * randn_like(torch.empty(shape, device=device))

    # 2) build noise schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
                      sigma_max ** (1 / rho)
                      + (step_indices / (num_steps - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
              ) ** rho
    # append final zero
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # 3) Euler–Maruyama loop
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        sigma = t_cur.float()
        dt = (t_next - t_cur).float()

        # score estimate via denoising network
        denoised = net(x, sigma)  # [B,C,H,W]
        score = (x - denoised) / sigma  # [B,C,H,W]

        # Euler update
        x = x + score * dt

    x = x.squeeze(0)
    real = x[0]
    imag = x[1]
    complex = torch.complex(real, imag)
    return complex
