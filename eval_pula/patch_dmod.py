"""Patch adaptation of D-MOD (Ahmed & Jacob, arXiv:2610.40110v2, Alg. 1, Eq. 10).
The MRI linear system and noise convention come from the existing PaDIS pULA code.
No posterior Langevin step, DPS, DDIM, or final extra denoising is performed.
"""
import math
import random


def make_sigmas(T, sigma_max, sigma_min, rho):
    if T < 2 or not all(math.isfinite(v) for v in (sigma_max, sigma_min, rho)) or not (sigma_max > sigma_min > 0 and rho > 0):
        raise ValueError('Require T>=2, sigma_max>sigma_min>0, rho>0')
    lo, hi = sigma_min**(1/rho), sigma_max**(1/rho)
    result = [(hi + i/(T-1)*(lo-hi))**rho for i in range(T)]
    result[0], result[-1] = sigma_max, sigma_min
    return result


def make_grid_plan(T, K, S, pad, seed):
    """Uniform shifts; sample S distinct partitions within each MM iteration.
    Independent draws across groups, separate grid RNG from Gaussian noise RNG.
    Each prediction still has a uniform marginal over the PaDIS partition space.
    """
    if min(T,K,S,pad) < 1 or S > pad*pad:
        raise ValueError('Invalid T/K/S/pad or too many distinct shifts')
    rng = random.Random(seed)
    return [[[divmod(q,pad) for q in rng.sample(range(pad*pad), S)]
             for _ in range(K)] for _ in range(T)]


def reconstruct_patch_dmod(system, oracle, sigmas, shifts, generator, *, report=None):
    """Return final complex FOV and per-MM diagnostics. Reuse existing PULASystem.solve.

    system must use padis_real whitening (unit variance in each real component).
    Gaussian perturbations are applied to the entire zero-padded image.
    Every one of the S predictions uses the same frozen current MM iterate.
    """
    import torch
    import torch.nn.functional as F
    from eval_pula.recon_pula import real_unit_noise
    if system.convention != 'padis_real':
        raise ValueError('Patch-DMOD uses the PaDIS per-real-channel EDM convention')
    if len(sigmas) != len(shifts) or not sigmas:
        raise ValueError('Mismatched schedule and grid plan')
    with torch.no_grad():
        system.reset_update()
        aty = system.adjoint(system.y)
        x = torch.zeros_like(aty)  # Algorithm 1: zero image, not pULA initialization
        fov = slice(oracle.pad, oracle.pad+oracle.image_size)
        records = []
        nfe = 0
        for t, sigma in enumerate(sigmas):
            for k, grids in enumerate(shifts[t]):
                if not grids:
                    raise ValueError('Empty MM prediction group')
                # Freeze the state during all S predictions (Eq. 6).
                padded = F.pad(x, (oracle.pad,)*4)
                mean_pred = torch.zeros_like(x)
                for grid in grids:
                    noisy = padded + sigma*real_unit_noise(padded, generator)
                    pred = oracle.denoise_padded(noisy, sigma, grid)
                    mean_pred.add_(pred[:, :, fov, fov])
                    del pred, noisy
                    nfe += 1
                mean_pred.div_(len(grids))
                del padded
                # Eq. (10) in per-real-channel whitened coordinates.
                rhs = aty + mean_pred/sigma**2
                # DDS-style warm start: start CG at the denoised mean.
                x, cg_info = system.solve(rhs, mean_pred, sigma)
                del rhs, mean_pred
                if not torch.isfinite(x).all().item():
                    raise FloatingPointError(f'Nonfinite MM iterate at level {t+1}, iteration {k+1}')
                records.append(dict(level=t+1,mm_iteration=k+1,sigma=float(sigma),
                                    shifts=[list(g) for g in grids],**cg_info))
            if report is not None:
                report(t+1,len(sigmas),sigma,nfe,records[-1])
        return x, records
