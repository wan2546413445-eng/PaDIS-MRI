"""Single-step baseline pULA: original whitened system and CG."""
import torch


@torch.no_grad()
def pula_initialize(system, sigma):
    system.reset_update()
    rhs, initial = system.initialization_rhs(sigma)
    return system.solve(rhs, initial, sigma)


@torch.no_grad()
def pula_step(state, system, oracle, sigma, shift, gamma):
    """Exactly one baseline Langevin update on [1,1,H,W] complex state."""
    denoised = oracle.denoise(state, sigma, shift)
    prior_score = (denoised - state) / system.system_sigma(sigma)**2
    rhs, initial = system.update_rhs(state, prior_score, sigma, gamma)
    delta, info = system.solve(rhs, initial, sigma)
    return state + delta, info
