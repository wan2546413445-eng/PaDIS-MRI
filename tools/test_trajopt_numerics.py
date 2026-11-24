#!/usr/bin/env python3
"""CPU-only checks for the isolated noisy-input optimization sampler."""
import math
import torch

from eval_pula.recon_trajopt import optimize_noisy_state, reconstruct_trajopt
from eval_pula.run import karras_schedule


def close(x,y,tol=2e-5):
    if not torch.allclose(x,y,rtol=tol,atol=tol):
        raise AssertionError(f"max_err={(x-y).abs().max().item():.5g}")


class IdentitySystem:
    def __init__(self, shape=(1,1,4,4), y=None):
        self.y = torch.zeros(shape,dtype=torch.complex64) if y is None else y
        self.convention = "padis_real"
        self.generator = torch.Generator().manual_seed(123)
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.solve_calls = 0

    def forward(self, x):
        self.forward_calls+=1
        return x

    def adjoint(self,y):
        self.adjoint_calls+=1
        return y

    def solve(self, rhs, initial, sigma):
        self.solve_calls+=1
        denom=1+sigma**-2
        out = rhs/denom
        return out, {"iterations":1,"relative_residual":0.,"residual_norm":0.}

    def reset_update(self):
        pass

    def initialization_rhs(self,sigma):
        # The mock uses the same CG normal, but a deterministic initialization.
        return self.y, torch.zeros_like(self.y)

    def update_rhs(self, x, score, sigma, gamma):
        # For tests: deterministic Langevin direction (not a physics model).
        rhs = gamma * (self.adjoint(self.y-self.forward(x)) + score)
        return rhs, torch.zeros_like(rhs)


class LinearPatchOracle:
    def __init__(self, scale=0.7, image_size=4, pad=1):
        self.scale = scale
        self.image_size=image_size
        self.pad=pad
        self.calls=0
        self.shifts=[]

    def denoise_padded(self, x, sigma, shift):
        self.calls+=1
        self.shifts.append(tuple(shift))
        return self.scale*x

    @torch.no_grad()
    def denoise(self, state, sigma, shift):
        self.calls+=1
        self.shifts.append(tuple(shift))
        return self.scale*state


def test_identity_input_vjp_analytic():
    torch.manual_seed(5)
    state=torch.randn(1,1,4,4,dtype=torch.complex64)
    y=torch.randn_like(state)
    s=IdentitySystem(y=y)
    oracle=LinearPatchOracle(scale=0.7)
    sigma=0.45
    optimized,diag,_=optimize_noisy_state(s,oracle,state,sigma,(1,1),
                                step_scale=1.,trust_sigma=1e5)
    analytic_grad=.7*(.7*state-y)
    expected=state-analytic_grad/(1+sigma**-2)
    close(optimized,expected)
    if diag["step_clip_factor"]!=1.:
        raise AssertionError("Unclipped analytic step was clipped")
    if s.solve_calls!=1 or oracle.calls!=1:
        raise AssertionError("Optimizer VJP uses one forward and one CG")


def test_complex_gradient_sense_fft():
    # Unit-test 0.5 complex SSE gradient vs analytic A^H r for a masked
    # multi-coil FFT, matching the forward/backprop conventions used by DPS.
    torch.manual_seed(7)
    u=torch.randn(1,1,8,8,dtype=torch.complex64,requires_grad=True)
    maps=torch.tensor([.8+.1j,.3-.5j],dtype=torch.complex64).view(1,2,1,1)
    mask=torch.zeros(1,1,8,8)
    mask[...,::2,:]=1.
    def A(x):
        return mask*torch.fft.fft2(maps*x,norm="ortho")
    def AH(y):
        return (maps.conj()*torch.fft.ifft2(mask*y,norm="ortho")).sum(1,keepdim=True)
    measured=A(torch.randn_like(u))
    a=.6
    residual=A(a*u)-measured
    actual=torch.autograd.grad(0.5*residual.abs().square().sum(),u)[0]
    expected=a*AH(residual.detach())
    close(actual,expected)


def test_trust_radius_and_grad_detach():
    state=torch.ones(1,1,4,4,dtype=torch.complex64)
    y=torch.full_like(state,20.)
    system=IdentitySystem(y=y)
    oracle=LinearPatchOracle(scale=1.)
    optimized,diag,_=optimize_noisy_state(system,oracle,state,.2,(0,0),
                      step_scale=20,trust_sigma=.25)
    if diag["step_rms"]>.05+1e-6:
        raise AssertionError("Trust radius exceeded")
    if diag["step_clip_factor"]>=1:
        raise AssertionError("Expected clipping")
    if optimized.requires_grad:
        raise AssertionError("Optimizer leaked autograd graph")


def test_full_budget_and_shifts():
    grid=karras_schedule(47,.003,1.,7.)
    grid[0],grid[-1]=1.,.003
    shifts=[(0,i) for i in range(5)]
    system=IdentitySystem()
    oracle=LinearPatchOracle()
    state,timing=reconstruct_trajopt(system,oracle,grid,shifts,.5,
                          start_level=23,step_scale=1.,trust_sigma=.5,
                          report=lambda _:None)
    expected={
      "network_forwards":235,
      "network_backprops":24,
      "pula_updates":211,
      "optimized_input_updates":24,
      "cg_solves":236,
      "pula_cg_solves":211,
      "optimization_cg_solves":24,
    }
    for key,value in expected.items():
        if timing[key]!=value:
            raise AssertionError(f"{key}: expected {value}; got {timing[key]}")
    if oracle.calls!=235 or system.solve_calls!=236:
        raise AssertionError("Actual model/CG call count mismatch")
    if len(oracle.shifts)!=235 or oracle.shifts[:5]!=shifts or oracle.shifts[-5:]!=shifts:
        raise AssertionError("Patch grid sequence changed")
    if not torch.isfinite(state).all().item() or state.requires_grad:
        raise AssertionError("Invalid final state")
    if len(timing["opt_layers"])!=24:
        raise AssertionError("Incorrect number of optimized levels")
    if timing["opt_layers"][0]["level_number_one_based"]!=24:
        raise AssertionError("Off-by-one at start_level=23")


def test_optimizer_has_no_extra_noise_draw():
    system=IdentitySystem(y=torch.ones(1,1,4,4,dtype=torch.complex64))
    oracle=LinearPatchOracle()
    state=torch.zeros(1,1,4,4,dtype=torch.complex64)
    initial_rng=system.generator.get_state().clone()
    optimize_noisy_state(system,oracle,state,.1,(0,0))
    if not torch.equal(initial_rng,system.generator.get_state()):
        raise AssertionError("Noisy state optimization unexpectedly used the Langevin RNG")


def main():
    tests=[test_identity_input_vjp_analytic,test_complex_gradient_sense_fft,
           test_trust_radius_and_grad_detach,test_full_budget_and_shifts,
           test_optimizer_has_no_extra_noise_draw]
    for fn in tests:
        fn()
        print('PASS',fn.__name__)
    print(f'PASS all {len(tests)} TrajOpt CPU checks')


if __name__=='__main__':
    main()
