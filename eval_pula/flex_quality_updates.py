"""Opt-in padding bridge and covariance DPS; baseline kernels unchanged."""
from math import sqrt
import torch
import torch.nn.functional as F
from eval_pula.flex_pula import pula_step, pula_initialize
from eval_pula.flex_dps import dps_step
from eval_pula.recon_pula import native_complex_noise
from eval_pula.score_opa import real_channels
from eval_pula.flex_quality_cov import solve_weighted_residual


def covariance_dps_step(padded,oracle,raw,y,generator,sigma,shift,zeta,final_level,
                        covariance,var_y,cg_iters,cg_rtol):
    fov=slice(oracle.pad,oracle.pad+oracle.image_size)
    alpha=.5*sigma**2
    with torch.enable_grad():
        current=padded.detach().requires_grad_(True)
        noisy=current+sigma*native_complex_noise(current,generator)
        denoised=oracle.denoise_padded(noisy,sigma,shift)
        r=y-raw.forward(real_channels(denoised[:,:,fov,fov]))
        c=covariance.map(sigma,shift)
        q,norm,info=solve_weighted_residual(raw,r.detach(),c,var_y,cg_iters,cg_rtol)
        # d(r^H W r)/dx = 2 Re[(W r)^H dr/dx], W fixed.
        # The detached solve avoids differentiating through CG and adds no NN calls.
        surrogate=2*(q.conj()*r).real.sum()
        grad=torch.autograd.grad(surrogate,current)[0]
        score=(denoised-noisy)/sigma**2
        updated=current-(zeta/norm[:,None,None,None])*grad+(alpha/2)*score
        if not final_level:
            updated=updated+sqrt(alpha)*native_complex_noise(updated,generator)
    info.update(measurement_residual_norm=torch.linalg.vector_norm(r.detach().flatten(1),dim=1).cpu().tolist(),
                weighted_residual_norm=norm.cpu().tolist(),covariance_mean=c.mean().item())
    return updated.detach(),info


class QualityDispatcher:
    def __init__(self,state,system,oracle,raw_operator,raw_measurement,padding_mode='reset',
                 covariance=None,var_y=None,cov_cg_iters=8,cov_cg_rtol=1e-3):
        self.state=state; self.system=system; self.oracle=oracle
        self.raw=raw_operator; self.y=raw_measurement; self.padded=None
        self.fov=slice(oracle.pad,oracle.pad+oracle.image_size)
        self.padding_mode=padding_mode; self.covariance=covariance; self.var_y=var_y
        self.cov_cg_iters=cov_cg_iters; self.cov_cg_rtol=cov_cg_rtol
        self.previous=None
    def step(self,method,sigma,shift,gamma,zeta,final_level):
        if method=='P':
            if self.padding_mode=='reset': self.padded=None
            self.state,info=pula_step(self.state,self.system,self.oracle,sigma,shift,gamma)
        elif method=='D':
            if self.padded is None:
                self.padded=F.pad(self.state,(self.oracle.pad,)*4)
            elif self.previous=='P':
                # retain perimeter; replace only FOV by newest pULA state
                self.padded=self.padded.detach().clone()
                self.padded[:,:,self.fov,self.fov]=self.state
            args=(self.padded,self.oracle,self.raw,self.y,self.system.generator,sigma,shift,zeta,final_level)
            if self.covariance is None:
                self.padded,info=dps_step(*args)
            else:
                self.padded,info=covariance_dps_step(*args,self.covariance,self.var_y,self.cov_cg_iters,self.cov_cg_rtol)
            self.state=self.padded[:,:,self.fov,self.fov].detach()
        else:
            raise ValueError(method)
        self.previous=method
        return info
