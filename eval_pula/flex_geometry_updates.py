"""Shared pULA normal-operator geometry for DPS, without extra NN or CG.
All baseline P/D kernels and random draws are retained. This is a heuristic
preconditioned DC update, not a claim of exact posterior invariance.
"""
from math import sqrt
import torch
import torch.nn.functional as F
from eval_pula.flex_quality_updates import QualityDispatcher
from eval_pula.recon_pula import native_complex_noise
from eval_pula.score_opa import real_channels


class SharedGeometry:
    def __init__(self, system, order=2):
        if order < 2:
            raise ValueError('Polynomial order must be >= 2')
        self.system = system
        self.order = order
        # A_w = P F S, with orthonormal FFT and binary mask. Thus ||A_w||^2
        # <= max_pixel sum_coils |S_w|^2 (including coil whitening).
        self.bound = float(system.operator.maps.detach().abs().square().sum(dim=1).max())
        if not 0 <= self.bound < float('inf'):
            raise ValueError('Invalid whitened coil-map spectral bound')

    @torch.no_grad()
    def direction(self, g, sigma):
        """Norm-matched truncated Neumann approximation to B_sigma^{-1} g.

        B=A_w^H A_w + sigma_system^{-2} I; eta=1/(L+sigma_system^{-2}).
        M_m=eta sum_{j=0}^{m-1}(I-eta B)^j. The positive scalar eta cancels
        in normalization. The spectrum of I-eta B is in [0,1).
        """
        invvar = self.system.system_sigma(sigma)**-2
        eta = 1.0/(self.bound+invvar)
        term = g
        direction = g.clone()
        for _ in range(1,self.order):
            # Avoid subtracting two large nearly equal values in I-eta*B.
            term = (self.bound*eta)*term - eta*self.system.operator.adjoint(
                self.system.operator.forward(real_channels(term)))
            direction = direction + term
        gn = torch.linalg.vector_norm(g.flatten(1),dim=1)
        pn = torch.linalg.vector_norm(direction.flatten(1),dim=1)
        factor = torch.where(pn>0,gn/pn.clamp_min(torch.finfo(pn.dtype).tiny),torch.ones_like(pn))
        direction = direction*factor[:,None,None,None]
        dot = (g.conj()*direction).real.flatten(1).sum(1)
        den = gn.square()
        cos = torch.where(den>0,dot/den.clamp_min(torch.finfo(den.dtype).tiny),torch.ones_like(den))
        rel = torch.where(gn>0,torch.linalg.vector_norm((direction-g).flatten(1),dim=1)/gn.clamp_min(torch.finfo(gn.dtype).tiny),torch.zeros_like(gn))
        return direction, {'geometry_cos':cos.cpu().tolist(),'geometry_relative_change':rel.cpu().tolist()}


def geometry_dps_step(padded, oracle, raw, y, generator, sigma, shift, zeta,
                      final_level, geometry):
    fov=slice(oracle.pad,oracle.pad+oracle.image_size)
    alpha=.5*sigma**2
    # Same stochastic denoiser input, same measurement loss, same VJP as baseline.
    with torch.enable_grad():
        current=padded.detach().requires_grad_(True)
        noisy=current+sigma*native_complex_noise(current,generator)
        denoised=oracle.denoise_padded(noisy,sigma,shift)
        residual=y-raw.forward(real_channels(denoised[:,:,fov,fov]))
        norm=torch.linalg.vector_norm(residual.reshape(residual.shape[0],-1),dim=-1)
        if not torch.isfinite(norm).all().item() or (norm==0).any().item():
            raise FloatingPointError('Invalid DPS residual norm')
        grad=torch.autograd.grad(norm.square().sum(),current)[0]
    # Once backward has finished, the MRI normal operator acts on a detached
    # gradient. No graph, no Hessian, no additional activation checkpoint.
    with torch.no_grad():
        score=(denoised.detach()-noisy.detach())/sigma**2
        norm=norm.detach()
        new_fov,info=geometry.direction(grad[:,:,fov,fov].detach(),sigma)
        grad=grad.detach().clone()
        grad[:,:,fov,fov]=new_fov
        updated=current.detach()-(zeta/norm[:,None,None,None])*grad
        updated=updated+(alpha/2)*score
        if not final_level:
            updated=updated+sqrt(alpha)*native_complex_noise(updated,generator)
        info.update(measurement_residual_norm=norm.cpu().tolist(),geometry_normal_calls=geometry.order-1)
    return updated.detach(),info


class GeometryDispatcher(QualityDispatcher):
    def __init__(self,*args,geometry_mode='off',poly_order=2,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry_mode=geometry_mode
        self.geometry=SharedGeometry(self.system,poly_order) if geometry_mode!='off' else None
        if self.geometry is not None and self.covariance is not None:
            raise ValueError('Geometry and covariance modes must be isolated')

    def step(self,method,sigma,shift,gamma,zeta,final_level,mixed_level=False):
        use=method=='D' and (self.geometry_mode=='all' or (self.geometry_mode=='mixed' and mixed_level))
        if not use:
            return super().step(method,sigma,shift,gamma,zeta,final_level)
        if self.padded is None:
            self.padded=F.pad(self.state,(self.oracle.pad,)*4)
        elif self.previous=='P':
            self.padded=self.padded.detach().clone()
            self.padded[:,:,self.fov,self.fov]=self.state
        self.padded,info=geometry_dps_step(self.padded,self.oracle,self.raw,self.y,
                     self.system.generator,sigma,shift,zeta,final_level,self.geometry)
        self.state=self.padded[:,:,self.fov,self.fov].detach()
        self.previous=method
        return info
