"""Fixed diagonal complex image-error covariance, estimated on training data.

For already coil-whitened scalar noise variance v_y=E|n|^2, solve
  (I + A C A^H/v_y) q = r.
Then differentiate sqrt(real(r^H q)) with C fixed. This uses raw DPS units
and exactly recovers the original residual objective when C=0.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from eval_pula.score_opa import real_channels


def file_identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_coordinates(image_size, pad, psize, shift, device):
    y = (torch.arange(image_size,device=device)+pad-int(shift[0])) % psize
    x = (torch.arange(image_size,device=device)+pad-int(shift[1])) % psize
    return y[:,None]*psize+x[None,:]


class PatchCovariance:
    def __init__(self, path, model_path, image_size, pad, psize, device, mode='patch', scale=1.):
        import json
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['metadata'].item()))
            sigmas=np.asarray(f['sigmas'],dtype=np.float64)
            variance=np.asarray(f['variance'],dtype=np.float32)
        if meta['model']!=file_identity(model_path) or [meta[k] for k in ('image_size','pad','psize')] != [image_size,pad,psize]:
            raise ValueError('Covariance cache model/geometry mismatch; recalibrate')
        if meta.get('noise_convention')!='native_complex_dps' or variance.shape!=(len(sigmas),psize,psize) or len(sigmas)<2 or not np.all(np.diff(sigmas)>0) or not np.isfinite(variance).all() or np.any(variance<=0):
            raise ValueError('Invalid covariance cache')
        if mode not in ('patch','isotropic') or not math.isfinite(scale) or scale<0:
            raise ValueError('Invalid covariance mode/scale')
        self.sigmas=sigmas; self.variance=torch.as_tensor(variance,device=device)
        self.image_size=image_size; self.pad=pad; self.psize=psize
        self.mode=mode; self.scale=scale

    def map(self,sigma,shift):
        if sigma<self.sigmas[0]*(1-1e-6) or sigma>self.sigmas[-1]*(1+1e-6):
            raise ValueError(f'sigma={sigma} outside cache coverage')
        j=int(np.clip(np.searchsorted(self.sigmas,sigma),1,len(self.sigmas)-1))
        t=float(np.clip((math.log(sigma)-math.log(self.sigmas[j-1]))/(math.log(self.sigmas[j])-math.log(self.sigmas[j-1])),0,1))
        v=torch.lerp(self.variance[j-1].log(),self.variance[j].log(),t).exp()*self.scale
        if self.mode=='isotropic':
            return v.mean().expand(1,1,self.image_size,self.image_size)
        idx=patch_coordinates(self.image_size,self.pad,self.psize,shift,v.device)
        return v.flatten()[idx][None,None]


def inner(a,b):
    return (a.conj()*b).real.flatten(1).sum(dim=1)


def broadcast(x, like):
    return x.reshape((-1,)+(1,)*(like.ndim-1))


@torch.no_grad()
def solve_weighted_residual(operator,r,c,var_y,iters=8,rtol=1e-3):
    """Complex Hermitian CG; batch size one (current flex runner)."""
    if r.shape[0]!=1 or not isinstance(var_y,(float,int)) or not math.isfinite(var_y) or var_y<=0:
        raise ValueError('Covariance DPS requires scalar calibrated coil-whitened complex noise variance and B=1')
    if iters<1 or not 0<rtol<1 or not torch.isfinite(c).all() or (c<0).any():
        raise ValueError('Invalid covariance CG arguments')
    def matvec(q):
        return q+operator.forward(real_channels(c*operator.adjoint(q)))/var_y
    # Start from zero; for C=0 converges exactly in one iteration.
    q=torch.zeros_like(r); e=r.clone(); p=e.clone()
    rr=inner(e,e); rr0=rr.clone()
    if rr0.item()<=0: raise FloatingPointError('Zero residual')
    k=0
    for k in range(1,iters+1):
        ap=matvec(p); denom=inner(p,ap)
        if not torch.isfinite(denom).all() or denom.item()<=0:
            raise FloatingPointError('Covariance CG breakdown')
        alpha=rr/denom
        q=q+broadcast(alpha,q)*p; e=e-broadcast(alpha,e)*ap
        nxt=inner(e,e)
        if (nxt<=rtol**2*rr0).all():
            rr=nxt; break
        p=e+broadcast(nxt/rr,p)*p; rr=nxt
    err=r-matvec(q)
    rel=(inner(err,err)/rr0).sqrt().item()
    energy=inner(r,q)
    if not torch.isfinite(energy).all() or (energy<=0).any():
        raise FloatingPointError('Invalid weighted residual energy')
    return q, energy.sqrt(), {'cov_cg_iters':k,'cov_cg_relres':rel}
