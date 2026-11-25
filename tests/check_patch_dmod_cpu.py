"""CPU checks for Patch-DMOD. Run from the PaDIS repository root."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from eval_pula.patch_dmod import make_sigmas, make_grid_plan, reconstruct_patch_dmod
from eval_pula.recon_pula import real_unit_noise
from eval_pula.cg import solve_cg


class ToySystem:
    """Positive diagonal normal operator with an analytic solution."""
    convention='padis_real'
    def __init__(self):
        self.diag=torch.tensor([[[[2.,3.],[4.,5.]]]])
        self.y=torch.tensor([[[[1+2j,2-1j],[3+1j,1-2j]]]],dtype=torch.complex64)
        self.calls=[]
    def reset_update(self): pass
    def adjoint(self,y): return y*self.diag.sqrt()
    def solve(self,rhs,initial,sigma):
        diagonal=self.diag+sigma**-2
        sol,info=solve_cg(lambda x:diagonal*x,rhs,initial,10,1e-7)
        torch.testing.assert_close(sol,rhs/diagonal,atol=1e-6,rtol=1e-5)
        self.calls.append((rhs.clone(),initial.clone(),sigma))
        return sol,info


class ToyOracle:
    pad=2;image_size=2
    def __init__(self): self.calls=[]
    def denoise_padded(self, state, sigma, shift):
        assert not torch.is_grad_enabled()
        self.calls.append((state.clone(),sigma,shift))
        return .6*state+complex(shift[0],shift[1])*.01


def main():
    for T in (30,47):
        sigmas=make_sigmas(T,1,.003,7)
        plan=make_grid_plan(T,1,5,64,123)
        assert sigmas[0]==1 and sigmas[-1]==.003 and all(a>b for a,b in zip(sigmas,sigmas[1:]))
        assert plan==make_grid_plan(T,1,5,64,123)
        assert all(len(set(group))==5 and all(0<=i<64 and 0<=j<64 for i,j in group) for level in plan for group in level)
        assert sum(len(g) for level in plan for g in level)==T*5
    print('PASS: T30/T47 schedule, uniform distinct partitions, 150/235 NFE')
    gen=torch.Generator().manual_seed(123)
    ref_gen=torch.Generator().manual_seed(123)
    toy=ToySystem();oracle=ToyOracle()
    sigmas=[.5,.2];plan=make_grid_plan(2,2,3,2,123)
    x,records=reconstruct_patch_dmod(toy,oracle,sigmas,plan,gen)
    assert len(oracle.calls)==12 and len(toy.calls)==len(records)==4
    ref=torch.zeros_like(toy.y);aty=toy.adjoint(toy.y);slot=0;mm=0
    for sigma,level in zip(sigmas,plan):
        for group in level:
            padded=F.pad(ref,(2,)*4); preds=[]
            for shift in group:
                noisy=padded+sigma*real_unit_noise(padded,ref_gen)
                seen,s,g=oracle.calls[slot];slot+=1
                torch.testing.assert_close(noisy,seen)
                assert s==sigma and g==shift
                assert noisy[:,:,:2,:].abs().max()>0 # padding was renoised
                preds.append((.6*noisy+complex(*shift)*.01)[:,:,2:4,2:4])
            mean=torch.stack(preds).mean(0)
            rhs=aty+mean/sigma**2
            saved_rhs,initial,saved_sigma=toy.calls[mm];mm+=1
            torch.testing.assert_close(rhs,saved_rhs)
            torch.testing.assert_close(mean,initial)
            assert saved_sigma==sigma
            ref=rhs/(toy.diag+sigma**-2)
    torch.testing.assert_close(x,ref,rtol=1e-5,atol=1e-6)
    print('PASS: zero initialization, frozen iterate for S draws, full-padding renoise, exact MM RHS, DDS warm start, complex CG analytic solution, K>1 sequencing')
    a=real_unit_noise(torch.zeros(100000,dtype=torch.complex64),torch.Generator().manual_seed(123))
    assert abs(a.real.var().item()-1)<.025 and abs(a.imag.var().item()-1)<.025
    print('PASS: per-real-channel noise variance ~1')

if __name__=='__main__':main()
