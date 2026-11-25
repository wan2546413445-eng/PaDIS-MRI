"""CPU-only checks; run after installing on existing Flex-PD project."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'eval'),str(ROOT/'train/padis-mri')]
import torch
from eval_pula.flex_grids import build_grid_plan,grid_summary
from eval_pula.flex_quality_grids import quality_grid_plan
from eval_pula.flex_schedule import make_schedule,counts
from eval_pula.flex_quality_cov import solve_weighted_residual,patch_coordinates
from eval_pula.flex_quality_updates import QualityDispatcher,covariance_dps_step
from eval_pula.flex_dps import dps_step
from eval_pula.flex_updates import UpdateDispatcher
from eval_pula.recon_pula import PULASystem
from eval_pula.flex_pula import pula_initialize
import torch.nn.functional as F

rows=make_schedule(47,blocks='1-10:PPPPP;11-25:PPPDD;26-35:PPDDD;36-47:DDDDD')
base=build_grid_plan(rows,64,123,'cycle_k','cycle_k','random_step')
assert base==quality_grid_plan(rows,64,123,'cycle_k','cycle_k','random_step')
assert counts(rows)['pula_updates']==115 and counts(rows)['dps_updates']==120
for pm,dm in [('sobol','random_step'),('cycle_k','sobol'),('sobol','sobol')]:
    g=quality_grid_plan(rows,64,123,'cycle_k',pm,dm)
    assert g==quality_grid_plan(rows,64,123,'cycle_k',pm,dm)
    assert all(0<=z<64 for row in g for shift in row for z in shift)
    assert all(g[i][j]==base[i][j] for i,r in enumerate(rows) for j,m in enumerate(r) if (m=='P' and pm!='sobol') or (m=='D' and dm!='sobol'))
    print('PASS grid',pm,dm,grid_summary(rows,g))
assert torch.equal(patch_coordinates(4,2,2,(1,0),'cpu'),torch.tensor([[2,3,2,3],[0,1,0,1],[2,3,2,3],[0,1,0,1]]))

class Operator:
    def __init__(self): self.maps=torch.tensor([[[[1+.2j]],[[.4-.3j]]]],dtype=torch.complex64)
    def forward(self,x): return self.maps*torch.complex(x[:,0:1],x[:,1:2])
    def adjoint(self,y): return (self.maps.conj()*y).sum(1,keepdim=True)
class Oracle:
    pad=2;image_size=4;psize=2
    def denoise_padded(self,x,sigma,shift): return .8*x
    @torch.no_grad()
    def denoise(self,x,sigma,shift): return .8*x
class Cov:
    def __init__(self,c): self.c=c
    def map(self,*args):return self.c
op=Operator();oracle=Oracle()
gen=torch.Generator().manual_seed(123)
r=torch.randn((1,2,4,4),dtype=torch.complex64,generator=gen)
c=torch.linspace(.01,.3,16).reshape(1,1,4,4)
q,n,info=solve_weighted_residual(op,r,c,.02,80,1e-6)
# analytic inverse of 2x2 rank-one covariance per pixel
s=op.maps
expected=r-(c/(.02+c*s.abs().square().sum(1,keepdim=True)))*s*(s.conj()*r).sum(1,keepdim=True)
torch.testing.assert_close(q,expected,rtol=2e-5,atol=2e-6)
assert info['cov_cg_relres']<1e-5
print('PASS covariance CG against analytic solution',info)
x=torch.randn((1,1,8,8),dtype=torch.complex64,generator=gen)
a,_=dps_step(x,oracle,op,r,torch.Generator().manual_seed(99),.1,(0,0),3.,False)
b,_=covariance_dps_step(x,oracle,op,r,torch.Generator().manual_seed(99),.1,(0,0),3.,False,Cov(torch.zeros_like(c)),.02,16,1e-3)
torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
print('PASS C=0 DPS equivalence')
# Actual pULA + DPS dispatchers, default path must be unchanged.
def fixture(cls,mode='reset'):
    system=PULASystem(op,r,torch.Generator().manual_seed(77),10,1e-6)
    state,_=pula_initialize(system,.5)
    return cls(state,system,oracle,op,r) if cls is UpdateDispatcher else cls(state,system,oracle,op,r,mode)
a=fixture(UpdateDispatcher); b=fixture(QualityDispatcher)
for method,sigma in zip('PDPDD',[.5,.3,.2,.1,.05]):
    a.step(method,sigma,(0,0),.5,3.,False);b.step(method,sigma,(0,0),.5,3.,False)
    torch.testing.assert_close(a.state,b.state,rtol=0,atol=0)
print('PASS default dispatcher exact equivalence')
p=fixture(QualityDispatcher,'persistent');p.step('D',.3,(0,0),.5,3.,False)
prev=p.padded.clone();p.step('P',.2,(0,0),.5,3.,False)
torch.testing.assert_close(p.padded,prev,rtol=0,atol=0)
p.step('D',.1,(0,0),.5,3.,False)
assert torch.isfinite(p.state).all()
print('PASS persistent bridge finite, retained outer state during P')
