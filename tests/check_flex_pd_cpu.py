"""Run on project server: python tests/check_flex_pd_cpu.py (CPU only).
Compare new single-step composition with old reconstruct on a small fixture.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'eval'),str(ROOT/'train/padis-mri')]
import torch
import torch.nn.functional as F
from eval_pula.recon_pula import PULASystem,reconstruct
from eval_pula.flex_updates import pula_initialize,UpdateDispatcher
from eval_pula.flex_schedule import make_schedule,make_sigmas,counts

class Operator:
    def forward(self,x): return torch.complex(x[:,0:1],x[:,1:2])
    def adjoint(self,y): return y

class Net(torch.nn.Module):
    sigma_data=.5
    def forward(self,x): return .8*x

class Oracle:
    image_size=4;pad=2;psize=2
    def __init__(self): self.net=Net()
    def denoise_padded(self,x,sigma,shift): return self.net(x)
    @torch.no_grad()
    def denoise(self,x,sigma,shift):
        f=slice(self.pad,self.pad+self.image_size)
        return self.denoise_padded(F.pad(x,(self.pad,)*4),sigma,shift)[:,:,f,f]

op=Operator(); y=torch.full((1,1,4,4),.25+.1j,dtype=torch.complex64)
sigmas=[.5,*make_sigmas(4,.5,.05,'karras')]; shifts=[(0,0),(1,1)]
def setup():
    return PULASystem(op,y,torch.Generator().manual_seed(123),10,1e-6),Oracle()
def old(switch):
    sys,oracle=setup()
    state,*_=reconstruct(sys,oracle,sigmas,shifts,.5,pula_levels=switch,
                        dps_operator=op,dps_measurement=y,dps_sigmas=sigmas[1+switch:],
                        dps_shifts=shifts*(4-switch),dps_inner_loops=2,dps_zeta=3.)
    return state

def new(rows):
    sys,oracle=setup(); x,_=pula_initialize(sys,sigmas[0]); d=UpdateDispatcher(x,sys,oracle,op,y)
    for i,row in enumerate(rows):
        for j,method in enumerate(row):d.step(method,sigmas[i+1],shifts[j],.5,3.,i==3)
    return d.state.squeeze(1)

for switch in [0,2,4]:
    a=old(switch);b=new(['PP']*switch+['DD']*(4-switch))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    print(f'PASS stage switch {switch}: max diff {(a-b).abs().max().item():.3g}')
for pattern in ['PD','DP']:
    a=new([pattern]*4);b=new([pattern]*4)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    print(f'PASS {pattern}: finite, deterministic')
assert counts(make_schedule(47,blocks='1-21:PPPDD;22-47:PPDDD'))['pula_updates']==115
print('CPU fixture checks passed')
