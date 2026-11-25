"""CPU algebra and dispatcher checks (no actual checkpoint/GPU needed)."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'eval'),str(ROOT/'train/padis-mri')]
import torch
from types import SimpleNamespace
from eval_pula.flex_geometry_updates import SharedGeometry,GeometryDispatcher,geometry_dps_step
from eval_pula.flex_quality_updates import QualityDispatcher
from eval_pula.flex_dps import dps_step
from eval_pula.flex_pula import pula_initialize
from eval_pula.recon_pula import PULASystem

class Operator:
    def __init__(self):
        self.maps=torch.linspace(.1,2,16).reshape(1,1,4,4).to(torch.complex64)
    def forward(self,x): return self.maps*torch.complex(x[:,0:1],x[:,1:2])
    def adjoint(self,y): return (self.maps.conj()*y).sum(1,keepdim=True)
class Oracle:
    pad=2;image_size=4;psize=2
    def denoise_padded(self,x,sigma,shift): return .8*x
    @torch.no_grad()
    def denoise(self,x,sigma,shift): return .8*x
op=Operator(); oracle=Oracle(); gen=torch.Generator().manual_seed(123)
y=torch.randn((1,1,4,4),dtype=torch.complex64,generator=gen)
g=torch.randn(y.shape,dtype=y.dtype,generator=gen)
system=SimpleNamespace(operator=op,system_sigma=lambda s:s)
geo=SharedGeometry(system,2)
p,info=geo.direction(g,.5)
B=op.maps.abs().square()+4; eta=1/(float(B.max()))
expected=(2-eta*B)*g; expected*=g.norm()/expected.norm()
torch.testing.assert_close(p,expected,rtol=1e-6,atol=1e-6)
assert info['geometry_cos'][0]>0
assert info['geometry_relative_change'][0]>0
assert torch.equal(geo.direction(torch.zeros_like(g),.5)[0],torch.zeros_like(g))
print('PASS polynomial formula, norm, descent and zero gradient')
# Identity curvature -> unchanged direction (and same DPS noise sequence).
op_identity=Operator();op_identity.maps.fill_(1)
geo_id=SharedGeometry(SimpleNamespace(operator=op_identity,system_sigma=lambda s:s),2)
x=torch.randn((1,1,8,8),dtype=torch.complex64,generator=gen)
ga=torch.Generator().manual_seed(99);gb=torch.Generator().manual_seed(99)
a,_=dps_step(x,oracle,op_identity,y,ga,.1,(0,0),3.,False)
b,_=geometry_dps_step(x,oracle,op_identity,y,gb,.1,(0,0),3.,False,geo_id)
torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
assert torch.equal(ga.get_state(),gb.get_state())
print('PASS scalar geometry recovers DPS and identical RNG')
# Parent path off: exact numerical equivalence, including mixed dispatch.
def fixture(cls,**kwargs):
    sys=PULASystem(op,y,torch.Generator().manual_seed(77),10,1e-6)
    state,_=pula_initialize(sys,.5)
    return cls(state,sys,oracle,op,y,**kwargs)
a=fixture(QualityDispatcher);b=fixture(GeometryDispatcher,geometry_mode='off')
for m,s in zip('PDPDD',[.5,.3,.2,.1,.05]):
    a.step(m,s,(0,0),.5,3.,False);b.step(m,s,(0,0),.5,3.,False,mixed_level=True)
    torch.testing.assert_close(a.state,b.state,rtol=0,atol=0)
print('PASS off exact equivalence')
c=fixture(GeometryDispatcher,geometry_mode='mixed')
assert 'geometry_cos' not in c.step('P',.5,(0,0),.5,3.,False,mixed_level=True)
assert 'geometry_cos' in c.step('D',.3,(0,0),.5,3.,False,mixed_level=True)
assert 'geometry_cos' not in c.step('D',.1,(0,0),.5,3.,True,mixed_level=False)
assert torch.isfinite(c.state).all()
print('PASS mixed-only selection')
