"""Standalone CPU tests requiring torch only, not fastMRI / real model weights."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint

HERE=Path(__file__).resolve().parent.parent
SANDBOX=Path(os.environ.get('JOINT_CT_REPO_ROOT','/mnt/SSD/wsy/projects/PaDIS-MRI-main'))


def fake_decorator(cls):
    class Decorator(cls):
        pass
    return Decorator


def install_stubs():
    persistence=types.ModuleType('torch_utils.persistence')
    persistence.persistent_class=fake_decorator
    stats=types.SimpleNamespace(report=lambda *a,**k: None)
    torch_utils=types.ModuleType('torch_utils')
    torch_utils.persistence=persistence
    torch_utils.training_stats=stats
    sys.modules['torch_utils']=torch_utils
    sys.modules['torch_utils.persistence']=persistence
    training=types.ModuleType('training')
    training.__path__=[]
    sys.modules['training']=training
    networks=types.ModuleType('training.networks')
    class StubPrecond(torch.nn.Module):
        def __init__(self,img_channels,out_channels=2,sigma_data=.5,**k):
            super().__init__()
            self.model=CaptureModel(out_channels)
            self.label_dim=0
            self.sigma_data=sigma_data
            self.use_fp16=False
            self.img_channels=img_channels
    networks.Patch_EDMPrecond=fake_decorator(StubPrecond)
    sys.modules['training.networks']=networks
    patch_loss=types.ModuleType('training.patch_loss')
    class Patch_EDMLoss:
        def __init__(self,P_mean=-1.2,P_std=1.2,sigma_data=.5):
            self.P_mean=P_mean;self.P_std=P_std;self.sigma_data=sigma_data
        def __call__(self,net,images,patch_size,resolution,labels=None,augment_pipe=None):
            return torch.zeros((1,),device=images.device)
    patch_loss.Patch_EDMLoss=Patch_EDMLoss
    sys.modules['training.patch_loss']=patch_loss
    overlap=types.ModuleType('training.patch_overlap_loss')
    def geom(images,psize):
        b=images.shape[0]; x=torch.zeros(b,dtype=torch.long,device=images.device)
        return x,x,x,x,True,psize
    def crop(images,top,left,psize):
        return torch.stack([a[:,int(t):int(t)+psize,int(l):int(l)+psize] for a,t,l in zip(images,top,left)])
    overlap._sample_overlap_geometry=geom
    overlap._extract_patches=crop
    overlap._position_maps=lambda top,left,b,psize,res,device: torch.zeros((b,2,psize,psize),device=device)
    overlap._overlap_views=lambda x,y,h,o:(x,y)
    overlap._boundary_distances=lambda *args:(torch.zeros(1),torch.zeros(1))
    sys.modules['training.patch_overlap_loss']=overlap
    return crop

class CaptureModel(torch.nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.linear=torch.nn.Conv2d(6,out_channels,1,bias=False)
        torch.nn.init.constant_(self.linear.weight,.02)
        self.last=None
    def forward(self,x,noise,class_labels=None,**kw):
        self.last=x
        return self.linear(x)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def test_network():
    mod=load_module('training.joint_context',HERE/'train/padis-mri/training/joint_context.py')
    net=mod.JointContextEDMPrecond(img_channels=4)
    x=torch.randn(2,2,64,64,requires_grad=True)
    g=torch.randn(2,2,64,64,requires_grad=True)
    pos=torch.zeros_like(x)
    pred=net(x,torch.tensor([1.,1.]),x_pos=pos,global_context=g)
    assert pred.shape==x.shape
    assert net.model.last.shape==(2,6,64,64)
    pred.mean().backward()
    assert x.grad is not None and g.grad is not None and g.grad.abs().sum()>0
    x2=torch.randn(1,2,16,16)
    net(x2,torch.tensor([1.]),x_pos=torch.zeros_like(x2))
    assert net.model.last[:,4:].abs().sum()==0
    try:
        net(x,torch.tensor([1.,1.]),x_pos=pos)
    except ValueError:
        pass
    else:
        raise AssertionError('64 input without full global must fail loudly')
    print('PASS network: 6 channels, EDM local skip, 16 branch, global gradient')


def test_opa():
    mod=load_module('training.patch_overlap_independent_loss',
                    SANDBOX/'train/padis-mri/training/patch_overlap_independent_loss.py')
    class CaptureNet:
        def __call__(self, x,sigma,x_pos=None,class_labels=None,augment_labels=None,global_context=None,**kw):
            assert global_context is not None
            assert x.shape==(4,2,64,64) and global_context.shape==(4,2,64,64)
            # upper-left patch starts at pixel zero and global 64x64 first cell
            # averages an 8x8 region in the SAME full noisy image.
            expected=x[:,:,:8,:8].mean(dim=(2,3))
            actual=global_context[:,:,0,0]
            assert torch.allclose(expected,actual,rtol=3e-5,atol=3e-6), (expected,actual)
            assert not torch.allclose(global_context[0],global_context[2])
            return x * .7
    loss=mod.IndependentNoiseOverlapPatch_EDMLoss(lambda_overlap=1.25,
                noise_mode='independent',alignment_mode='symmetric',
                pair_geometry='identical',spatial_weight_mode='uniform',
                joint_context=True)
    data=torch.randn(2,2,512,512)
    loss_map=loss(CaptureNet(),data,64,512)
    assert torch.isfinite(loss_map).all()
    assert loss_map.shape==(2,2,64,64)
    assert loss(CaptureNet(),data,16,512).sum()==0  # original Patch_EDMLoss
    print('PASS OPA: independent noisy full views, same-noise local/global, unchanged 16/32 path')


def test_eval():
    file=SANDBOX/'eval/denoise_padding.py'
    code=ast.parse(file.read_text(),filename=str(file))
    nodes=[n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='denoisedFromPatches']
    assert len(nodes)==1
    module=ast.Module(body=nodes,type_ignores=[])
    scope={'torch':torch,'np':np,'os':os,'checkpoint':checkpoint}
    exec(compile(module,str(file),'exec'),scope)
    f=scope['denoisedFromPatches']
    class FakeJoint:
        use_joint_global=True
        def __call__(self,xi,sigma,pi,ci,global_context=None):
            assert xi.shape[0]==global_context.shape[0]
            return xi + .1*global_context.mean(dim=(2,3),keepdim=True)
    x=torch.randn(1,2,512,512,requires_grad=True)
    pos=torch.zeros_like(x)
    starts=[64+i*64 for i in range(6)] # use 36 patches? pad calc =512-6*64=128
    # Use exact 49 random-partition positions covering the 384 FOV.
    starts=[i*64 for i in range(7)]
    idx=[[i,i+64,j,j+64] for i in starts for j in starts]
    for bs, ckpt in [('49','0'),('7','0'),('7','1')]:
        os.environ['PADIS_PATCH_BATCH']=bs
        os.environ['PADIS_PATCH_CHECKPOINT']=ckpt
        x.grad=None
        with torch.enable_grad():
            y=f(FakeJoint(),x,torch.tensor(1.),pos,None,idx,t_goal=0)
            assert y.shape==(1,2,512,512)
            loss=y.sum()
            loss.backward()
        assert x.grad is not None and x.grad.abs().sum()>0
        print(f'PASS DPS: patch_batch={bs}, checkpoint={ckpt}, global gradient')
    os.environ.pop('PADIS_PATCH_BATCH',None)
    os.environ.pop('PADIS_PATCH_CHECKPOINT',None)


if __name__=='__main__':
    install_stubs()
    test_network()
    test_opa()
    test_eval()
    print('ALL CPU TESTS PASS')
