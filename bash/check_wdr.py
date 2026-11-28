"""Check all patch sizes and print learned-layer + attention MACs per 64px patch."""
import argparse
import pickle
import torch
from training.networks import SongUNet, Conv2d, Linear, UNetBlock, haar_split, haar_merge

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', action='append', default=[])
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
torch.set_num_threads(2)
device = args.device
for path in args.snapshot:
    with open(path, 'rb') as stream:
        ema = pickle.load(stream)['ema'].eval().to(device)
    x = torch.randn(1,2,64,64,device=device,requires_grad=True)
    torch.autograd.grad(ema(x,x.new_ones(1),torch.zeros_like(x)).sum(),x)
    print('loaded',path)
    del ema

for variant in ('baseline','v0','v1','v2','v3','v4'):
    def build(**kw):
        return SongUNet(512,4,2,0,channel_mult=[2,2,2],**kw).to(device).eval()
    torch.manual_seed(123)
    net = build(wdr_variant=variant)
    if variant in ('baseline','v0'):
        reference = build(**({'num_blocks':[3,4,4],'wdr':True} if variant=='v0' else {}))
        reference.load_state_dict(net.state_dict(),strict=True)
    for p in (16,32,64):
        x = torch.randn(1,4,p,p,device=device,requires_grad=True)
        ll,h = haar_split(x)
        torch.testing.assert_close(haar_merge(ll,h),x,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(ll/2,torch.nn.functional.avg_pool2d(x,2))
        torch.testing.assert_close(haar_merge(2*ll,torch.zeros_like(h)),torch.nn.functional.interpolate(ll,scale_factor=2))
        macs = []
        def count(m, inputs, out):
            if isinstance(m,(Conv2d,torch.nn.Conv2d,Linear)) and m.weight is not None:
                macs.append(out.numel()*m.weight[0].numel())
            if isinstance(m,UNetBlock) and m.num_heads:
                macs.append(2*out.numel()*out.shape[-2]*out.shape[-1])
        handles = [m.register_forward_hook(count) for m in net.modules()] if p==64 else []
        y = net(x,x.new_zeros(1),None)
        for handle in handles:
            handle.remove()
        if variant in ('baseline','v0'):
            with torch.no_grad():
                torch.testing.assert_close(y, reference(x,x.new_zeros(1),None),rtol=0,atol=0)
        assert y.shape==(1,2,p,p)
        net.zero_grad(set_to_none=True)
        y.square().mean().backward()
        assert x.grad.isfinite().all() and x.grad.abs().max()>0
        assert all(w.grad is not None and w.grad.isfinite().all() for w in net.parameters())
    print(variant, sum(w.numel() for w in net.parameters()),sum(macs)/1e9,flush=True)
    if variant in ('baseline','v0'):
        del reference
    del net,y
