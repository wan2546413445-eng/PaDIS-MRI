import torch
from training.networks import SongUNet, haar_merge, haar_split

torch.manual_seed(123)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SongUNet(512, 4, 2, 0, channel_mult=[2,2,2], num_blocks=[3,4,4], wdr=True).to(device)
for p in (16,32,64):
    x = torch.randn(1,4,p,p,device=device,requires_grad=True)
    ll,h = haar_split(x)
    torch.testing.assert_close(haar_merge(ll,h),x,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(ll/2,torch.nn.functional.avg_pool2d(x,2))
    torch.testing.assert_close(haar_merge(2*ll,torch.zeros_like(h)),torch.nn.functional.interpolate(ll,scale_factor=2))
    y = net(x,x.new_zeros(1),None)
    grads = torch.autograd.grad(y.square().mean(), (x,net.high_filter.weight,net.high_prediction.weight))
    assert y.shape == (1,2,p,p) and all(g.isfinite().all() and g.abs().max()>0 for g in grads)
    print(p, 'passed')
