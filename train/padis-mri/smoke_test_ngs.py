"""Actual-repo smoke test, run in the existing PaDIS directory before training."""
import argparse
import pickle
import io
import torch
from training.ngs_networks import NGSEDMPrecond
from training.ngs_loss import NGSOverlapLoss


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cbase',type=int,default=16)
    p.add_argument('--resolution',type=int,default=128)
    a=p.parse_args(); dev=torch.device(a.device)
    torch.manual_seed(123)
    m=NGSEDMPrecond(img_resolution=a.resolution,img_channels=4,out_channels=2,
        model_channels=a.cbase,channel_mult=[2,2,2],num_blocks=1,dropout=0.).to(dev)
    l=NGSOverlapLoss(lambda_overlap=1.25,alignment_mode='symmetric',pair_geometry='identical',spatial_weight_mode='uniform')
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    for size in (16,32,64):
        m.train();opt.zero_grad(set_to_none=True)
        x=torch.randn(1,2,a.resolution,a.resolution,device=dev)*.2
        loss=l(m,x,size,a.resolution).mean()
        assert torch.isfinite(loss)
        loss.backward()
        gs=[p.grad for p in m.ngs.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gs)
        if size==64: assert sum(g.abs().sum().item() for g in gs)>0
        else: assert all(g.count_nonzero().item()==0 for g in gs)
        opt.step();print(f'PASS {size}x{size} loss={loss.item():.6f}')
    m.eval()
    full=torch.randn(1,2,a.resolution,a.resolution,device=dev,requires_grad=True)
    sig=torch.tensor([.2],device=dev); pos=torch.zeros(1,2,64,64,device=dev); coords=torch.zeros(1,2,dtype=torch.long,device=dev)
    def forward(net):
        return net(full[:,:,:64,:64],sig,pos,prepared_global=net.prepare_global(full,sig),patch_coords=coords)
    out=forward(m)
    grad,=torch.autograd.grad(out.square().mean(),full)
    assert torch.isfinite(grad).all() and grad[:,:,64:,:].abs().sum().item()>0
    print('PASS full-context DPS gradient')
    buf=io.BytesIO();pickle.dump(m,buf);buf.seek(0);loaded=pickle.load(buf).eval()
    with torch.no_grad(): torch.testing.assert_close(forward(m),forward(loaded))
    print('PASS snapshot pickle round trip')

if __name__=='__main__': main()
