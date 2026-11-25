"""Estimate complex denoising MSE by within-patch position on TRAINING slices only."""
import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'eval'),str(ROOT/'train/padis-mri')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path',required=True)
    p.add_argument('--train_pt',help='Training .pt with complex x_est_gt; default from model training_options.json')
    p.add_argument('--out',required=True)
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--anchors',type=int,default=10)
    p.add_argument('--sigma_min',type=float,default=.003)
    p.add_argument('--sigma_max',type=float,default=.45)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--image_size',type=int,default=384)
    p.add_argument('--pad',type=int,default=64)
    p.add_argument('--psize',type=int,default=64)
    a=p.parse_args()
    if min(a.samples,a.anchors)<2 or not 0<a.sigma_min<a.sigma_max or a.pad!=a.psize or a.image_size%a.psize:
        p.error('Invalid samples, anchors, sigma range or geometry')
    model=Path(a.model_path).resolve()
    if a.train_pt is None:
        options=json.loads((model.parent/'training_options.json').read_text())
        a.train_pt=options['dataset_kwargs']['path']
    train=Path(a.train_pt).resolve()
    if not train.is_file(): raise FileNotFoundError(train)
    if 'eval' in train.name.lower() or 'val' in train.name.lower() or 'test' in train.name.lower() or any(v in ('val','test','fastmri_3t_eval') for v in map(str.lower,train.parts)):
        raise ValueError('Use the training dataset, not validation/test GT')
    import torch
    import torch.nn.functional as F
    import dnnlib
    from eval_pula.score_opa import PatchScore
    from eval_pula.recon_pula import native_complex_noise
    from eval_pula.flex_quality_cov import file_identity,patch_coordinates
    device=torch.device('cuda:0')
    with dnnlib.util.open_url(str(model),verbose=False) as f:
        net=pickle.load(f)['ema'].to(device).eval()
    oracle=PatchScore(net,a.image_size,a.pad,a.psize,device)
    try:
        data=torch.load(train,map_location='cpu',weights_only=False,mmap=True)
    except (RuntimeError,TypeError):
        data=torch.load(train,map_location='cpu',weights_only=False)
    gt=data['x_est_gt']
    if not isinstance(gt,torch.Tensor) or not gt.is_complex() or gt.ndim!=3 or tuple(gt.shape[-2:])!=(a.image_size,a.image_size):
        raise ValueError('Expected training x_est_gt complex tensor [N,H,W]')
    rng=random.Random(a.seed)
    indices=rng.sample(range(len(gt)),min(a.samples,len(gt)))
    if len(indices)<2: raise ValueError('Need >=2 training slices')
    gen=torch.Generator(device=device).manual_seed(a.seed)
    sigmas=np.geomspace(a.sigma_min,a.sigma_max,a.anchors)
    variances=[]
    fov=slice(a.pad,a.pad+a.image_size)
    with torch.no_grad():
        for sigma in sigmas:
            acc=torch.zeros(a.psize**2,device=device,dtype=torch.float64)
            num=torch.zeros_like(acc)
            for idx in indices:
                x=gt[idx].to(device=device,dtype=torch.complex64)[None,None]
                shift=(rng.randrange(a.pad),rng.randrange(a.pad))
                padded=F.pad(x,(a.pad,)*4)
                # Same perturbation convention as flex_dps.py, including padding.
                noisy=padded+float(sigma)*native_complex_noise(padded,gen)
                d=oracle.denoise_padded(noisy,float(sigma),shift)[:,:,fov,fov]
                err=(d-x).abs().square().reshape(-1).double()
                bins=patch_coordinates(a.image_size,a.pad,a.psize,shift,device).reshape(-1)
                acc.scatter_add_(0,bins,err)
                num.scatter_add_(0,bins,torch.ones_like(err))
            if (num==0).any() or not torch.isfinite(acc).all(): raise FloatingPointError('Invalid calibration statistics')
            v=(acc/num).reshape(a.psize,a.psize)
            # Numerical floor only; no empirical smoothing/tuning.
            v=v.clamp_min(max(v.mean().item()*1e-6,1e-12))
            variances.append(v.float().cpu().numpy())
            print(f'sigma={sigma:.6g} mean_complex_MSE={v.mean().item():.6g} min={v.min().item():.6g} max={v.max().item():.6g}',flush=True)
    meta=dict(model=file_identity(model),train=file_identity(train),indices=indices,seed=a.seed,
              image_size=a.image_size,pad=a.pad,psize=a.psize,noise_convention='native_complex_dps',
              estimator='training_complex_MSE_by_patch_position',patch_batch=os.environ.get('PADIS_PATCH_BATCH','49'))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(f'Cache already exists: {out}; use a new name')
    tmp=out.with_name(out.name+'.tmp.npz')
    np.savez_compressed(tmp,sigmas=sigmas,variance=np.stack(variances),metadata=json.dumps(meta))
    tmp.replace(out)
    print('Saved',out,flush=True)


if __name__=='__main__': main()
