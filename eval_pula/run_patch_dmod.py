"""Standalone Patch-DMOD runner; original PaDIS evaluation and I/O conventions."""
import argparse
import csv
import json
import math
import os
import pickle
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'eval'),str(ROOT/'train/padis-mri')]
from eval_pula.patch_dmod import make_sigmas, make_grid_plan, reconstruct_patch_dmod


def sample_list(value):
    # Same inclusive index/range convention as eval_pula.run.sample_list.
    import re
    indices=[]
    for token in value.split(','):
        m=re.fullmatch(r'(\d+)(?:-(\d+))?',token.strip())
        if not m:
            raise argparse.ArgumentTypeError(f'Invalid sample or inclusive range: {token}')
        first,last=int(m[1]),int(m[2] or m[1])
        if last<first:
            raise argparse.ArgumentTypeError(f'Reversed range: {token}')
        indices.extend(range(first,last+1))
    return list(dict.fromkeys(indices))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path',required=True)
    p.add_argument('--val_dir',required=True)
    p.add_argument('--out_root',required=True)
    p.add_argument('--sample_indices',type=sample_list,default='0,26')
    p.add_argument('--mask_select',type=int,default=4)
    p.add_argument('--T',type=int,default=30)
    p.add_argument('--S',type=int,default=5)
    p.add_argument('--K',type=int,default=1)
    p.add_argument('--cg_iters',type=int,default=10)
    p.add_argument('--cg_rtol',type=float,default=1e-6)
    p.add_argument('--sigma_max',type=float,default=1.)
    p.add_argument('--sigma_min',type=float,default=.003)
    p.add_argument('--rho',type=float,default=7.)
    p.add_argument('--beta_scale',type=float,default=1.,help='Multiplier of calibrated measurement noise STD, not variance')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--grid_seed',type=int,default=123)
    p.add_argument('--image_size',type=int,default=384)
    p.add_argument('--pad',type=int,default=64)
    p.add_argument('--psize',type=int,default=64)
    noise=p.add_mutually_exclusive_group()
    noise.add_argument('--noise_dir')
    noise.add_argument('--noise_covariance')
    noise.add_argument('--noise_complex_var',type=float)
    p.add_argument('--dry_run',action='store_true')
    a=p.parse_args()
    try:
        if min(a.S,a.K,a.cg_iters,a.image_size,a.pad,a.psize)<=0 or not 0<=a.cg_rtol<1:
            raise ValueError('Invalid counts, dimensions or CG tolerance')
        if not math.isfinite(a.beta_scale) or a.beta_scale<=0:
            raise ValueError('beta_scale must be positive and finite')
        if a.pad!=a.psize or a.image_size%a.psize:
            raise ValueError('PaDIS requires pad=psize and image_size divisible by psize')
        if a.mask_select not in (4,7):
            raise ValueError('This runner uses the established R4/R7 datasets')
        sigmas=make_sigmas(a.T,a.sigma_max,a.sigma_min,a.rho)
        shifts=make_grid_plan(a.T,a.K,a.S,a.pad,a.grid_seed)
    except ValueError as e:
        p.error(str(e))
    for key in ('model_path','val_dir','out_root','noise_dir','noise_covariance'):
        v=getattr(a,key)
        if v is not None: setattr(a,key,str(Path(v).expanduser().resolve()))
    return a,sigmas,shifts


def main():
    a,sigmas,shifts=parse_args()
    patch_count=(a.image_size//a.psize+1)**2
    # Required: complete partition in one batch; never silently reduce batch or use checkpoint.
    os.environ.setdefault('PADIS_PATCH_BATCH',str(patch_count))
    os.environ.setdefault('PADIS_PATCH_CHECKPOINT','0')
    if int(os.environ['PADIS_PATCH_BATCH'])!=patch_count or os.environ['PADIS_PATCH_CHECKPOINT']!='0':
        raise ValueError(f'Patch-DMOD requires full partition batch={patch_count}, checkpoint=0')
    totals=dict(annealing_levels=a.T,mm_updates=a.T*a.K,logical_nfe=a.T*a.K*a.S,
                network_backprops=0,cg_solves=a.T*a.K,cg_iters_max=a.cg_iters)
    config={k:v for k,v in vars(a).items() if k not in ('sample_indices','out_root','dry_run')}
    config.update(algorithm='patch_dmod_v1',**totals,patch_batch=patch_count,patch_checkpoint=0,
                  init='zero',convention='padis_real',noise='independent_full_padded_real_unit',
                  partition_policy='uniform_distinct_within_mm',cg_initial='mean_prediction',
                  schedule='karras',sigmas=sigmas,shifts=shifts)
    # Canonicalize tuples to lists so config comparison survives JSON serialization.
    config=json.loads(json.dumps(config))
    if a.dry_run:
        print(json.dumps(config,indent=2));return
    print(json.dumps({**totals,'patch_batch':patch_count,'patch_checkpoint':0,'sigma_max':a.sigma_max,
                      'sigma_min':a.sigma_min,'beta_scale':a.beta_scale}),flush=True)
    import torch
    import dnnlib
    from eval_pula.run import (load_measurement,load_noise,save_image,write_json,read_json,
                               valid_image,valid_metrics,claim_config)
    from eval_pula.score_opa import PatchScore
    from eval_pula.recon_pula import PULASystem,whiten_measurement
    from utils import post_eval_normalize
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for MRI reconstruction')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
    oracle=None;results=[]
    for index in a.sample_indices:
        out=root/f'sample_{index}'/'evaluate';out.mkdir(parents=True,exist_ok=True)
        cfg=out/'config.json';tp=out/'timing.json';rp=out/'recons'/f'recon_patch_{index}.npy';mp=out/'comp_plots'/'results.json'
        expected={**config,'sample_index':index}
        prev=read_json(cfg)
        if prev is None:
            if any(out.iterdir()): raise RuntimeError(f'Cannot verify existing output {out}')
            claim_config(cfg,expected)
        elif prev!=expected:
            raise RuntimeError(f'Configuration mismatch: {out}; choose new TAG')
        timing=read_json(tp)
        complete=valid_image(rp,a.image_size) and isinstance(timing,dict) and all(timing.get(k)==v for k,v in totals.items())
        if not complete:
            if oracle is None:
                print('Loading',a.model_path,flush=True)
                with dnnlib.util.open_url(a.model_path,verbose=False) as f:
                    net=pickle.load(f)['ema'].to(device).eval().requires_grad_(False)
                oracle=PatchScore(net,a.image_size,a.pad,a.psize,device)
            data=torch.load(Path(a.val_dir)/f'sample_{index}.pt',map_location='cpu',weights_only=False)
            covariance,calibration=load_noise(a,index,sample=data)
            raw,y=load_measurement(a.val_dir,index,a.mask_select,device,data=data)
            if y.shape[-2:]!=(a.image_size,a.image_size): raise ValueError('Unexpected FOV')
            # beta_scale multiplies standard deviation, hence covariance by beta_scale squared.
            effective_covariance=covariance*a.beta_scale**2
            aw,yw=whiten_measurement(raw,y,effective_covariance,convention='padis_real')
            generator=torch.Generator(device=device).manual_seed(a.seed)
            system=PULASystem(aw,yw,generator,a.cg_iters,a.cg_rtol,convention='padis_real')
            del data,raw,y
            def report(level,total,sigma,nfe,cg):
                if level==1 or level%5==0 or level==total:
                    print(f'R{a.mask_select} sample {index} | level {level}/{total} sigma={sigma:.6g} NFE={nfe} CG={cg["iterations"]} relres={cg["relative_residual"]}',flush=True)
            net_calls=[0]
            def count_forward(*unused): net_calls[0]+=1
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            started=perf_counter()
            with oracle.net.register_forward_hook(count_forward):
                x,steps=reconstruct_patch_dmod(system,oracle,sigmas,shifts,generator,report=report)
            torch.cuda.synchronize(device);elapsed=perf_counter()-started
            if net_calls[0]!=totals['logical_nfe']:
                raise RuntimeError(f'Actual network forwards {net_calls[0]} != expected {totals["logical_nfe"]}; check full partition batching')
            rels=[r['relative_residual'] for r in steps if r['relative_residual'] is not None]
            timing=dict(**totals,network_forwards=net_calls[0],reconstruction_seconds=elapsed,
                        cg_iterations_total=sum(r['iterations'] for r in steps),
                        cg_relres_max=max(rels,default=0.),
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                        noise_calibration=calibration,beta_scale=a.beta_scale,steps=steps)
            save_image(rp,x.detach().squeeze(1));write_json(tp,timing);mp.unlink(missing_ok=True)
            del x,system,aw,yw
        if not valid_metrics(mp,index):
            post_eval_normalize(recon_dir=str(rp.parent),val_dir=a.val_dir,plot_dir=str(mp.parent),mask_select=a.mask_select)
        if not valid_metrics(mp,index) or not valid_image(rp,a.image_size):
            raise RuntimeError(f'Invalid result {out}')
        m=read_json(mp)['per_image'][str(index)]
        result=dict(sample=index,**{k:float(m[k]) for k in ('psnr','ssim','nrmse')},
                    time=timing['reconstruction_seconds'],peak_allocated_gib=timing['peak_allocated_gib'],
                    peak_reserved_gib=timing['peak_reserved_gib'],**totals)
        results.append(result);print(result,flush=True)
    with (root/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    summary=dict(samples=len(results),**totals)
    for k in ('psnr','ssim','nrmse','time'):
        vals=[r[k] for r in results];summary[k+'_mean']=statistics.mean(vals)
        if k!='time':summary[k+'_std']=statistics.stdev(vals) if len(vals)>1 else None
    for k in ('peak_allocated_gib','peak_reserved_gib'):
        summary[k+'_max']=max(r[k] for r in results)
    write_json(root/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
