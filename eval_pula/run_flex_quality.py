"""Opt-in flexible P/D schedule. Does not modify existing PaDIS runners."""
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
# Preserve original patch import order from eval_pula/run.py.
sys.path[:0] = [str(ROOT), str(ROOT/'eval'), str(ROOT/'train/padis-mri')]
from eval_pula.run import sample_list
from eval_pula.flex_schedule import make_schedule, make_sigmas, counts
from eval_pula.flex_grids import MODES, grid_summary
from eval_pula.flex_quality_grids import quality_grid_plan


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--sample_indices', type=sample_list, default='0,26')
    p.add_argument('--mask_select', type=int, default=4)
    p.add_argument('--num_noise_levels', type=int, default=47)
    p.add_argument('--pattern', help='Default per-level pattern, e.g. PPPPD')
    p.add_argument('--blocks', help='1-based inclusive override ranges, e.g. 1-23:PPPPP;24-47:DDDDD')
    p.add_argument('--schedule_file', help='Text file, one range:pattern per line; # comments')
    p.add_argument('--sigma_schedule', choices=['karras','exponential'], default='karras')
    p.add_argument('--sigma_max', type=float, default=1.)
    p.add_argument('--sigma_min', type=float, default=.003)
    p.add_argument('--rho', type=float, default=7.)
    p.add_argument('--gamma', type=float, default=.5)
    p.add_argument('--dps_zeta', type=float, default=3.)
    p.add_argument('--cg_iters', type=int, default=10)
    p.add_argument('--cg_rtol', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--grid_seed', type=int, default=123)
    p.add_argument('--grid_mode', choices=['cycle_k','fixed','random_step'], default='cycle_k', help='Legacy shared grid mode')
    p.add_argument('--pula_grid_mode', choices=MODES+('sobol',), default=None)
    p.add_argument('--dps_grid_mode', choices=MODES+('sobol',), default=None)
    p.add_argument('--pula_convention', choices=['padis_real','paper_complex'], default='padis_real')
    p.add_argument('--image_size', type=int, default=384)
    p.add_argument('--pad', type=int, default=64)
    p.add_argument('--psize', type=int, default=64)
    noise = p.add_mutually_exclusive_group()
    noise.add_argument('--noise_dir')
    noise.add_argument('--noise_covariance')
    noise.add_argument('--noise_complex_var', type=float)
    p.add_argument('--padding_mode',choices=['reset','persistent'],default='reset')
    p.add_argument('--dps_cov_mode',choices=['off','patch','isotropic'],default='off')
    p.add_argument('--cov_cache')
    p.add_argument('--cov_scale',type=float,default=1.)
    p.add_argument('--cov_cg_iters',type=int,default=16)
    p.add_argument('--cov_cg_rtol',type=float,default=1e-3)
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    try:
        if not all(math.isfinite(v) for v in (a.sigma_max,a.sigma_min,a.rho,a.gamma,a.dps_zeta,a.cg_rtol)):
            raise ValueError('Nonfinite numeric argument')
        if min(a.gamma,a.dps_zeta,a.cg_iters,a.image_size,a.pad,a.psize) <= 0 or not 0 <= a.cg_rtol < 1:
            raise ValueError('Invalid step or size argument')
        if a.pad != a.psize or a.image_size % a.psize:
            raise ValueError('PaDIS requires pad=psize and image_size divisible by psize')
        if a.dps_cov_mode!='off' and not a.cov_cache:
            raise ValueError('Covariance DPS requires --cov_cache')
        if a.dps_cov_mode=='off' and a.cov_cache:
            raise ValueError('Set covariance mode when supplying cache')
        if not math.isfinite(a.cov_scale) or a.cov_scale<0 or a.cov_cg_iters<1 or not 0<a.cov_cg_rtol<1:
            raise ValueError('Invalid covariance CG configuration')
        if not a.sample_indices:
            raise ValueError('No sample indices')
        pattern = a.pattern if (a.pattern is not None or a.blocks or a.schedule_file) else 'PPPPD'
        rows = make_schedule(a.num_noise_levels,pattern,a.blocks,a.schedule_file)
        sigmas = make_sigmas(a.num_noise_levels,a.sigma_max,a.sigma_min,a.sigma_schedule,a.rho)
        if max(map(len,rows)) > a.pad**2:
            raise ValueError('Too many partition slots at one level')
    except ValueError as e:
        p.error(str(e))
    for key in ('model_path','val_dir','out_root','noise_dir','noise_covariance','cov_cache'):
        v=getattr(a,key)
        if v is not None: setattr(a,key,str(Path(v).expanduser().resolve()))
    return a,rows,sigmas


def main():
    a,rows,sigmas = parse_args()
    totals=counts(rows)
    totals['pula_cg_solves']=totals['cg_solves']
    totals['covariance_cg_solves']=totals['dps_updates'] if a.dps_cov_mode!='off' else 0
    totals['cg_solves']+=totals['covariance_cg_solves']
    plan=[dict(level=i+1,sigma=s,pattern=r) for i,(s,r) in enumerate(zip(sigmas,rows))]
    shifts=quality_grid_plan(rows,a.pad,a.grid_seed,a.grid_mode,a.pula_grid_mode,a.dps_grid_mode)
    grids=grid_summary(rows,shifts)
    for entry,grid in zip(plan,shifts):
        entry['shifts']=[list(s) for s in grid]
    diagnostics=dict(grid_summary=grids,padding_mode=a.padding_mode,dps_cov_mode=a.dps_cov_mode,**totals)
    if a.dry_run:
        print(json.dumps(dict(plan=plan,**diagnostics),indent=2),flush=True)
        return
    print(json.dumps(diagnostics),flush=True)
    import torch
    import dnnlib
    from eval_pula.run import (load_measurement,load_noise,save_image,write_json,read_json,
                               valid_image,valid_metrics,claim_config)
    from eval_pula.score_opa import PatchScore
    from eval_pula.recon_pula import PULASystem,whiten_measurement
    from eval_pula.flex_quality_updates import pula_initialize,QualityDispatcher
    from eval_pula.flex_quality_cov import PatchCovariance,sha256_file
    from utils import post_eval_normalize
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for MRI reconstruction')
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    config={k:v for k,v in vars(a).items() if k not in ('sample_indices','out_root','dry_run','pattern','blocks','schedule_file')}
    config.update(sampler='flex_pd_quality_v1',plan=plan,grid_summary=grids,shifts=[[list(s) for s in row] for row in shifts],
                  patch_batch=int(os.environ.get('PADIS_PATCH_BATCH',49)),
                  patch_checkpoint=os.environ.get('PADIS_PATCH_CHECKPOINT','0'),
                  initialization='pula_cg_all_patterns',bridge=a.padding_mode,
                  rng_policy='one_legacy_private_generator_per_sample',pula_variant='baseline',**totals)
    cov_model=None
    if a.dps_cov_mode!='off':
        config['cov_cache_sha256']=sha256_file(a.cov_cache)
        cov_model=PatchCovariance(a.cov_cache,a.model_path,a.image_size,a.pad,a.psize,device,a.dps_cov_mode,a.cov_scale)
        for sigma,row in zip(sigmas,rows):
            if 'D' in row:
                cov_model.map(sigma,(0,0)) # validate coverage before running
    root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
    oracle=None; all_rows=[]
    for index in a.sample_indices:
        out=root/a.grid_mode/f'sample_{index}'/'evaluate';out.mkdir(parents=True,exist_ok=True)
        cfg=out/'config.json'; tp=out/'timing.json'; rp=out/'recons'/f'recon_patch_{index}.npy'; mp=out/'comp_plots'/'results.json'
        expected={**config,'sample_index':index}
        previous=read_json(cfg)
        if previous is None:
            if any(out.iterdir()): raise RuntimeError(f'Cannot verify existing output {out}')
            claim_config(cfg,expected)
        elif previous!=expected:
            raise RuntimeError(f'Configuration mismatch: {out}; choose a new output path')
        timing=read_json(tp)
        complete=valid_image(rp,a.image_size) and isinstance(timing,dict) and all(timing.get(k)==v for k,v in totals.items())
        if not complete:
            if oracle is None:
                print('Loading',a.model_path,flush=True)
                with dnnlib.util.open_url(a.model_path,verbose=False) as f:
                    net=pickle.load(f)['ema'].to(device).eval()
                oracle=PatchScore(net,a.image_size,a.pad,a.psize,device)
            data=torch.load(Path(a.val_dir)/f'sample_{index}.pt',map_location='cpu',weights_only=False)
            covariance,calibration=load_noise(a,index,sample=data)
            raw,y=load_measurement(a.val_dir,index,a.mask_select,device,data=data)
            if y.shape[-2:]!=(a.image_size,a.image_size): raise ValueError('Unexpected FOV')
            aw,yw=whiten_measurement(raw,y,covariance,convention=a.pula_convention)
            generator=torch.Generator(device=device).manual_seed(a.seed)
            system=PULASystem(aw,yw,generator,a.cg_iters,a.cg_rtol,convention=a.pula_convention)
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            started=perf_counter(); state,init=pula_initialize(system,a.sigma_max)
            torch.cuda.synchronize(device); init_time=perf_counter()-started
            if cov_model is not None and not isinstance(covariance,(float,int)):
                raise ValueError('Covariance DPS currently requires scalar calibrated raw noise covariance')
            dispatcher=QualityDispatcher(state,system,oracle,raw,y,a.padding_mode,cov_model,covariance,a.cov_cg_iters,a.cov_cg_rtol)
            steps=[]; p_seconds=d_seconds=0.; net_calls=[0]
            def count_forward(*unused): net_calls[0]+=1
            # CUDA events measure each kernel without an extra synchronization per update.
            events=[]
            with oracle.net.register_forward_hook(count_forward):
                for i,(sigma,row) in enumerate(zip(sigmas,rows)):
                    for j,method in enumerate(row):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        info=dispatcher.step(method,sigma,shifts[i][j],a.gamma,a.dps_zeta,i==len(rows)-1)
                        end.record();events.append((method,begin,end))
                        steps.append(dict(level=i+1,slot=j+1,method=method,sigma=sigma,shift=list(shifts[i][j]),**info))
                    if i==0 or (i+1)%5==0 or i==len(rows)-1 or row!=rows[i-1]:
                        print(f'R{a.mask_select} sample {index} | level {i+1}/{len(rows)} sigma={sigma:.6g} {row}',flush=True)
            torch.cuda.synchronize(device);elapsed=perf_counter()-started
            for method,begin,end in events:
                if method=='P': p_seconds+=begin.elapsed_time(end)/1000
                else: d_seconds+=begin.elapsed_time(end)/1000
            timing=dict(**totals,langevin_updates=totals['pula_updates'],network_forwards=net_calls[0],
                        reconstruction_seconds=elapsed,initialization_seconds=init_time,
                        pula_step_seconds=p_seconds,dps_step_seconds=d_seconds,initialization_cg=init,
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                        noise_calibration=calibration,grid_summary=grids,steps=steps)
            if cov_model is not None:
                rels=[v['cov_cg_relres'] for v in steps if 'cov_cg_relres' in v]
                timing.update(cov_cg_relres_max=max(rels,default=0.),cov_cg_relres_mean=statistics.mean(rels) if rels else 0.,
                              cov_cg_not_converged=sum(v>a.cov_cg_rtol for v in rels))
                print('Covariance CG:',{k:timing[k] for k in ('cov_cg_relres_max','cov_cg_relres_mean','cov_cg_not_converged')},flush=True)
            save_image(rp,dispatcher.state.detach().squeeze(1));write_json(tp,timing)
            # Stale metrics must not be reused if a new reconstruction was produced.
            mp.unlink(missing_ok=True)
        if not valid_metrics(mp,index):
            post_eval_normalize(recon_dir=str(rp.parent),val_dir=a.val_dir,plot_dir=str(mp.parent),mask_select=a.mask_select)
        if not valid_metrics(mp,index) or not valid_image(rp,a.image_size): raise RuntimeError(f'Invalid result {out}')
        m=read_json(mp)['per_image'][str(index)]
        result=dict(sample=index,**{k:float(m[k]) for k in ('psnr','ssim','nrmse')},time=timing['reconstruction_seconds'],**totals)
        all_rows.append(result);print(result,flush=True)
    with (root/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    summary=dict(samples=len(all_rows),**totals)
    for k in ('psnr','ssim','nrmse','time'):
        vals=[r[k] for r in all_rows];summary[k+'_mean']=statistics.mean(vals)
        if k!='time':summary[k+'_std']=statistics.stdev(vals) if len(vals)>1 else None
    write_json(root/'summary.json',summary); print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
