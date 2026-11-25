"""Standalone entry using the existing PaDIS training loop and original OPA loss."""
import argparse
import json
import re
from pathlib import Path
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop


def int_list(value):
    try:
        result = [int(v) for v in value.split(',')]
        if not result or any(v < 1 for v in result):
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('Expected comma-separated positive integers')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--kimg',type=int,default=500)
    p.add_argument('--batch-gpu',type=int,default=1)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--cbase', type=int, default=64)
    p.add_argument('--cres', type=int_list, default=[1,2,4,4])
    p.add_argument('--num-blocks', type=int, default=2)
    p.add_argument('--attn-resolutions', type=int_list, default=[8])
    p.add_argument('--opa-lambda', type=float, default=1.25)
    p.add_argument('--noise-embedding', choices=['fourier','ddim'], default='fourier')
    p.add_argument('--init-mode', choices=['edm','legacy'], default='edm')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-rampup-kimg', type=float, default=10000*32/121)
    p.add_argument('--ema-kimg', type=float, default=500)
    p.add_argument('--tick', type=int, default=5)
    p.add_argument('--snap', type=int, default=20)
    p.add_argument('--dump', type=int, default=20)
    a=p.parse_args()
    if not 0 < a.lr < float('inf') or not 0 <= a.lr_rampup_kimg < float('inf') or not 0 <= a.ema_kimg < float('inf'):
        p.error('Invalid LR / warmup / EMA')
    if min(a.tick,a.snap,a.dump)<1: p.error('tick/snap/dump must be positive')
    if a.cbase < 32 or a.cbase % 32: p.error('cbase must be a positive multiple of 32')
    if len(a.cres) > 7 or a.num_blocks < 1: p.error('Invalid UNet depth or num-blocks')
    if not set(a.attn_resolutions).issubset({64//2**i for i in range(len(a.cres))}): p.error('Invalid attention resolution for selected depth')
    if not 0 <= a.opa_lambda < float('inf'): p.error('opa-lambda must be finite and nonnegative')
    if a.kimg < 1 or a.batch_gpu not in (1,2,4): p.error('kimg > 0 and batch-gpu in 1,2,4 required')
    if not (Path(a.data)/'noisy.pt').is_file(): p.error('Missing noisy.pt')
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_world_size()!=1: raise RuntimeError('This entry expects one GPU')
    parent=Path(a.outdir); parent.mkdir(parents=True,exist_ok=True)
    ids=[int(m.group()) for x in parent.iterdir() if x.is_dir() and (m:=re.match(r'^\d+',x.name))]
    cres_tag='-'.join(map(str,a.cres))
    attn_tag='-'.join(map(str,a.attn_resolutions))
    lam_tag=format(a.opa_lambda,'g').replace('.','p')
    run=parent/f'{max(ids,default=-1)+1:05d}-CT-DDIM-C{a.cbase}-M{cres_tag}-B{a.num_blocks}-A{attn_tag}-EDM-OPA{lam_tag}-p64-{a.noise_embedding}-{a.init_mode}' 
    run.mkdir()
    cfg=dict(
        run_dir=str(run),
        dataset_kwargs=dict(class_name='training.dataset.ImageFolderDatasetX',path=a.data,use_labels=False,xflip=False,cache=True),
        data_loader_kwargs=dict(pin_memory=True,num_workers=4,prefetch_factor=2),
        network_kwargs=dict(class_name='training.ct_paper_opa.CTPaperEDM',sigma_data=0.5,dropout=0.05,use_fp16=False,model_channels=a.cbase,channel_mult=a.cres,num_res_blocks=a.num_blocks,attn_resolutions=a.attn_resolutions,noise_embedding=a.noise_embedding,init_mode=a.init_mode),
        loss_kwargs=dict(class_name='training.patch_overlap_independent_loss.IndependentNoiseOverlapPatch_EDMLoss',P_mean=-1.2,P_std=1.2,sigma_data=0.5,lambda_overlap=a.opa_lambda,active_patch_size=64,noise_mode='independent',alignment_mode='symmetric',pair_geometry='identical',spatial_weight_mode='uniform',joint_context=True,global_conditioning=False),
        optimizer_kwargs=dict(class_name='torch.optim.Adam',lr=a.lr,betas=[0.9,0.999],eps=1e-8),
        seed=a.seed,batch_size=4,batch_gpu=a.batch_gpu,total_kimg=a.kimg,ema_halflife_kimg=a.ema_kimg,lr_rampup_kimg=a.lr_rampup_kimg,
        kimg_per_tick=a.tick,snapshot_ticks=a.snap,state_dump_ticks=a.dump,
        real_p=1.0,train_on_latents=False,progressive=False,padding=True,pad_width=64,
        four_channels=1,hash_channels=1,patch_list=[64],patch_probs=[1.0],
    )
    dnnlib.util.Logger(file_name=str(run/'log.txt'),file_mode='a',should_flush=True)
    print(json.dumps(cfg,indent=2),flush=True)
    (run/'training_options.json').write_text(json.dumps(cfg,indent=2))
    training_loop.training_loop(**cfg)

if __name__=='__main__': main()
