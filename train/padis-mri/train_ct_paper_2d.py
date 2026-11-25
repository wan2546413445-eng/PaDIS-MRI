"""Standalone training entry. Copy source files, no installer or monkey patch."""
import argparse
import copy
import json
import os
import pickle
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from training.ct_paper_2d import CTPaperPrior, vp_training_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--kimg', type=int, default=1000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--batch-gpu', type=int, default=4)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--snap-kimg', type=int, default=100)
    a = p.parse_args()
    if a.batch <= 0 or a.batch_gpu <= 0 or a.batch % a.batch_gpu or a.kimg <= 0 or a.snap_kimg <= 0:
        p.error('Positive batch, batch-gpu, kimg and snap-kimg required; batch divisible by batch-gpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for training')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    path = Path(a.data)
    if path.is_dir(): path = path / 'noisy.pt'
    raw = torch.load(path, map_location='cpu')['x_est_gt']
    if not torch.is_complex(raw) or raw.ndim != 3 or raw.shape[-2:] != (384,384):
        raise ValueError(f'Expected PaDIS complex [N,384,384], got {raw.shape} {raw.dtype}')
    images = F.pad(torch.view_as_real(raw).permute(0,3,1,2).contiguous().float(), (64,64,64,64))
    del raw
    parent = Path(a.outdir); parent.mkdir(parents=True, exist_ok=True)
    ids = [int(q.name.split('-')[0]) for q in parent.iterdir() if q.is_dir() and q.name.split('-')[0].isdigit()]
    out = parent / f'{max(ids,default=-1)+1:05d}-ct-paper-vp64'
    out.mkdir()
    with (out/'training_options.json').open('w') as f: json.dump(vars(a), f, indent=2)
    net = CTPaperPrior(dropout=a.dropout).cuda().train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print('Output:', out, '\nDataset:', tuple(images.shape), '\nParameters:', sum(x.numel() for x in net.parameters()), flush=True)
    print('VP: 1000 timesteps, beta=1e-4..0.02; 64 fixed; channels=6->2; OPA=off', flush=True)
    def snapshot(n):
        name = out / f'network-snapshot-{n//1000:06d}.pkl'
        tmp = name.with_suffix('.tmp')
        cpu = copy.deepcopy(ema).cpu()
        with tmp.open('wb') as f: pickle.dump({'ema':cpu}, f)
        os.replace(tmp,name)
        del cpu
        print('[Snapshot]',name,flush=True)
    snapshot(0)
    n=0; tick=5000; next_snap=a.snap_kimg*1000; begin=time.time(); recent=[]
    # Count actual sampled patches (one per full image), not optimizer updates.
    while n < a.kimg*1000:
        opt.zero_grad(set_to_none=True)
        total=0.
        current=min(a.batch, a.kimg*1000-n)
        for start in range(0,current,a.batch_gpu):
            count=min(a.batch_gpu,current-start)
            ids=torch.randint(len(images),(count,))
            loss=vp_training_loss(net,images[ids].cuda())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite VP loss')
            (loss*count/current).backward()
            total += loss.detach().item()*count/current
        opt.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.lerp_(pn,1-a.ema)
        n += current; recent.append(total)
        if n>=tick or n>=a.kimg*1000:
            print(f'kimg {n/1000:.3f} loss {np.mean(recent):.6f} time {time.time()-begin:.1f}s peak_GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
            recent=[]; tick += 5000; torch.cuda.reset_peak_memory_stats()
        if n>=next_snap or n>=a.kimg*1000:
            snapshot(n)
            next_snap += a.snap_kimg*1000
    print('Training complete.',flush=True)

if __name__ == '__main__': main()
