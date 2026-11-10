"""Independent OPA+AGGA entry point. Dryrun uses only the Python standard library."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

REFERENCE_SHA = '2e5f0b993355e6e9de1196dc7378660a30103cf4'
ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['dryrun', 'smoke', 'train'], default='train')
    p.add_argument('--data', required=True, help='Directory containing noisy.pt')
    p.add_argument('--outdir', default=str(ROOT / 'PaDIS-MRI-runs/opa_agga_training-runs'))
    p.add_argument('--gpu', default='0', help='Visible GPU for this single-process entry')
    p.add_argument('--total-kimg', type=int, help='Absolute training target, not additional kimg')
    p.add_argument('--opa-weight', type=float, default=1.0)
    p.add_argument('--rho', type=float, default=0.65)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--score-mode', choices=['multiply', 'add'], default='multiply')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--workers', type=int, default=4)
    restart = p.add_mutually_exclusive_group()
    restart.add_argument('--resume', help='Local training-state-NNNNNN.pt; matching snapshot required')
    restart.add_argument('--transfer', help='Local network pickle; resets training counter and optimizer')
    return p


def build_config(opts):
    if not (math.isfinite(opts.opa_weight) and opts.opa_weight >= 0
            and math.isfinite(opts.rho) and 0 <= opts.rho < 1
            and math.isfinite(opts.gamma) and opts.gamma >= 0):
        raise ValueError('Require finite opa_weight/gamma >= 0 and 0 <= rho < 1')
    total = opts.total_kimg if opts.total_kimg is not None else (1 if opts.mode == 'smoke' else 5000)
    if total < 1 or opts.workers < 1 or not 0 <= opts.seed < 2**31:
        raise ValueError('Require total-kimg/workers >= 1 and 0 <= seed < 2**31')
    c = dict(
        dataset_kwargs=dict(class_name='training.dataset.ImageFolderDatasetX',
                            path=str(Path(opts.data).expanduser().resolve()),
                            use_labels=False, xflip=False, cache=True),
        data_loader_kwargs=dict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2),
        network_kwargs=dict(class_name='training.networks.Patch_EDMPrecond',
                            model_type='SongUNet', embedding_type='positional',
                            encoder_type='standard', decoder_type='standard',
                            channel_mult_noise=1, resample_filter=[1, 1],
                            model_channels=128, channel_mult=[2, 2, 2], hash_channels=1,
                            sigma_data=0.5, dropout=0.05, use_fp16=False),
        loss_kwargs=dict(class_name='training.patch_opa_agga_loss.OPAAGGAPatchEDMLoss',
                         P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                         opa_weight=opts.opa_weight, rho=opts.rho, gamma=opts.gamma,
                         score_mode=opts.score_mode),
        optimizer_kwargs=dict(class_name='torch.optim.Adam', lr=1e-4,
                              betas=[0.9, 0.999], eps=1e-8),
        batch_size=4, batch_gpu=1, seed=opts.seed, total_kimg=total,
        ema_halflife_kimg=500, loss_scaling=1, cudnn_benchmark=True,
        kimg_per_tick=1 if opts.mode == 'smoke' else 5,
        snapshot_ticks=1 if opts.mode == 'smoke' else 200,
        state_dump_ticks=1 if opts.mode == 'smoke' else 1000,
        real_p=0.5, train_on_latents=False, progressive=False,
        padding=True, four_channels=1, hash_channels=1, pad_width=64,
        patch_list=[64] if opts.mode == 'smoke' else [16, 32, 64],
        patch_probs=[1.0] if opts.mode == 'smoke' else [0.2, 0.3, 0.5],
    )
    if opts.transfer:
        c.update(resume_pkl=str(Path(opts.transfer).expanduser().resolve()), ema_rampup_ratio=None)
    if opts.resume:
        state = Path(opts.resume).expanduser().resolve()
        match = re.fullmatch(r'training-state-(\d+)\.pt', state.name)
        if not match:
            raise ValueError('--resume must be training-state-NNNNNN.pt')
        progress = int(match.group(1))
        if total <= progress:
            raise ValueError('--total-kimg must exceed the resume counter (absolute target)')
        c.update(resume_state_dump=str(state), resume_kimg=progress,
                 resume_pkl=str(state.with_name(f'network-snapshot-{match.group(1)}.pkl')))
    return c


def main(argv=None):
    p = parser()
    opts = p.parse_args(argv)
    try:
        c = build_config(opts)
        # Dryrun validates configuration only; unavailable server paths are printed without opening them.
        if opts.mode != 'dryrun':
            required = [Path(c['dataset_kwargs']['path']) / 'noisy.pt']
            required += [Path(c[k]) for k in ('resume_pkl', 'resume_state_dump') if k in c]
            for path in required:
                if not path.is_file():
                    raise ValueError(f'Required input does not exist: {path}')
    except ValueError as error:
        p.error(str(error))
    tag = (f'opa{opts.opa_weight:g}_rho{opts.rho:g}_gamma{opts.gamma:g}_'
           f'{opts.score_mode}_seed{opts.seed}_{opts.mode}')
    metadata = dict(reference_sha=REFERENCE_SHA, mode=opts.mode, gpu=opts.gpu,
                    experiment_name=tag,
                    resume=opts.resume, transfer=opts.transfer,
                    augment=0, cli=vars(opts))
    print(json.dumps(dict(**c, opa_agga=metadata), indent=2), flush=True)
    print(f"Target: {c['total_kimg']} kimg = {c['total_kimg'] / 1000:g} Mimg (absolute counter)", flush=True)
    if opts.mode == 'dryrun':
        print('Dryrun complete; no training or output files created.')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    sys.path.insert(0, str(ROOT))
    import torch
    import dnnlib
    from torch_utils import distributed as dist
    from training import training_loop

    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()
    outdir = Path(opts.outdir).expanduser().resolve()
    if dist.get_rank() == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        ids = [int(m.group(1)) for d in outdir.iterdir()
               if d.is_dir() and (m := re.match(r'^(\d+)-', d.name))]
        run_dir = outdir / f'{max(ids, default=-1) + 1:05d}-{tag}'
        run_dir.mkdir()  # Never reuse an existing experiment directory.
        c['run_dir'] = str(run_dir)
        dnnlib.util.Logger(file_name=str(run_dir / 'log.txt'), file_mode='a', should_flush=True)
        print(json.dumps(dict(**c, opa_agga=metadata), indent=2))
        with (run_dir / 'training_options.json').open('w') as stream:
            json.dump(dict(**c, opa_agga=metadata), stream, indent=2)
    else:
        c['run_dir'] = None
    training_loop.training_loop(**c)


if __name__ == '__main__':
    main()
