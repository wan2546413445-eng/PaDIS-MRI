# ---------------------------------------------------------------
# Modified from PaDIS-MRI training entry.
# New standalone entry for detail residual prior enhancement.
# ---------------------------------------------------------------

"""Train PaDIS-MRI with a multi-scale gated detail residual head."""

import os
import sys
import re
import json
import click
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import dnnlib

from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')


# ----------------------------------------------------------------------------

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_float_list(s):
    if isinstance(s, list):
        return s
    return [float(p) for p in s.split(',')]


def parse_bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in ['1', 'true', 'yes', 'y']:
        return True
    if x in ['0', 'false', 'no', 'n']:
        return False
    raise click.BadParameter(f'Cannot parse boolean value: {x}')


# ----------------------------------------------------------------------------

@click.command()
# Patch options.
@click.option('--real_p', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)
@click.option('--train_on_latents', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--progressive', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--padding', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--four_channels', metavar='INT', type=int, default=1, show_default=True)
@click.option('--hash_channels', metavar='INT', type=int, default=1, show_default=True)
@click.option('--pad_width', metavar='INT', type=int, required=True)
@click.option('--patch-list', type=str)
@click.option('--patch-probs', type=str)
# Main options.
@click.option('--outdir', metavar='DIR', type=str, required=True)
@click.option('--data', metavar='ZIP|DIR', type=str, required=True)
@click.option('--cond', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--arch', type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond', type=click.Choice(['vp', 've', 'edm', 'pedm']), default='pedm', show_default=True)
# Hyperparameters.
@click.option('--duration', metavar='MIMG', type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu', metavar='INT', type=click.IntRange(min=1))
@click.option('--cbase', metavar='INT', type=int)
@click.option('--cres', metavar='LIST', type=parse_int_list)
@click.option('--lr', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema', metavar='MIMG', type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--implicit_mlp', metavar='BOOL', type=bool, default=False, show_default=True)
# Performance.
@click.option('--fp16', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--ls', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--cache', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--workers', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
# I/O.
@click.option('--desc', metavar='STR', type=str)
@click.option('--nosubdir', is_flag=True)
@click.option('--tick', metavar='KIMG', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump', metavar='TICKS', type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed', metavar='INT', type=int)
@click.option('--transfer', metavar='PKL|URL', type=str)
@click.option('--resume', metavar='PT', type=str)
# Detail residual options.
@click.option('--detail-hidden', type=int, default=48, show_default=True)
@click.option('--detail-eta', type=float, default=0.15, show_default=True)
@click.option('--detail-dilations', type=str, default='1,2,5', show_default=True)
@click.option('--detail-use-pos', type=parse_bool, default=True, show_default=True)
@click.option('--detail-gate-bias', type=float, default=-1.0, show_default=True)
@click.option('--detail-init-scale', type=float, default=1e-3, show_default=True)
@click.option('--detail-detach-base', type=parse_bool, default=True, show_default=True)
@click.option('--lambda-residual', type=float, default=0.2, show_default=True)
@click.option('--lambda-gradient', type=float, default=0.1, show_default=True)
@click.option('--lambda-edge', type=float, default=0.1, show_default=True)
@click.option('--edge-alpha', type=float, default=2.0, show_default=True)
@click.option('--detail-use-sigma-weight', type=parse_bool, default=True, show_default=True)
@click.option('-n', '--dry-run', is_flag=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDatasetX', path=opts.data,
        use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache,
    )
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)
    c.real_p = opts.real_p
    c.train_on_latents = opts.train_on_latents
    c.progressive = opts.progressive
    c.padding = opts.padding
    c.four_channels = opts.four_channels
    c.hash_channels = opts.hash_channels
    c.pad_width = opts.pad_width

    if opts.patch_list is not None:
        c.patch_list = parse_int_list(opts.patch_list)
    if opts.patch_probs is not None:
        c.patch_probs = parse_float_list(opts.patch_probs)

    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1, 1], model_channels=128, channel_mult=[2, 2, 2], hash_channels=c.hash_channels)
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=128, channel_mult=[2, 2, 2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1, 2, 3, 4])

    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'pedm':
        c.network_kwargs.class_name = 'training.networks_detail_residual.Patch_EDMPrecond_DetailResidual'
        c.network_kwargs.sigma_data = 0.5
        c.network_kwargs.update(
            detail_hidden=opts.detail_hidden,
            detail_eta=opts.detail_eta,
            detail_dilations=parse_int_list(opts.detail_dilations),
            detail_use_pos=opts.detail_use_pos,
            detail_gate_bias=opts.detail_gate_bias,
            detail_init_scale=opts.detail_init_scale,
            detail_detach_base=opts.detail_detach_base,
        )
        c.loss_kwargs.class_name = 'training.patch_loss_detail_residual.DetailResidualPatch_EDMLoss'
        c.loss_kwargs.update(
            P_mean=-1.2, P_std=1.2, sigma_data=0.5,
            lambda_residual=opts.lambda_residual,
            lambda_gradient=opts.lambda_gradient,
            lambda_edge=opts.lambda_edge,
            edge_alpha=opts.edge_alpha,
            detail_sigma_weight=opts.detail_use_sigma_weight,
        )
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    if opts.implicit_mlp:
        c.network_kwargs.implicit_mlp = True
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    dataset_name = 'aapm_3'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'

    eta_tag = str(opts.detail_eta).replace('.', 'p')
    res_tag = str(opts.lambda_residual).replace('.', 'p')
    grad_tag = str(opts.lambda_gradient).replace('.', 'p')
    edge_tag = str(opts.lambda_edge).replace('.', 'p')
    dil_tag = 'p'.join(str(x) for x in parse_int_list(opts.detail_dilations))
    desc += f'-detail-h{opts.detail_hidden}-eta{eta_tag}-res{res_tag}-grad{grad_tag}-edge{edge_tag}-dil{dil_tag}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0('Detail residual head:    enabled')
    dist.print0(f'Detail hidden:           {opts.detail_hidden}')
    dist.print0(f'Detail eta:              {opts.detail_eta}')
    dist.print0(f'Detail dilations:        {parse_int_list(opts.detail_dilations)}')
    dist.print0(f'Lambda residual/grad/edge: {opts.lambda_residual}/{opts.lambda_gradient}/{opts.lambda_edge}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    training_loop.training_loop(**c)


if __name__ == '__main__':
    main()
