"""Train the isolated PaDIS-MRI low-noise magnitude-SSIM experiment."""

import json
import os
import re
import sys

import click
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings

warnings.filterwarnings(
    'ignore',
    'Grad strides do not match bucket view strides',
)


BASELINE = {
    'precond': 'pedm',
    'batch': 2,
    'lr': 1e-4,
    'dropout': 0.05,
    'augment': 0.0,
    'real_p': 0.5,
    'padding': True,
    'pad_width': 96,
    'patch_list': [16, 32, 64],
    'patch_probs': [0.2, 0.3, 0.5],
    'seed': 123,
    'workers': 4,
    'fp16': False,
    'batch_gpu': None,
}


def parse_int_list(value):
    if isinstance(value, list):
        return value
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for part in value.split(','):
        match = range_re.match(part)
        if match:
            ranges.extend(
                range(int(match.group(1)), int(match.group(2)) + 1)
            )
        else:
            ranges.append(int(part))
    return ranges


def parse_float_list(value):
    if isinstance(value, list):
        return value
    return [float(part) for part in value.split(',')]


def _require_fixed(name, actual, expected):
    """Reject accidental drift from the stage-1 baseline configuration."""
    if actual != expected:
        raise click.ClickException(
            f'{name} is fixed to {expected!r} for SSIM stage 1; '
            f'received {actual!r}'
        )


@click.command()
@click.option('--outdir', type=str, required=True)
@click.option('--data', type=str, required=True)
@click.option(
    '--duration',
    type=click.FloatRange(min=0, min_open=True),
    default=0.001,
    show_default=True,
    help='Training duration in millions of images.',
)
@click.option('--desc', type=str)
@click.option('--nosubdir', is_flag=True)
@click.option('--tick', type=click.IntRange(min=1), default=1)
@click.option(
    '--snap',
    type=click.IntRange(min=0),
    default=50,
    help='Snapshot interval in ticks; 0 disables snapshots.',
)
@click.option(
    '--dump',
    type=click.IntRange(min=0),
    default=500,
    help='Training-state interval in ticks; 0 disables state dumps.',
)
@click.option('--batch-gpu', type=click.IntRange(min=1))
@click.option('--precond', type=click.Choice(['pedm']), default='pedm')
@click.option('--batch', type=click.IntRange(min=1), default=2)
@click.option('--lr', type=click.FloatRange(min=0, min_open=True), default=1e-4)
@click.option(
    '--dropout',
    type=click.FloatRange(min=0, max=1),
    default=0.05,
)
@click.option(
    '--augment',
    type=click.FloatRange(min=0, max=1),
    default=0.0,
)
@click.option(
    '--real_p',
    type=click.FloatRange(min=0, max=1),
    default=0.5,
)
@click.option('--padding', type=bool, default=True)
@click.option('--pad_width', type=int, default=96)
@click.option('--patch-list', type=str, default='16,32,64')
@click.option('--patch-probs', type=str, default='0.2,0.3,0.5')
@click.option('--seed', type=int, default=123)
@click.option('--workers', type=click.IntRange(min=1), default=4)
@click.option('--fp16', type=bool, default=False)
@click.option(
    '--ssim-weight',
    type=click.FloatRange(min=0),
    default=0.50,
    show_default=True,
)
@click.option(
    '--ssim-sigma-max',
    type=click.FloatRange(min=0, min_open=True),
    default=0.50,
    show_default=True,
)
@click.option(
    '--ssim-window-size',
    type=click.IntRange(min=1, max=16),
    default=11,
    show_default=True,
)
@click.option(
    '--ssim-window-sigma',
    type=click.FloatRange(min=0, min_open=True),
    default=1.5,
    show_default=True,
)
@click.option(
    '--ssim-k1',
    type=click.FloatRange(min=0, min_open=True),
    default=0.01,
    show_default=True,
)
@click.option(
    '--ssim-k2',
    type=click.FloatRange(min=0, min_open=True),
    default=0.03,
    show_default=True,
)
@click.option(
    '--ssim-data-range',
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    show_default=True,
)
@click.option(
    '--magnitude-eps',
    type=click.FloatRange(min=0, min_open=True),
    default=1e-8,
    show_default=True,
)
@click.option('-n', '--dry-run', is_flag=True)
def main(**kwargs):
    """Run Baseline + low-noise magnitude SSIM without AGG/Overlap/LGFC."""
    opts = dnnlib.EasyDict(kwargs)
    patch_list = parse_int_list(opts.patch_list)
    patch_probs = parse_float_list(opts.patch_probs)

    # 所有非 SSIM 变量均锁定为原始 PaDIS-MRI Baseline。
    _require_fixed('precond', opts.precond, BASELINE['precond'])
    _require_fixed('batch', opts.batch, BASELINE['batch'])
    _require_fixed('lr', opts.lr, BASELINE['lr'])
    _require_fixed('dropout', opts.dropout, BASELINE['dropout'])
    _require_fixed('augment', opts.augment, BASELINE['augment'])
    _require_fixed('real_p', opts.real_p, BASELINE['real_p'])
    _require_fixed('padding', opts.padding, BASELINE['padding'])
    _require_fixed('pad_width', opts.pad_width, BASELINE['pad_width'])
    _require_fixed('patch_list', patch_list, BASELINE['patch_list'])
    _require_fixed('patch_probs', patch_probs, BASELINE['patch_probs'])
    _require_fixed('seed', opts.seed, BASELINE['seed'])
    _require_fixed('workers', opts.workers, BASELINE['workers'])
    _require_fixed('fp16', opts.fp16, BASELINE['fp16'])
    _require_fixed('batch_gpu', opts.batch_gpu, BASELINE['batch_gpu'])

    if opts.ssim_window_size % 2 == 0:
        raise click.ClickException('--ssim-window-size must be odd')

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    config = dnnlib.EasyDict()
    config.dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDatasetX',
        path=opts.data,
        use_labels=False,
        xflip=False,
        cache=True,
    )
    config.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True,
        num_workers=opts.workers,
        prefetch_factor=2,
    )
    config.network_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Patch_EDMPrecond',
        model_type='SongUNet',
        embedding_type='positional',
        encoder_type='standard',
        decoder_type='standard',
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        hash_channels=1,
        dropout=opts.dropout,
        use_fp16=opts.fp16,
    )
    config.loss_kwargs = dnnlib.EasyDict(
        class_name=(
            'training.patch_ssim_loss.'
            'MagnitudeSSIMPatchEDMLoss'
        ),
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        ssim_weight=opts.ssim_weight,
        ssim_sigma_max=opts.ssim_sigma_max,
        ssim_window_size=opts.ssim_window_size,
        ssim_window_sigma=opts.ssim_window_sigma,
        ssim_k1=opts.ssim_k1,
        ssim_k2=opts.ssim_k2,
        ssim_data_range=opts.ssim_data_range,
        magnitude_eps=opts.magnitude_eps,
    )
    config.optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=opts.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    config.real_p = opts.real_p
    config.train_on_latents = False
    config.progressive = False
    config.padding = opts.padding
    config.four_channels = 1
    config.hash_channels = 1
    config.pad_width = opts.pad_width
    config.patch_list = patch_list
    config.patch_probs = patch_probs
    config.total_kimg = max(int(opts.duration * 1000), 1)
    config.ema_halflife_kimg = 500
    config.batch_size = opts.batch
    config.batch_gpu = opts.batch_gpu
    config.loss_scaling = 1
    config.cudnn_benchmark = True
    config.kimg_per_tick = opts.tick
    config.snapshot_ticks = None if opts.snap == 0 else opts.snap
    config.state_dump_ticks = None if opts.dump == 0 else opts.dump
    config.seed = opts.seed

    cond_string = 'uncond'
    dtype_string = 'fp16' if opts.fp16 else 'fp32'
    description = (
        f'aapm_3-{cond_string}-ddpmpp-pedm-'
        f'gpus{dist.get_world_size()}-batch{opts.batch}-{dtype_string}'
    )
    weight_tag = str(opts.ssim_weight).replace('.', 'p')
    sigma_tag = str(opts.ssim_sigma_max).replace('.', 'p')
    description += (
        f'-magnitude-ssim-w{weight_tag}'
        f'-smax{sigma_tag}-win{opts.ssim_window_size}'
    )
    if opts.desc:
        description += f'-{opts.desc}'

    if dist.get_rank() != 0:
        config.run_dir = None
    elif opts.nosubdir:
        config.run_dir = opts.outdir
    else:
        previous_ids = []
        if os.path.isdir(opts.outdir):
            for entry in os.listdir(opts.outdir):
                if not os.path.isdir(os.path.join(opts.outdir, entry)):
                    continue
                match = re.match(r'^\d+', entry)
                if match:
                    previous_ids.append(int(match.group()))
        run_id = max(previous_ids, default=-1) + 1
        config.run_dir = os.path.join(
            opts.outdir,
            f'{run_id:05d}-{description}',
        )
        if os.path.exists(config.run_dir):
            raise click.ClickException(
                f'output directory already exists: {config.run_dir}'
            )

    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(config, indent=2))
    dist.print0()
    dist.print0(f'Output directory: {config.run_dir}')
    dist.print0(f'Dataset path:     {config.dataset_kwargs.path}')
    dist.print0(
        'Experiment:       '
        'PaDIS-MRI Baseline + low-noise magnitude SSIM'
    )
    dist.print0()

    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(config.run_dir, exist_ok=True)
        with open(
            os.path.join(config.run_dir, 'training_options.json'),
            'wt',
            encoding='utf-8',
        ) as handle:
            json.dump(config, handle, indent=2)
        dnnlib.util.Logger(
            file_name=os.path.join(config.run_dir, 'log.txt'),
            file_mode='a',
            should_flush=True,
        )

    training_loop.training_loop(**config)


if __name__ == '__main__':
    main()
