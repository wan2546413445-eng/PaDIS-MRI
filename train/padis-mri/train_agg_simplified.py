# ---------------------------------------------------------------
# Modified from:
# https://github.com/jasonhu4/PaDIS/blob/main/train.py
#
# The license for the original version of this file can be
# found here: https://github.com/jasonhu4/PaDIS/blob/main/LICENSE.
# ---------------------------------------------------------------

"""Train PaDIS-MRI with simplified AGG and one patch per source image.

All patch scales use the support-fraction-plus-gradient sampler from
training.agg_sampler. The standard EDM objective and batch accounting remain
unchanged.
"""

import os
import sys
import re
import json
import click
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import dnnlib

from torch_utils import distributed as dist
from training import training_loop

import warnings

warnings.filterwarnings('ignore',
                        'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
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
    if isinstance(s, list): return s
    out = []
    for p in s.split(','):
        out.append(float(p))
    return out



def _load_sampler_audit_images(data_path, pad_width, max_images):
    payload = torch.load(data_path, map_location='cpu')
    if 'x_est_gt' not in payload:
        raise KeyError(f"{data_path} must contain key 'x_est_gt'")
    tensor = payload['x_est_gt']

    if torch.is_complex(tensor):
        images = torch.view_as_real(tensor).permute(0, 3, 1, 2)
    elif tensor.ndim == 4 and tensor.shape[-1] == 2:
        images = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 4 and tensor.shape[1] >= 2:
        images = tensor[:, :2]
    else:
        raise ValueError(
            'x_est_gt must be complex [N,H,W], [N,H,W,2], or [N,2,H,W]'
        )

    images = images[:min(int(max_images), images.shape[0])].float()
    if images.shape[0] == 0:
        raise ValueError('sampler audit received an empty dataset')
    if int(pad_width) > 0:
        images = F.pad(
            images,
            [int(pad_width), int(pad_width), int(pad_width), int(pad_width)],
        )
    return images


def _coverage_from_locations(top, left, patch_size, height, width):
    """Convert sampled top-left coordinates into a patch-coverage heatmap."""
    diff = torch.zeros([height + 1, width + 1], dtype=torch.float64)
    patch_size = int(patch_size)
    for row, col in zip(top.cpu().tolist(), left.cpu().tolist()):
        diff[row, col] += 1
        diff[row + patch_size, col] -= 1
        diff[row, col + patch_size] -= 1
        diff[row + patch_size, col + patch_size] += 1
    return diff.cumsum(0).cumsum(1)[:-1, :-1]


def run_sampler_audit(
    data_path,
    output_dir,
    patch_list,
    pad_width,
    rho,
    gamma,
    image_count,
    draws,
    seed,
):
    """Verify one-patch cardinality and compare uniform/guided heatmaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from training.agg_sampler import sample_single_locations
    from training.patch_loss_simplified_agg import (
        SimplifiedAggPatchEDMLoss,
    )

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = _load_sampler_audit_images(
        data_path, pad_width=pad_width, max_images=image_count
    ).to(device)
    batch, _, height, width = images.shape
    magnitude = images[:, :min(2, images.shape[1])].square().sum(1).sqrt()
    mean_magnitude = magnitude.mean(0)
    peak = magnitude.amax(dim=(1, 2), keepdim=True)
    support = magnitude > (0.05 * peak)

    report = {
        'data_path': data_path,
        'seed': int(seed),
        'input_images': int(batch),
        'padded_shape': [int(height), int(width)],
        'rho': float(rho),
        'gamma': float(gamma),
        'draws_per_image': int(draws),
        'definition': (
            'coverage heatmap; every sampled patch adds one count to every '
            'pixel that it covers'
        ),
        'patch_sizes': {},
    }

    loss_probe = SimplifiedAggPatchEDMLoss(
        agg_rho=rho, agg_gamma=gamma
    )

    for patch_size in patch_list:
        patch_size = int(patch_size)
        probe_patches, probe_positions = loss_probe.pachify(
            images, patch_size
        )
        if probe_patches.shape[0] != batch:
            raise RuntimeError('single-patch cardinality check failed')
        if probe_positions.shape[0] != batch:
            raise RuntimeError('position-map cardinality check failed')

        mode_results = {}
        coverages = {}
        for mode, mode_rho in [('uniform', 0.0), ('guided', float(rho))]:
            coverage = torch.zeros([height, width], dtype=torch.float64)
            support_sum = 0.0
            gradient_sum = 0.0
            observed = 0

            grad_x = torch.zeros_like(magnitude)
            grad_y = torch.zeros_like(magnitude)
            grad_x[:, :, 1:] = magnitude[:, :, 1:] - magnitude[:, :, :-1]
            grad_y[:, 1:, :] = magnitude[:, 1:, :] - magnitude[:, :-1, :]
            gradient = (grad_x.square() + grad_y.square()).sqrt()

            for _ in range(int(draws)):
                top, left = sample_single_locations(
                    images, patch_size, mode_rho, gamma
                )
                if top.numel() != batch or left.numel() != batch:
                    raise RuntimeError('sampler did not return one location per image')
                coverage += _coverage_from_locations(
                    top, left, patch_size, height, width
                )
                for index in range(batch):
                    row = int(top[index].item())
                    col = int(left[index].item())
                    region = (
                        slice(row, row + patch_size),
                        slice(col, col + patch_size),
                    )
                    support_sum += float(
                        support[index][region].float().mean().item()
                    )
                    gradient_sum += float(
                        gradient[index][region].mean().item()
                    )
                    observed += 1

            expected = int(batch * draws)
            if observed != expected:
                raise RuntimeError(
                    f'expected {expected} sampled patches, observed {observed}'
                )
            coverages[mode] = coverage
            mode_results[mode] = {
                'expected_patch_count': expected,
                'observed_patch_count': observed,
                'patches_per_source_per_call': 1.0,
                'mean_selected_support_fraction': support_sum / observed,
                'mean_selected_gradient_magnitude': gradient_sum / observed,
            }

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        vmax = max(
            float(coverages['uniform'].max()),
            float(coverages['guided'].max()),
            1.0,
        )
        for axis, mode in zip(axes, ['uniform', 'guided']):
            axis.imshow(mean_magnitude.cpu().numpy(), cmap='gray')
            heat = axis.imshow(
                coverages[mode].numpy(),
                cmap='magma',
                alpha=0.62,
                vmin=0.0,
                vmax=vmax,
            )
            axis.set_title(
                f"{mode}: support="
                f"{mode_results[mode]['mean_selected_support_fraction']:.3f}"
            )
            axis.axis('off')
        figure.colorbar(heat, ax=axes, shrink=0.78, label='patch coverage count')
        figure.suptitle(
            f'Patch size {patch_size}: uniform vs simplified AGG '
            f'(rho={rho}, gamma={gamma})'
        )
        heatmap_name = f'sampling_heatmap_p{patch_size}.png'
        figure.savefig(os.path.join(output_dir, heatmap_name), dpi=180)
        plt.close(figure)

        report['patch_sizes'][str(patch_size)] = {
            'probe_input_batch': int(batch),
            'probe_output_patch_batch': int(probe_patches.shape[0]),
            'probe_position_batch': int(probe_positions.shape[0]),
            'heatmap': heatmap_name,
            **mode_results,
        }

    report_path = os.path.join(output_dir, 'sampler_audit.json')
    with open(report_path, 'wt') as file:
        json.dump(report, file, indent=2)
    return report_path

# ----------------------------------------------------------------------------

@click.command()
# Patch options
@click.option('--real_p', help='Full size image ratio', metavar='INT', type=click.FloatRange(min=0, max=1), default=0.5,
              show_default=True)
@click.option('--train_on_latents', help='Training on latent embeddings', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--progressive', help='Training on latent embeddings', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--padding', help='Zero padding for training', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--four_channels', help='Number of Fourier embedding freqs', metavar='INT', type=int, default=1,
              show_default=True)
@click.option('--hash_channels', help='Number of hash embedding outputs', metavar='INT', type=int, default=1,
              show_default=True)
@click.option('--pad_width', help='Width on all sides of zero padding', metavar='INT', type=int, required=True)
@click.option('--patch-list', help='Comma-separated patch sizes, e.g. 96,192,384', type=str)
@click.option('--patch-probs', help='Comma-separated probabilities for patch sizes, e.g. 0.2,0.3,0.5', type=str)
# Main options.
@click.option('--outdir', help='Where to save the results', metavar='DIR', type=str, required=True)
@click.option('--data', help='Path to the dataset', metavar='ZIP|DIR', type=str, required=True)
@click.option('--cond', help='Train class-conditional model', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--arch', help='Network architecture', metavar='ddpmpp|ncsnpp|adm',
              type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond', help='Preconditioning & loss function', metavar='vp|ve|edm',
              type=click.Choice(['vp', 've', 'edm', 'pedm']), default='pedm', show_default=True)
# Hyperparameters.
@click.option('--duration', help='Training duration', metavar='MIMG', type=click.FloatRange(min=0, min_open=True),
              default=200, show_default=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), default=512,
              show_default=True)
@click.option('--batch-gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1))
@click.option('--cbase', help='Channel multiplier  [default: varies]', metavar='INT', type=int)
@click.option('--cres', help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr', help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=10e-4,
              show_default=True)
@click.option('--ema', help='EMA half-life', metavar='MIMG', type=click.FloatRange(min=0), default=0.5,
              show_default=True)
@click.option('--dropout', help='Dropout probability', metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.13, show_default=True)
@click.option('--augment', help='Augment probability', metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.12, show_default=True)
@click.option('--xflip', help='Enable dataset x-flips', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--implicit_mlp', help='encoding coordbefore sending to the conv', metavar='BOOL', type=bool,
              default=False, show_default=True)
# Performance-related.
@click.option('--fp16', help='Enable mixed-precision training', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--ls', help='Loss scaling', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1,
              show_default=True)
@click.option('--bench', help='Enable cuDNN benchmarking', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--cache', help='Cache dataset in CPU memory', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=1,
              show_default=True)
# I/O-related.
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
@click.option('--nosubdir', help='Do not create a subdirectory for results', is_flag=True)
@click.option('--tick', help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--snap', help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--dump', help='How often to dump state', metavar='TICKS', type=click.IntRange(min=1), default=500,
              show_default=True)
@click.option('--seed', help='Random seed  [default: random]', metavar='INT', type=int)
@click.option('--transfer', help='Transfer learning from network pickle', metavar='PKL|URL', type=str)
@click.option('--resume', help='Resume from previous training state', metavar='PT', type=str)
@click.option('--agg-rho', help='Mixture weight of structure-guided sampling',
              type=click.FloatRange(min=0, max=1), default=0.65, show_default=True)
@click.option('--agg-gamma', help='Gradient contribution in the AGG score',
              type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--sampler-audit-dir', help='Write sampler audit JSON and heatmaps, then exit',
              metavar='DIR', type=str)
@click.option('--sampler-audit-images', help='Number of dataset images used by sampler audit',
              type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--sampler-audit-draws', help='Sampling repetitions per audit image',
              type=click.IntRange(min=1), default=200, show_default=True)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Example:

    torchrun --standalone --nproc_per_node=1 train_agg_simplified.py \\
        --outdir=training-runs --data=training-data.pt --pad_width=64 \\
        --patch-list=16,32,64 --patch-probs=0.2,0.3,0.5
    """
    opts = dnnlib.EasyDict(kwargs)

    if opts.sampler_audit_dir is not None:
        patch_list = (
            parse_int_list(opts.patch_list)
            if opts.patch_list is not None
            else [16, 32, 64]
        )
        report_path = run_sampler_audit(
            data_path=opts.data,
            output_dir=opts.sampler_audit_dir,
            patch_list=patch_list,
            pad_width=opts.pad_width if opts.padding else 0,
            rho=opts.agg_rho,
            gamma=opts.agg_gamma,
            image_count=opts.sampler_audit_images,
            draws=opts.sampler_audit_draws,
            seed=opts.seed if opts.seed is not None else 0,
        )
        click.echo(f'Sampler audit completed: {report_path}')
        return

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDatasetX', path=opts.data,
                                       use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
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
    if (opts.patch_list is None) != (opts.patch_probs is None):
        raise click.ClickException('--patch-list and --patch-probs must be set together')
    if opts.patch_list is not None:
        if len(c.patch_list) != len(c.patch_probs):
            raise click.ClickException('--patch-list and --patch-probs must have equal length')
        if not c.patch_list or min(c.patch_list) < 2:
            raise click.ClickException('all patch sizes must be at least 2')

    # Validate dataset options.
    # try:
    #     dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    #     dataset_name = dataset_obj.name
    #     c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
    #     c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
    #     #print(len(dataset_obj))
    #     if opts.cond and not dataset_obj.has_labels:
    #         raise click.ClickException('--cond=True requires labels specified in dataset.json')
    #     del dataset_obj # conserve memory
    # except IOError as err:
    #     raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard',
                                decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1, 1], model_channels=128,
                                channel_mult=[2, 2, 2], hash_channels=c.hash_channels)
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual',
                                decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=128,
                                channel_mult=[2, 2, 2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1, 2, 3, 4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'pedm':
        c.network_kwargs.class_name = 'training.networks.Patch_EDMPrecond'
        c.network_kwargs.sigma_data = 0.5

        c.loss_kwargs.class_name = (
            'training.patch_loss_simplified_agg.'
            'SimplifiedAggPatchEDMLoss'
        )
        c.loss_kwargs.update(
            P_mean=-1.2,
            P_std=1.2,
            sigma_data=0.5,
            agg_rho=opts.agg_rho,
            agg_gamma=opts.agg_gamma,
        )

    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
        # c.augment_kwargs.update(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        # c.network_kwargs.augment_dim = 6
    if opts.implicit_mlp:
        c.network_kwargs.implicit_mlp = True
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
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

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    dataset_name = 'aapm_3'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    rho_tag = str(opts.agg_rho).replace('.', 'p')
    gamma_tag = str(opts.agg_gamma).replace('.', 'p')
    desc += (
        f'-agg-single-rho{rho_tag}'
        f'-gamma{gamma_tag}'
    )
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
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

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0('Training mode:           simplified AGG, single patch')
    dist.print0(f'AGG mixture weight:      {opts.agg_rho}')
    dist.print0(f'AGG gradient weight:     {opts.agg_gamma}')
    dist.print0('Patches per source:      1')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
