"""Offline plots. The empirical mean is never supplied to reconstruction."""

import json
from copy import copy
from pathlib import Path

import numpy as np

TITLE = 'Within-step variation of full-FOV denoised estimates'


def calculate_maps(estimates, gt):
    estimates = np.asarray(estimates)
    gt = np.asarray(gt)
    if estimates.shape != (10, 384, 384) or not np.iscomplexobj(estimates):
        raise ValueError('Expected ten raw complex [384,384] estimates.')
    if gt.shape != (384, 384):
        raise ValueError('GT dimensions differ from original FOV.')
    if not np.isfinite(estimates).all() or not np.isfinite(gt).all():
        raise ValueError('Non-finite tensor values cannot be plotted.')
    # Float64 statistics avoid accumulating additional rounding error in the
    # diagnostic mean. This does not modify the original .pt tensor values.
    complex_values = estimates.astype(np.complex128)
    mean = complex_values.mean(axis=0)
    differences = np.abs(complex_values - mean[None])
    spread = np.sqrt(np.mean(differences ** 2, axis=0))  # population RMS, 1/10
    return {'mean_complex': mean, 'magnitudes': np.abs(complex_values),
            'gt_magnitude': np.abs(gt), 'differences': differences, 'spread': spread,
            'gt_residuals': np.abs(complex_values - gt.astype(np.complex128)[None])}


def positive_limit(value):
    # Identical estimates are a valid scientific outcome. Avoid a degenerate
    # 0..0 color scale, and explicitly record the plotting fallback.
    return float(value) if value > 0 else 1.0


def export_arrays(estimates, gt, offsets, sigma, output_dir, args):
    """Render saved arrays. Does not import any original PaDIS/GPU modules."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    maps = calculate_maps(estimates, gt)
    magnitude_limit_raw = max(float(maps['magnitudes'].max()), float(maps['gt_magnitude'].max()))
    difference_limit_raw = max(float(maps['differences'].max()), float(maps['spread'].max()))
    magnitude_vmax = args.magnitude_vmax or positive_limit(magnitude_limit_raw)
    difference_vmax = args.difference_vmax or positive_limit(difference_limit_raw)
    residual_vmax = positive_limit(float(maps['gt_residuals'].max()))
    x, y, width, height = args.roi
    region = (slice(y, y + height), slice(x, x + width))
    selected = [j - 1 for j in args.display_indices]
    destination = Path(output_dir) / 'figures'
    destination.mkdir(exist_ok=False)
    full_dir = destination / 'full_fov'
    roi_dir = destination / 'roi'
    diff_dir = destination / 'difference_to_complex_mean'
    diff_roi_dir = destination / 'difference_roi'
    for directory in [full_dir, roi_dir, diff_dir, diff_roi_dir]:
        directory.mkdir()
    assets = []

    def save_image(path, array, cmap, vmax, zoom=1):
        image_array = np.repeat(np.repeat(array, zoom, axis=0), zoom, axis=1) if zoom > 1 else array
        plt.imsave(path, image_array, cmap=cmap, vmin=0.0, vmax=vmax)
        assets.append(str(path.relative_to(Path(output_dir))))

    for j in range(10):
        save_image(full_dir / f'd_step060_inner{j+1:02d}.png', maps['magnitudes'][j], 'gray', magnitude_vmax)
    save_image(full_dir / 'gt_magnitude.png', maps['gt_magnitude'], 'gray', magnitude_vmax)
    save_image(full_dir / 'complex_mean_magnitude.png', np.abs(maps['mean_complex']), 'gray', magnitude_vmax)
    for j in selected:
        save_image(roi_dir / f'd_step060_inner{j+1:02d}_roi.png', maps['magnitudes'][j][region],
                   'gray', magnitude_vmax, zoom=4)
        save_image(diff_dir / f'D_inner{j+1:02d}.png', maps['differences'][j], 'magma', difference_vmax)
        save_image(diff_roi_dir / f'D_inner{j+1:02d}_roi.png', maps['differences'][j][region],
                   'magma', difference_vmax, zoom=4)
    save_image(roi_dir / 'gt_magnitude_roi.png', maps['gt_magnitude'][region], 'gray', magnitude_vmax, zoom=4)
    save_image(destination / 'within_step_spread_S.png', maps['spread'], 'magma', difference_vmax)
    save_image(destination / 'within_step_spread_S_roi.png', maps['spread'][region], 'magma', difference_vmax, zoom=4)

    def colorbar_file(filename, cmap, upper, label):
        fig, ax = plt.subplots(figsize=(5.4, 0.95), dpi=220)
        fig.colorbar(ScalarMappable(norm=Normalize(0, upper), cmap=cmap), cax=ax,
                     orientation='horizontal', label=label)
        fig.subplots_adjust(left=.08, right=.96, bottom=.53, top=.9)
        fig.savefig(destination / filename, dpi=220, facecolor='white')
        plt.close(fig)
        assets.append(str((destination / filename).relative_to(Path(output_dir))))

    colorbar_file('magnitude_colorbar.png', 'gray', magnitude_vmax, 'Magnitude (raw tensor units)')
    colorbar_file('variation_colorbar.png', 'magma', difference_vmax,
                  'Complex deviation / population RMS (raw tensor units)')

    fig, axes = plt.subplots(3, 5, figsize=(16.5, 10.6), dpi=200)
    for col, j in enumerate(selected):
        axes[0, col].imshow(maps['magnitudes'][j], cmap='gray', vmin=0, vmax=magnitude_vmax,
                            interpolation='nearest')
        axes[0, col].add_patch(Rectangle((x-.5, y-.5), width, height, fill=False,
                                       edgecolor='#e2ba59', linewidth=1.0))
        off = offsets[j]
        axes[0, col].set_title(f'Inner loop {j+1}\noffset ({int(off[0])}, {int(off[1])})', fontsize=11)
        axes[1, col].imshow(maps['magnitudes'][j][region], cmap='gray', vmin=0,
                            vmax=magnitude_vmax, interpolation='nearest')
        axes[1, col].set_title(f'Same ROI, inner {j+1}', fontsize=10)
        axes[2, col].imshow(maps['differences'][j], cmap='magma', vmin=0,
                            vmax=difference_vmax, interpolation='nearest')
        axes[2, col].set_title(f'$|d_{{60,{j+1}}} - \\bar d_{{60}}|$', fontsize=11)
    axes[0, 4].imshow(maps['gt_magnitude'], cmap='gray', vmin=0, vmax=magnitude_vmax,
                      interpolation='nearest')
    axes[0, 4].add_patch(Rectangle((x-.5, y-.5), width, height, fill=False,
                                 edgecolor='#e2ba59', linewidth=1.0))
    axes[0, 4].set_title('GT magnitude', fontsize=11)
    axes[1, 4].imshow(maps['gt_magnitude'][region], cmap='gray', vmin=0,
                      vmax=magnitude_vmax, interpolation='nearest')
    axes[1, 4].set_title('GT, same ROI', fontsize=10)
    axes[2, 4].imshow(maps['spread'], cmap='magma', vmin=0, vmax=difference_vmax,
                      interpolation='nearest')
    axes[2, 4].set_title('$S$: population RMS over all 10', fontsize=10)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(TITLE, fontsize=17, y=.975)
    fig.text(.5, .935, f'Original Baseline, sample 3, outer step 60, actual sigma={sigma:.8g}',
             ha='center', fontsize=11)
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.16, wspace=.10, hspace=.25)
    cax1 = fig.add_axes([.12, .093, .32, .018])
    cax2 = fig.add_axes([.57, .093, .32, .018])
    fig.colorbar(ScalarMappable(norm=Normalize(0, magnitude_vmax), cmap='gray'), cax=cax1,
                 orientation='horizontal', label='Shared magnitude scale (raw tensor units)')
    fig.colorbar(ScalarMappable(norm=Normalize(0, difference_vmax), cmap='magma'), cax=cax2,
                 orientation='horizontal', label='Shared D / S scale (raw tensor units)')
    fig.text(.5, .018, 'Sequential inner loops: partition, reconstruction state and injected noise all vary.\n'
             'The complex mean is an offline diagnostic only.', ha='center', fontsize=10)
    fig.savefig(destination / 'overview.png', dpi=220, facecolor='white')
    plt.close(fig)
    assets.append('figures/overview.png')

    if args.gt_residual:
        residual_dir = destination / 'gt_complex_residual_roi'
        residual_dir.mkdir()
        fig, axs = plt.subplots(1, 4, figsize=(13, 4.6), dpi=200)
        for ax, j in zip(axs, selected):
            residual = maps['gt_residuals'][j][region]
            save_image(residual_dir / f'GT_complex_residual_inner{j+1:02d}_roi.png', residual,
                       'magma', residual_vmax, zoom=4)
            ax.imshow(residual, cmap='magma', vmin=0, vmax=residual_vmax, interpolation='nearest')
            ax.set_title(f'Inner {j+1}: $|d_{{60,{j+1}}} - GT|$')
            ax.axis('off')
        fig.suptitle('Optional auxiliary: complex GT residual in the same ROI', fontsize=13)
        fig.subplots_adjust(left=.02, right=.98, top=.78, bottom=.34, wspace=.12)
        cax = fig.add_axes([.3, .18, .4, .025])
        fig.colorbar(ScalarMappable(norm=Normalize(0, residual_vmax), cmap='magma'), cax=cax,
                     orientation='horizontal', label='Complex GT residual (raw tensor units)')
        fig.text(.5, .025, 'Complex GT residual is an auxiliary output, not deviation to the within-step mean.\n'
                 'No magnitude PSNR interpretation or phase alignment is applied.', ha='center', fontsize=9)
        fig.savefig(destination / 'gt_residual_overview.png', dpi=220, facecolor='white')
        plt.close(fig)
        assets.append('figures/gt_residual_overview.png')
        colorbar_file('gt_complex_residual_colorbar.png', 'magma', residual_vmax,
                      'Complex GT residual (raw tensor units)')

    metadata = {
        'figure_title': TITLE, 'input_pt': str(getattr(args, 'input_pt', '') or ''),
        'display_inner_loops': list(args.display_indices), 'roi_xywh_zero_based': list(args.roi),
        'roi_slice_row_col': [y, y+height, x, x+width],
        'roi_png_zoom': '4x nearest-neighbor replication, identical for every panel',
        'magnitude_range': [0.0, magnitude_vmax], 'variation_range_D_and_S': [0.0, difference_vmax],
        'default_magnitude_limit_raw': magnitude_limit_raw,
        'default_variation_limit_raw': difference_limit_raw,
        'magnitude_scale_policy': 'one shared full-FOV range for all ten estimates and GT',
        'variation_scale_policy': 'one shared full-FOV range across all ten D_j and S, also used for ROI',
        'per_panel_normalization': False, 'phase_alignment': False,
        'difference_formula': 'D_j = abs(d_j - mean_complex(all 10))',
        'spread_formula': 'S = sqrt(mean_j(abs(d_j - mean_complex(all 10))**2)), divisor=10',
        'statistic_dtype': 'complex128 / float64, computed offline from saved tensor values',
        'gt_residual_enabled': bool(args.gt_residual),
        'gt_residual_formula': 'abs(d_j - gt), complex residual, not magnitude residual',
        'gt_residual_range': [0.0, residual_vmax] if args.gt_residual else None,
        'interpretation': 'Variation also includes sequential state updates and injected noise. '
                          'Does not isolate partition effects or prove EMA/LGFC effectiveness.',
        'png_assets': assets,
    }
    with (Path(output_dir) / 'visualization_metadata.json').open('x', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print(f'[plot] Exported {len(assets)} PNG assets to {destination}', flush=True)
    return metadata


def export_from_pt(pt_path, output_dir, args):
    import torch

    payload = torch.load(pt_path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'padis_baseline_within_step_v1':
        raise ValueError('Not a validated within-step Baseline capture file.')
    if payload.get('outer_step') != 60 or not payload['metadata'].get('capture_complete'):
        raise ValueError('Need a complete real outer-step-60 capture.')
    if payload['metadata'].get('lgfc_enabled') is not False:
        raise ValueError('Input must be an original Baseline capture without LGFC.')
    if payload['inner_loop_numbers'].tolist() != list(range(1, 11)):
        raise ValueError('Inner-loop observations must be consecutive 1..10.')
    if payload['metadata'].get('denoiser_call_numbers') != list(range(591, 601)):
        raise ValueError('Unexpected real denoiser call provenance for step 60.')
    sigmas = payload['sigmas']
    if sigmas.shape != (10,) or not torch.equal(sigmas, sigmas[0].expand_as(sigmas)):
        raise ValueError('All ten estimates must share the actual sigma.')
    if tuple(payload['patch_indices'].shape) != (10, 49, 4):
        raise ValueError('Expected the ten actual 49-patch partitions.')
    offsets = payload['patch_indices'][:, 0, [0, 2]]
    if not torch.equal(offsets, payload['partition_offsets']):
        raise ValueError('Saved partition offsets and indices disagree.')
    plot_args = copy(args)
    plot_args.input_pt = str(Path(pt_path).resolve())
    metadata = export_arrays(payload['denoised_complex'].numpy(), payload['gt'].numpy(),
                             offsets.numpy(), float(sigmas[0]), output_dir, plot_args)
    # The input .pt is never modified or rewritten by offline plotting.
    return metadata
