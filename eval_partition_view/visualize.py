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
    # Figure display references only. Keep every raw map/statistic unchanged.
    magnitude_limit_raw = float(np.percentile(maps['gt_magnitude'], 99))
    difference_limit_raw = float(np.percentile(np.concatenate(
        [maps['differences'].reshape(-1), maps['spread'].reshape(-1)]), 99))
    magnitude_vmax = args.magnitude_vmax or positive_limit(magnitude_limit_raw)
    difference_vmax = args.difference_vmax or positive_limit(difference_limit_raw)
    magnitude_display = np.clip(maps['magnitudes'] / magnitude_vmax, 0, 1)
    gt_display = np.clip(maps['gt_magnitude'] / magnitude_vmax, 0, 1)
    mean_display = np.clip(np.abs(maps['mean_complex']) / magnitude_vmax, 0, 1)
    magnitude_label = ('Magnitude / shared manual reference (display only)' if args.magnitude_vmax
                       else 'Magnitude / GT P99 (display only)')
    difference_label = ('D / S (raw units; manual clipping, P99 default)' if args.difference_vmax
                        else 'D / S (raw units; joint P99 clipping)')
    residual_vmax = 0.2  # Shared GT residual display limit only; raw maps stay unchanged.
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
        save_image(full_dir / f'd_step060_inner{j+1:02d}.png', magnitude_display[j], 'gray', 1.0)
    save_image(full_dir / 'gt_magnitude.png', gt_display, 'gray', 1.0)
    save_image(full_dir / 'complex_mean_magnitude.png', mean_display, 'gray', 1.0)
    for j in selected:
        save_image(roi_dir / f'd_step060_inner{j+1:02d}_roi.png', magnitude_display[j][region],
                   'gray', 1.0, zoom=4)
        save_image(diff_dir / f'D_inner{j+1:02d}.png', maps['differences'][j], 'turbo', difference_vmax)
        save_image(diff_roi_dir / f'D_inner{j+1:02d}_roi.png', maps['differences'][j][region],
                   'turbo', difference_vmax, zoom=4)
    save_image(roi_dir / 'gt_magnitude_roi.png', gt_display[region], 'gray', 1.0, zoom=4)
    save_image(destination / 'within_step_spread_S.png', maps['spread'], 'turbo', difference_vmax)
    save_image(destination / 'within_step_spread_S_roi.png', maps['spread'][region], 'turbo', difference_vmax, zoom=4)

    # Image-only assets for PPT/Illustrator: reuse the same ROI and display scales.
    asset_dir = Path(output_dir) / 'assets'
    asset_dir.mkdir(exist_ok=False)
    for inner in (3, 6, 10):
        save_image(asset_dir / f'roi_inner{inner}.png', magnitude_display[inner-1][region],
                   'gray', 1.0, zoom=4)
        if args.gt_residual:
            save_image(asset_dir / f'residual_inner{inner}.png',
                       maps['gt_residuals'][inner-1][region], 'turbo', residual_vmax, zoom=4)
    save_image(asset_dir / 'gt_roi.png', gt_display[region], 'gray', 1.0, zoom=4)

    def colorbar_file(filename, cmap, upper, label, ticks=None):
        fig, ax = plt.subplots(figsize=(5.4, 0.95), dpi=220)
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, upper), cmap=cmap), cax=ax,
                           orientation='horizontal', label=label, ticks=ticks)
        if ticks is not None:
            bar.set_ticklabels([format(value, 'g') for value in ticks])
        fig.subplots_adjust(left=.08, right=.96, bottom=.53, top=.9)
        fig.savefig(destination / filename, dpi=220, facecolor='white')
        plt.close(fig)
        assets.append(str((destination / filename).relative_to(Path(output_dir))))

    colorbar_file('magnitude_colorbar.png', 'gray', 1.0, magnitude_label)
    colorbar_file('variation_colorbar.png', 'turbo', difference_vmax, difference_label)

    # Explicit physical sizes keep every ROI panel and the vertical colorbar
    # exactly the same height, including when a rectangular ROI is requested.
    panel_width, gap, left = 2.3, .10, .42
    bar_gap, bar_width, right = .08, .10, 1.05
    fov_height, roi_height = panel_width, panel_width * height / width
    row_gap, top, bottom = .18, .34, .14
    figure_width = left + 4*panel_width + 3*gap + bar_gap + bar_width + right
    figure_height = top + fov_height + 2*roi_height + 2*row_gap + bottom
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=200)
    row_bottoms = [bottom + 2*roi_height + 2*row_gap,
                   bottom + roi_height + row_gap, bottom]
    row_heights = [fov_height, roi_height, roi_height]
    row_labels = ['Full-FOV', 'ROI', 'Difference']

    for col, j in enumerate(selected):
        arrays = [magnitude_display[j], magnitude_display[j][region],
                  maps['differences'][j][region]]
        for row, array in enumerate(arrays):
            ax = fig.add_axes([(left + col*(panel_width+gap))/figure_width,
                               row_bottoms[row]/figure_height,
                               panel_width/figure_width, row_heights[row]/figure_height])
            ax.imshow(array, cmap='gray' if row < 2 else 'turbo', vmin=0,
                      vmax=1.0 if row < 2 else difference_vmax, interpolation='nearest')
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Partition {j+1}', fontsize=12, pad=5)
                ax.add_patch(Rectangle((x-.5, y-.5), width, height, fill=False,
                                       edgecolor='#e2ba59', linewidth=1.0))
            if col == 0:
                ax.text(-.055, .5, row_labels[row], transform=ax.transAxes,
                        rotation=90, ha='right', va='center', fontsize=11)
    spread_ax = fig.add_axes([(left + 3*(panel_width+gap))/figure_width,
                             bottom/figure_height, panel_width/figure_width,
                             roi_height/figure_height])
    spread_ax.imshow(maps['spread'][region], cmap='turbo', vmin=0,
                     vmax=difference_vmax, interpolation='nearest')
    spread_ax.axis('off')
    spread_ax.set_title('Variation S', fontsize=12, pad=5)
    cax = fig.add_axes([(left + 4*panel_width + 3*gap + bar_gap)/figure_width,
                       bottom/figure_height, bar_width/figure_width,
                       roi_height/figure_height])
    colorbar_ticks = [0.0, difference_vmax/2, difference_vmax]
    # Significant-digit formatting selects decimal/scientific notation from
    # the actual scale, without hiding the upper endpoint in an offset label.
    for precision in range(4, 11):
        colorbar_tick_labels = ['0'] + [format(v, f'.{precision}g') for v in colorbar_ticks[1:]]
        if len(set(colorbar_tick_labels)) == 3:
            break
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, difference_vmax), cmap='turbo'),
                           cax=cax, orientation='vertical', ticks=colorbar_ticks)
    colorbar.set_ticklabels(colorbar_tick_labels)
    colorbar.ax.tick_params(labelsize=10, length=3, pad=4)
    fig.savefig(destination / 'overview.png', dpi=220, facecolor='white')
    plt.close(fig)
    assets.append('figures/overview.png')

    if args.gt_residual:
        residual_dir = destination / 'gt_complex_residual_roi'
        residual_dir.mkdir()
        fig, axs = plt.subplots(1, len(selected), figsize=(3.25*len(selected), 4.6), dpi=200)
        for ax, j in zip(axs, selected):
            residual = maps['gt_residuals'][j][region]
            save_image(residual_dir / f'GT_complex_residual_inner{j+1:02d}_roi.png', residual,
                       'turbo', residual_vmax, zoom=4)
            ax.imshow(residual, cmap='turbo', vmin=0, vmax=residual_vmax, interpolation='nearest')
            ax.set_title(f'Inner {j+1}: $|d_{{60,{j+1}}} - GT|$')
            ax.axis('off')
        fig.suptitle('Optional auxiliary: complex GT residual in the same ROI', fontsize=13)
        fig.subplots_adjust(left=.02, right=.98, top=.78, bottom=.34, wspace=.12)
        cax = fig.add_axes([.3, .18, .4, .025])
        residual_colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, residual_vmax), cmap='turbo'), cax=cax,
            orientation='horizontal', label='Complex GT residual (raw tensor units)',
            ticks=[0.0, 0.1, 0.2])
        residual_colorbar.set_ticklabels(['0', '0.1', '0.2'])
        fig.text(.5, .025, 'Complex GT residual is an auxiliary output, not deviation to the within-step mean.\n'
                 'No magnitude PSNR interpretation or phase alignment is applied.', ha='center', fontsize=9)
        fig.savefig(destination / 'gt_residual_overview.png', dpi=220, facecolor='white')
        plt.close(fig)
        assets.append('figures/gt_residual_overview.png')
        colorbar_file('gt_complex_residual_colorbar.png', 'turbo', residual_vmax,
                      'Complex GT residual (raw tensor units)', ticks=[0.0, 0.1, 0.2])

    metadata = {
        'figure_title': TITLE, 'input_pt': str(getattr(args, 'input_pt', '') or ''),
        'display_inner_loops': list(args.display_indices), 'roi_xywh_zero_based': list(args.roi),
        'roi_slice_row_col': [y, y+height, x, x+width],
        'roi_png_zoom': '4x nearest-neighbor replication, identical for every panel',
        'magnitude_range': [0.0, magnitude_vmax], 'variation_range_D_and_S': [0.0, difference_vmax],
        'magnitude_display_range': [0.0, 1.0],
        'gt_magnitude_q99': magnitude_limit_raw, 'magnitude_percentile': 99,
        'magnitude_reference_actual': magnitude_vmax,
        'magnitude_manual_override': args.magnitude_vmax is not None,
        'magnitude_zero_percentile_fallback': args.magnitude_vmax is None and magnitude_limit_raw <= 0,
        'magnitude_display_formula': 'clip(abs(I) / shared_reference, 0, 1); default shared_reference=P99(abs(GT))',
        'magnitude_normalization_scope': 'Figure display only; not historical MATLAB ACS-99 data normalization',
        'default_magnitude_limit_raw': magnitude_limit_raw,
        'default_variation_limit_raw': difference_limit_raw,
        'magnitude_scale_policy': 'one shared GT full-FOV P99 reference for all ten estimates, GT, mean and ROI; manual override supported',
        'variation_scale_policy': 'one shared joint full-FOV P99 range across all ten D_j and S, also used for ROI; manual override supported',
        'variation_percentile': 99, 'variation_colormap': 'turbo',
        'variation_percentile_population': 'concatenated full-FOV pixels of all ten D_j and one S, each map included once',
        'variation_manual_override': args.difference_vmax is not None,
        'variation_zero_percentile_fallback': args.difference_vmax is None and difference_limit_raw <= 0,
        'percentile_interpolation': 'linear (NumPy percentile default)',
        'variation_clipping': 'color saturation only, raw D_j and S are unchanged',
        'overview_third_row': 'three selected ROI differences and ROI population RMS over all ten estimates',
        'overview_colorbar_ticks': colorbar_ticks,
        'overview_colorbar_tick_labels': colorbar_tick_labels,
        'display_only': True, 'raw_complex_tensors_modified': False,
        'per_panel_normalization': False, 'phase_alignment': False,
        'difference_formula': 'D_j = abs(d_j - mean_complex(all 10))',
        'spread_formula': 'S = sqrt(mean_j(abs(d_j - mean_complex(all 10))**2)), divisor=10',
        'statistic_dtype': 'complex128 / float64, computed offline from saved tensor values',
        'gt_residual_enabled': bool(args.gt_residual),
        'gt_residual_formula': 'abs(d_j - gt), complex residual, not magnitude residual',
        'gt_residual_range': [0.0, residual_vmax] if args.gt_residual else None,
        'gt_residual_colormap': 'turbo' if args.gt_residual else None,
        'gt_residual_scale_policy': 'fixed shared display range [0, 0.2] for all complex GT residual panels and assets; color saturation only',
        'gt_residual_colorbar_ticks': [0.0, 0.1, 0.2] if args.gt_residual else None,
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
