import os
import re
import sys
import csv
import json
import glob
import math
import pickle
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dnnlib

from utils import fftmod
from inverse_operators import MRI_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure overlap consistency for PaDIS-MRI patch denoiser without modifying existing training/recon code.'
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to network-snapshot .pkl')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory containing sample_*.pt files')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save CSV / JSON / figures')

    parser.add_argument('--sample_indices', type=str, default='', help='Comma-separated sample ids, e.g. 0,1,2. If empty, uses first --num_samples files.')
    parser.add_argument('--num_samples', type=int, default=4, help='How many samples to use when --sample_indices is empty')
    parser.add_argument('--mask_select', type=int, default=7, help='Which mask_{R} to use when input_source=adjoint')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--pad', type=int, default=64)
    parser.add_argument('--psize', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32, help='Stride used to form overlapping patches. Must be smaller than psize.')
    parser.add_argument('--pair_mode', type=str, default='both', choices=['horizontal', 'vertical', 'both'])
    parser.add_argument('--max_pairs_per_sample', type=int, default=16, help='Randomly subsample at most this many overlapping pairs per sample')
    parser.add_argument('--repeats', type=int, default=1, help='Repeat each pair with fresh noise this many times')
    parser.add_argument('--input_source', type=str, default='gt', choices=['gt', 'adjoint'], help='Base image used before adding synthetic VE noise')

    parser.add_argument('--sigma_pairs', type=str, default='0.1:0.1,0.1:0.5', help='Comma-separated sigma_a:sigma_b pairs. Example: 0.1:0.1,0.1:0.5')
    parser.add_argument('--share_noise', action='store_true', help='Use the same random noise tensor for the two patches in a pair')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_maps', type=int, default=8, help='Maximum number of qualitative heatmaps to save')
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_ints(csv_str: str) -> List[int]:
    if csv_str is None or csv_str.strip() == '':
        return []
    return [int(x.strip()) for x in csv_str.split(',') if x.strip() != '']


def parse_sigma_pairs(spec: str) -> List[Tuple[float, float]]:
    pairs = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'Invalid sigma pair: {item}. Expected format sigma_a:sigma_b')
        a_str, b_str = item.split(':', 1)
        sigma_a = float(a_str)
        sigma_b = float(b_str)
        if sigma_a <= 0 or sigma_b <= 0:
            raise ValueError('All sigma values must be positive.')
        pairs.append((sigma_a, sigma_b))
    if not pairs:
        raise ValueError('No sigma pairs were provided.')
    return pairs


def list_available_sample_indices(val_dir: str) -> List[int]:
    paths = glob.glob(os.path.join(val_dir, 'sample_*.pt'))
    ids = []
    for path in paths:
        match = re.search(r'sample_(\d+)\.pt$', os.path.basename(path))
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def resolve_sample_indices(val_dir: str, sample_indices_csv: str, num_samples: int) -> List[int]:
    requested = parse_csv_ints(sample_indices_csv)
    available = list_available_sample_indices(val_dir)
    if not available:
        raise FileNotFoundError(f'No sample_*.pt files found in {val_dir}')
    if requested:
        missing = [idx for idx in requested if idx not in available]
        if missing:
            raise FileNotFoundError(f'Requested sample indices not found: {missing}')
        return requested
    return available[:min(num_samples, len(available))]


def load_model(model_path: str, device: torch.device):
    print(f'Loading network from {model_path}')
    with dnnlib.util.open_url(model_path, verbose=False) as f:
        model = pickle.load(f)['ema']
    model = model.to(device).eval()
    return model


def build_latents_pos(image_size: int, pad: int, device: torch.device) -> torch.Tensor:
    resolution = image_size + 2 * pad
    x = torch.linspace(-1, 1, resolution, device=device)
    y = torch.linspace(-1, 1, resolution, device=device)
    x_pos = x.view(1, -1).repeat(resolution, 1)
    y_pos = y.view(-1, 1).repeat(1, resolution)
    pos = torch.stack([x_pos, y_pos], dim=0)
    return pos.unsqueeze(0)


def load_base_image(
    val_dir: str,
    sample_idx: int,
    mask_select: int,
    input_source: str,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    sample_path = os.path.join(val_dir, f'sample_{sample_idx}.pt')
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(sample_path)
    data = torch.load(sample_path, weights_only=False)

    if input_source == 'gt':
        base = data['gt'][None, None, ...].to(device)
    elif input_source == 'adjoint':
        s_maps = fftmod(data['s_map'])[None, ...].to(device)
        fs_ksp = fftmod(data['ksp'])[None, ...].to(device)
        mask = data[f'mask_{mask_select}'][None, ...].to(device)
        ksp = mask * fs_ksp
        mri_utils = MRI_utils(mask=mask, maps=s_maps)
        base = mri_utils.adjoint(ksp)
    else:
        raise ValueError(f'Unsupported input_source: {input_source}')

    if pad > 0:
        base = torch.nn.functional.pad(base, (pad, pad, pad, pad), mode='constant', value=0)
    return base


def complex_to_two_channel(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x.squeeze(1).contiguous()).permute(0, 3, 1, 2).contiguous()


def generate_patch_boxes(height: int, width: int, psize: int, stride: int) -> List[Tuple[int, int, int, int]]:
    if stride >= psize:
        raise ValueError(f'stride ({stride}) must be smaller than psize ({psize}) to create overlap.')
    ys = list(range(0, height - psize + 1, stride))
    xs = list(range(0, width - psize + 1, stride))
    return [(y, y + psize, x, x + psize) for y in ys for x in xs]


def generate_neighbor_pairs(
    height: int,
    width: int,
    psize: int,
    stride: int,
    pair_mode: str,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    boxes = generate_patch_boxes(height, width, psize, stride)
    by_origin = {(y0, x0): (y0, y1, x0, x1) for (y0, y1, x0, x1) in boxes}
    pairs = []
    for (y0, y1, x0, x1) in boxes:
        if pair_mode in ('horizontal', 'both'):
            right = by_origin.get((y0, x0 + stride))
            if right is not None:
                pairs.append(((y0, y1, x0, x1), right))
        if pair_mode in ('vertical', 'both'):
            down = by_origin.get((y0 + stride, x0))
            if down is not None:
                pairs.append(((y0, y1, x0, x1), down))
    return pairs


def overlap_local_slices(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> Tuple[Tuple[slice, slice], Tuple[slice, slice], Tuple[int, int]]:
    ay0, ay1, ax0, ax1 = box_a
    by0, by1, bx0, bx1 = box_b
    oy0, oy1 = max(ay0, by0), min(ay1, by1)
    ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
    if oy0 >= oy1 or ox0 >= ox1:
        raise ValueError('Patch pair does not overlap.')

    a_slices = (slice(oy0 - ay0, oy1 - ay0), slice(ox0 - ax0, ox1 - ax0))
    b_slices = (slice(oy0 - by0, oy1 - by0), slice(ox0 - bx0, ox1 - bx0))
    return a_slices, b_slices, (oy1 - oy0, ox1 - ox0)


def crop_patch(x: torch.Tensor, box: Tuple[int, int, int, int]) -> torch.Tensor:
    y0, y1, x0, x1 = box
    return x[:, :, y0:y1, x0:x1]


def make_noise_like(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def rel_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    diff = (a - b).reshape(-1)
    ref = (0.5 * (a + b)).reshape(-1)
    return (torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(ref) + eps)).item()


def summarize_condition(rows: List[Dict], key: str) -> Dict[str, float]:
    values = [float(row[key]) for row in rows]
    if len(values) == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'count': 0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {'mean': mean, 'std': std, 'count': len(values)}


def save_heatmap_figure(
    out_path: Path,
    base_a: torch.Tensor,
    base_b: torch.Tensor,
    denoise_err: torch.Tensor,
    score_err: torch.Tensor,
    title: str,
) -> None:
    base_a_np = torch.sqrt(torch.sum(base_a ** 2, dim=0)).detach().cpu().numpy()
    base_b_np = torch.sqrt(torch.sum(base_b ** 2, dim=0)).detach().cpu().numpy()
    denoise_np = denoise_err.detach().cpu().numpy()
    score_np = score_err.detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(base_a_np, cmap='gray')
    ax1.set_title('Patch A overlap')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(base_b_np, cmap='gray')
    ax2.set_title('Patch B overlap')
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.imshow(denoise_np)
    ax3.set_title('Denoise diff')
    ax3.axis('off')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(1, 4, 4)
    im4 = ax4.imshow(score_np)
    ax4.set_title('Score diff')
    ax4.axis('off')
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / 'maps'
    sigma_pairs = parse_sigma_pairs(args.sigma_pairs)
    sample_indices = resolve_sample_indices(args.val_dir, args.sample_indices, args.num_samples)

    model = load_model(args.model_path, device)
    latents_pos = build_latents_pos(args.image_size, args.pad, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    all_rows: List[Dict] = []
    saved_maps = 0

    for sample_idx in sample_indices:
        print(f'Processing sample_{sample_idx}.pt')
        base_cplx = load_base_image(
            val_dir=args.val_dir,
            sample_idx=sample_idx,
            mask_select=args.mask_select,
            input_source=args.input_source,
            pad=args.pad,
            device=device,
        )
        base_real = complex_to_two_channel(base_cplx)
        _, _, height, width = base_real.shape

        candidate_pairs = generate_neighbor_pairs(
            height=height,
            width=width,
            psize=args.psize,
            stride=args.stride,
            pair_mode=args.pair_mode,
        )
        if len(candidate_pairs) == 0:
            raise RuntimeError('No overlapping patch pairs were generated. Check psize / stride / pair_mode.')

        if len(candidate_pairs) > args.max_pairs_per_sample:
            chosen_ids = random.sample(range(len(candidate_pairs)), args.max_pairs_per_sample)
            selected_pairs = [candidate_pairs[i] for i in sorted(chosen_ids)]
        else:
            selected_pairs = candidate_pairs

        for pair_id, (box_a, box_b) in enumerate(selected_pairs):
            patch_a = crop_patch(base_real, box_a)[0]
            patch_b = crop_patch(base_real, box_b)[0]
            pos_a = crop_patch(latents_pos, box_a)[0]
            pos_b = crop_patch(latents_pos, box_b)[0]
            (a_sy, a_sx), (b_sy, b_sx), (overlap_h, overlap_w) = overlap_local_slices(box_a, box_b)

            for repeat_idx in range(args.repeats):
                for sigma_a, sigma_b in sigma_pairs:
                    eps_a = make_noise_like(patch_a, generator)
                    eps_b = eps_a.clone() if args.share_noise else make_noise_like(patch_b, generator)

                    noisy_a = patch_a + sigma_a * eps_a
                    noisy_b = patch_b + sigma_b * eps_b

                    x_in = torch.stack([noisy_a, noisy_b], dim=0)
                    pos_in = torch.stack([pos_a, pos_b], dim=0)
                    sigma_in = torch.tensor([sigma_a, sigma_b], dtype=torch.float32, device=device)

                    with torch.no_grad():
                        denoised = model(x_in, sigma_in, pos_in, None).to(torch.float32)

                    den_a = denoised[0]
                    den_b = denoised[1]
                    score_a = (den_a - noisy_a) / (sigma_a ** 2)
                    score_b = (den_b - noisy_b) / (sigma_b ** 2)

                    den_overlap_a = den_a[:, a_sy, a_sx]
                    den_overlap_b = den_b[:, b_sy, b_sx]
                    score_overlap_a = score_a[:, a_sy, a_sx]
                    score_overlap_b = score_b[:, b_sy, b_sx]
                    base_overlap_a = patch_a[:, a_sy, a_sx]
                    base_overlap_b = patch_b[:, b_sy, b_sx]

                    denoise_err_map = torch.sqrt(torch.sum((den_overlap_a - den_overlap_b) ** 2, dim=0))
                    score_err_map = torch.sqrt(torch.sum((score_overlap_a - score_overlap_b) ** 2, dim=0))

                    row = {
                        'sample_idx': sample_idx,
                        'pair_id': pair_id,
                        'repeat_idx': repeat_idx,
                        'condition': f'{sigma_a:.6f}:{sigma_b:.6f}',
                        'sigma_a': sigma_a,
                        'sigma_b': sigma_b,
                        'same_sigma': int(abs(sigma_a - sigma_b) < 1e-12),
                        'overlap_h': overlap_h,
                        'overlap_w': overlap_w,
                        'box_a': list(box_a),
                        'box_b': list(box_b),
                        'denoise_rmse': rmse(den_overlap_a, den_overlap_b),
                        'denoise_rel_l2': rel_l2(den_overlap_a, den_overlap_b),
                        'score_rmse': rmse(score_overlap_a, score_overlap_b),
                        'score_rel_l2': rel_l2(score_overlap_a, score_overlap_b),
                    }
                    all_rows.append(row)

                    if saved_maps < args.save_maps:
                        title = (
                            f'sample={sample_idx} pair={pair_id} rep={repeat_idx} '
                            f'sigma=({sigma_a:.4f},{sigma_b:.4f})'
                        )
                        out_path = maps_dir / f's{sample_idx:03d}_p{pair_id:03d}_r{repeat_idx:02d}_{sigma_a:.4f}_{sigma_b:.4f}.png'
                        save_heatmap_figure(
                            out_path=out_path,
                            base_a=base_overlap_a,
                            base_b=base_overlap_b,
                            denoise_err=denoise_err_map,
                            score_err=score_err_map,
                            title=title,
                        )
                        saved_maps += 1

    csv_path = out_dir / 'overlap_consistency_metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'sample_idx', 'pair_id', 'repeat_idx', 'condition', 'sigma_a', 'sigma_b', 'same_sigma',
                'overlap_h', 'overlap_w', 'box_a', 'box_b',
                'denoise_rmse', 'denoise_rel_l2', 'score_rmse', 'score_rel_l2'
            ]
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row['condition']].append(row)

    summary = {
        'config': {
            'model_path': args.model_path,
            'val_dir': args.val_dir,
            'sample_indices': sample_indices,
            'input_source': args.input_source,
            'mask_select': args.mask_select,
            'image_size': args.image_size,
            'pad': args.pad,
            'psize': args.psize,
            'stride': args.stride,
            'pair_mode': args.pair_mode,
            'max_pairs_per_sample': args.max_pairs_per_sample,
            'repeats': args.repeats,
            'sigma_pairs': sigma_pairs,
            'share_noise': bool(args.share_noise),
            'seed': args.seed,
        },
        'num_rows': len(all_rows),
        'conditions': {},
    }

    for condition, rows in grouped.items():
        summary['conditions'][condition] = {
            'denoise_rmse': summarize_condition(rows, 'denoise_rmse'),
            'denoise_rel_l2': summarize_condition(rows, 'denoise_rel_l2'),
            'score_rmse': summarize_condition(rows, 'score_rmse'),
            'score_rel_l2': summarize_condition(rows, 'score_rel_l2'),
        }

    json_path = out_dir / 'overlap_consistency_summary.json'
    with json_path.open('w') as f:
        json.dump(summary, f, indent=2)

    print('\nSaved:')
    print(f'  CSV:   {csv_path}')
    print(f'  JSON:  {json_path}')
    if saved_maps > 0:
        print(f'  MAPS:  {maps_dir}')

    print('\nCondition summary:')
    for condition, stats in summary['conditions'].items():
        d_rel = stats['denoise_rel_l2']['mean']
        s_rel = stats['score_rel_l2']['mean']
        print(f'  {condition} -> denoise_rel_l2={d_rel:.6f}, score_rel_l2={s_rel:.6f}')


if __name__ == '__main__':
    main()
