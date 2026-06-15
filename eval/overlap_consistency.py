import os, sys, csv, json, pickle, random, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
# Put this file in PaDIS-MRI/eval/.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dnnlib
from utils import fftmod
from inverse_operators import MRI_utils
from denoise_padding import denoisedFromPatches, getIndices
METRICS = [
    'input_overlap_rmse', 'state_rms', 'denoise_output_rms', 'denoise_rmse',
    'denoise_rel_l2', 'score_rmse', 'score_rel_l2', 'boundary_residual_mean',
    'interior_residual_mean', 'boundary_interior_ratio'
]
INT_FIELDS = {'sample_idx', 'outer_steps', 'updates', 'pair_id', 'repeat'}
FLOAT_FIELDS = {
    'native_sigma', 'probe_sigma', 'input_overlap_rmse', 'state_rms',
    'denoise_output_rms', 'denoise_rmse', 'denoise_rel_l2', 'score_rmse',
    'score_rel_l2', 'boundary_residual_mean', 'interior_residual_mean',
    'boundary_interior_ratio'
}
def parse_args():
    p = argparse.ArgumentParser(description='Overlap consistency along the real PaDIS-MRI dps2 trajectory.')
    p.add_argument('--model_path', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--sample_indices', default='all', help='Comma-separated ids, or "all".')
    p.add_argument('--map_sample_indices', default='18', help='Samples used for residual maps; use "none" to disable.')
    p.add_argument('--mask_select', type=int, default=7)
    p.add_argument('--image_size', type=int, default=384)
    p.add_argument('--pad', type=int, default=64)
    p.add_argument('--psize', type=int, default=64)
    p.add_argument('--num_steps', type=int, default=104)
    p.add_argument('--inner_loops', type=int, default=10)
    p.add_argument('--sigma_min', type=float, default=0.003)
    p.add_argument('--sigma_max', type=float, default=10.0)
    p.add_argument('--rho', type=float, default=7.0)
    p.add_argument('--zeta', type=float, default=3.0)
    p.add_argument('--checkpoint_steps', default='0,52,78,104')
    p.add_argument('--probe_sigma', type=float, default=0.1)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--num_pairs', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--save_maps_per_step', type=int, default=4)
    p.add_argument('--boundary_width', type=int, default=4)
    p.add_argument('--residual_percentile', type=float, default=99.5)
    p.add_argument('--resume', action='store_true', help='Skip samples with complete per-sample CSV files.')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()
def parse_ints(text):
    return sorted({int(x.strip()) for x in text.split(',') if x.strip()})
def discover_samples(val_dir):
    ids = []
    for path in Path(val_dir).glob('sample_*.pt'):
        try:
            ids.append(int(path.stem.split('_')[-1]))
        except ValueError:
            pass
    return sorted(set(ids))
def resolve_samples(text, val_dir):
    available = discover_samples(val_dir)
    if not available:
        raise FileNotFoundError(f'No sample_*.pt files found in {val_dir}')
    if text.strip().lower() == 'all':
        return available
    requested = parse_ints(text)
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f'Missing sample files: {missing}')
    return requested
def resolve_map_samples(text, selected_samples):
    text = text.strip().lower()
    if text in {'', 'none'}:
        return set()
    if text == 'all':
        return set(selected_samples)
    return set(parse_ints(text)) & set(selected_samples)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def load_model(path, device):
    print(f'Loading network from "{path}"...')
    with dnnlib.util.open_url(path, verbose=False) as f:
        net = pickle.load(f)['ema']
    return net.to(device).eval()
def build_pos(image_size, pad, device):
    n = image_size + 2 * pad
    axis = torch.linspace(-1, 1, n, device=device)
    return torch.stack([
        axis.view(1, -1).repeat(n, 1),
        axis.view(-1, 1).repeat(1, n),
    ], dim=0).unsqueeze(0)
def load_sample(val_dir, idx, mask_select, device):
    path = os.path.join(val_dir, f'sample_{idx}.pt')
    data = torch.load(path, map_location=device, weights_only=False)
    mask_key = f'mask_{mask_select}'
    if mask_key not in data:
        raise KeyError(f'{mask_key} not found in {path}')
    s_maps = fftmod(data['s_map'].clone())[None].to(device)
    full_ksp = fftmod(data['ksp'].clone())[None].to(device)
    mask = data[mask_key][None].to(device)
    return mask * full_ksp, MRI_utils(mask=mask, maps=s_maps)
def to_2ch(x):
    return torch.view_as_real(x.squeeze(1).contiguous()).permute(0, 3, 1, 2).contiguous()
def make_schedule(net, args, device):
    i = torch.arange(args.num_steps, dtype=torch.float64, device=device)
    t = (
        args.sigma_max ** (1 / args.rho)
        + i / (args.num_steps - 1)
        * (args.sigma_min ** (1 / args.rho) - args.sigma_max ** (1 / args.rho))
    ) ** args.rho
    return torch.cat([net.round_sigma(t), torch.zeros(1, dtype=torch.float64, device=device)])
def run_trajectory(net, measurement, operator, pos, checkpoints, args, device):
    """Run the original dps2 process and capture selected trajectory states."""
    patches = args.image_size // args.psize + 1
    spaced = np.linspace(0, (patches - 1) * args.psize, patches, dtype=int)
    x = operator.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x, (args.pad,) * 4, mode='constant', value=0)
    t_steps = make_schedule(net, args, device)
    captured = {}
    if 0 in checkpoints:
        captured[0] = {'x': x.detach().clone(), 'updates': 0, 'native_sigma': t_steps[0].item()}
    for outer in tqdm.tqdm(range(max(checkpoints)), desc='Real dps2 trajectory'):
        t_cur = t_steps[outer].float()
        alpha = 0.5 * t_cur ** 2
        for _ in range(args.inner_loops):
            indices = getIndices(spaced, patches, args.pad, args.psize)
            x = x.detach().requires_grad_(True)
            x_noisy = x + t_cur * torch.randn_like(x)
            den_2ch = denoisedFromPatches(
                net, to_2ch(x_noisy), t_cur, pos, None, indices, t_goal=0, wrong=False
            )
            den_cplx = torch.complex(den_2ch[:, 0], den_2ch[:, 1]).unsqueeze(1)
            score = (den_cplx - x_noisy) / (t_cur ** 2)
            x0_hat = den_2ch if args.pad == 0 else den_2ch[
                :, :, args.pad:args.pad + args.image_size, args.pad:args.pad + args.image_size
            ]
            residual = measurement - operator.forward(x0_hat)
            residual = residual.reshape(residual.shape[0], -1)
            sse_each = torch.linalg.vector_norm(residual, dim=-1) ** 2
            grad = torch.autograd.grad(sse_each.sum(), x)[0]
            x = x - (args.zeta / torch.sqrt(sse_each)[:, None, None, None]) * grad
            if outer < args.num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * torch.randn_like(x)
            else:
                x = x + (alpha / 2) * score
        completed = outer + 1
        if completed in checkpoints:
            captured[completed] = {
                'x': x.detach().clone(),
                'updates': completed * args.inner_loops,
                'native_sigma': t_steps[completed].item() if completed < args.num_steps else 0.0,
            }
    missing = sorted(set(checkpoints) - set(captured))
    if missing:
        raise RuntimeError(f'Failed to capture steps: {missing}')
    return captured
def overlap_slices(a, b):
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    oy0, oy1 = max(ay0, by0), min(ay1, by1)
    ox0, ox1 = max(ax0, bx0), min(ax1, bx1)
    if oy0 >= oy1 or ox0 >= ox1:
        raise ValueError('Patch pair does not overlap.')
    sa = (slice(oy0 - ay0, oy1 - ay0), slice(ox0 - ax0, ox1 - ax0))
    sb = (slice(oy0 - by0, oy1 - by0), slice(ox0 - bx0, ox1 - bx0))
    return sa, sb
def inside_fov(box, f0, f1):
    y0, y1, x0, x1 = box
    return y0 >= f0 and y1 <= f1 and x0 >= f0 and x1 <= f1
def sample_items(items, n, rng):
    if n >= len(items):
        return list(items)
    ids = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in ids]
def choose_pairs(args):
    """Select spatially fixed pairs, balanced between horizontal and vertical directions."""
    n = args.image_size + 2 * args.pad
    starts = range(0, n - args.psize + 1, args.stride)
    boxes = {(y, x): (y, y + args.psize, x, x + args.psize) for y in starts for x in starts}
    f0, f1 = args.pad, args.pad + args.image_size
    candidates = {'horizontal': [], 'vertical': []}
    for (y, x), box in boxes.items():
        for key, orientation in (((y, x + args.stride), 'horizontal'), ((y + args.stride, x), 'vertical')):
            neighbor = boxes.get(key)
            if neighbor is not None and inside_fov(box, f0, f1) and inside_fov(neighbor, f0, f1):
                candidates[orientation].append((box, neighbor, orientation))
    rng = random.Random(args.seed)
    target_h = args.num_pairs // 2
    target_v = args.num_pairs - target_h
    selected_h = sample_items(candidates['horizontal'], target_h, rng)
    selected_v = sample_items(candidates['vertical'], target_v, rng)
    remaining = args.num_pairs - len(selected_h) - len(selected_v)
    if remaining > 0:
        used = set(selected_h + selected_v)
        pool = [p for p in candidates['horizontal'] + candidates['vertical'] if p not in used]
        extra = sample_items(pool, remaining, rng)
    else:
        extra = []
    horizontal = selected_h + [p for p in extra if p[2] == 'horizontal']
    vertical = selected_v + [p for p in extra if p[2] == 'vertical']
    pairs = []
    for i in range(max(len(horizontal), len(vertical))):
        if i < len(horizontal):
            pairs.append(horizontal[i])
        if i < len(vertical):
            pairs.append(vertical[i])
    if not pairs:
        raise RuntimeError('No valid overlap pairs found.')
    print(f'Pair orientation: horizontal={len(horizontal)}, vertical={len(vertical)}')
    return pairs
def expected_distance_rows_per_sample(pairs, checkpoints, repeats, psize):
    bins = 0
    for box_a, box_b, _ in pairs:
        slice_a, slice_b = overlap_slices(box_a, box_b)
        dist_map = boundary_distance_map(slice_a, slice_b, psize, torch.device('cpu'))
        bins += int(dist_map.max().item()) + 1
    return len(checkpoints) * repeats * bins
def crop(x, box):
    y0, y1, x0, x1 = box
    return x[:, :, y0:y1, x0:x1]
def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()
def rel_l2(a, b, eps=1e-12):
    num = torch.linalg.vector_norm((a - b).reshape(-1))
    den = torch.sqrt(torch.linalg.vector_norm(a.reshape(-1)) ** 2 + torch.linalg.vector_norm(b.reshape(-1)) ** 2)
    return (num / (den + eps)).item()
def rms_energy(a, b=None):
    if b is None:
        return torch.sqrt(torch.mean(a ** 2)).item()
    return torch.sqrt(0.5 * (torch.mean(a ** 2) + torch.mean(b ** 2))).item()
def boundary_distance_map(slice_a, slice_b, psize, device):
    """Distance to the nearest patch boundary, measured in either patch coordinate system."""
    ya = torch.arange(slice_a[0].start, slice_a[0].stop, device=device)
    xa = torch.arange(slice_a[1].start, slice_a[1].stop, device=device)
    yb = torch.arange(slice_b[0].start, slice_b[0].stop, device=device)
    xb = torch.arange(slice_b[1].start, slice_b[1].stop, device=device)
    ya, xa = torch.meshgrid(ya, xa, indexing='ij')
    yb, xb = torch.meshgrid(yb, xb, indexing='ij')
    dist_a = torch.minimum(torch.minimum(ya, psize - 1 - ya), torch.minimum(xa, psize - 1 - xa))
    dist_b = torch.minimum(torch.minimum(yb, psize - 1 - yb), torch.minimum(xb, psize - 1 - xb))
    return torch.minimum(dist_a, dist_b)
def region_mean(image, mask):
    return image[mask].mean().item() if torch.any(mask) else float('nan')
def magnitude(x):
    return torch.sqrt(torch.sum(x ** 2, dim=0)).detach().cpu().numpy()
def probe_state(net, state, pos, pairs, noises, args, sample_idx, save_plots):
    """Probe one captured state; keep all repeat-level measurements."""
    x = state['x']
    state_2ch = to_2ch(x)
    rows, plot_records, distance_rows = [], [], []
    for repeat, noise in enumerate(noises):
        noisy_2ch = to_2ch(x + args.probe_sigma * noise)
        for pair_id, (box_a, box_b, orientation) in enumerate(pairs):
            noisy_a, noisy_b = crop(noisy_2ch, box_a)[0], crop(noisy_2ch, box_b)[0]
            pos_a, pos_b = crop(pos, box_a)[0], crop(pos, box_b)[0]
            (asy, asx), (bsy, bsx) = overlap_slices(box_a, box_b)
            input_a, input_b = noisy_a[:, asy, asx], noisy_b[:, bsy, bsx]
            input_rmse = rmse(input_a, input_b)
            if input_rmse > 1e-6:
                raise RuntimeError(f'Overlap input misalignment: {input_rmse:.3e}')
            sigma = torch.full((2,), args.probe_sigma, dtype=torch.float32, device=x.device)
            with torch.no_grad():
                den = net(torch.stack([noisy_a, noisy_b]), sigma, torch.stack([pos_a, pos_b]), None).float()
            den_a_full, den_b_full = den[0], den[1]
            score_a_full = (den_a_full - noisy_a) / (args.probe_sigma ** 2)
            score_b_full = (den_b_full - noisy_b) / (args.probe_sigma ** 2)
            den_a, den_b = den_a_full[:, asy, asx], den_b_full[:, bsy, bsx]
            score_a, score_b = score_a_full[:, asy, asx], score_b_full[:, bsy, bsx]
            state_overlap = crop(state_2ch, box_a)[0][:, asy, asx]
            den_res = torch.sqrt(torch.sum((den_a - den_b) ** 2, dim=0))
            score_res = torch.sqrt(torch.sum((score_a - score_b) ** 2, dim=0))
            dist_map = boundary_distance_map((asy, asx), (bsy, bsx), args.psize, x.device)
            boundary_mask = dist_map < args.boundary_width
            boundary_mean = region_mean(den_res, boundary_mask)
            interior_mean = region_mean(den_res, ~boundary_mask)
            rows.append({
                'sample_idx': sample_idx, 'outer_steps': state['outer_steps'], 'updates': state['updates'],
                'native_sigma': state['native_sigma'], 'probe_sigma': args.probe_sigma, 'pair_id': pair_id,
                'orientation': orientation, 'repeat': repeat, 'box_a': list(box_a), 'box_b': list(box_b),
                'input_overlap_rmse': input_rmse, 'state_rms': rms_energy(state_overlap),
                'denoise_output_rms': rms_energy(den_a, den_b), 'denoise_rmse': rmse(den_a, den_b),
                'denoise_rel_l2': rel_l2(den_a, den_b), 'score_rmse': rmse(score_a, score_b),
                'score_rel_l2': rel_l2(score_a, score_b), 'boundary_residual_mean': boundary_mean,
                'interior_residual_mean': interior_mean,
                'boundary_interior_ratio': boundary_mean / (interior_mean + 1e-12),
            })
            for distance in range(int(dist_map.max().item()) + 1):
                mask = dist_map == distance
                distance_rows.append({
                    'sample_idx': sample_idx, 'outer_steps': state['outer_steps'], 'updates': state['updates'],
                    'pair_id': pair_id, 'orientation': orientation, 'repeat': repeat, 'distance': distance,
                    'residual_mean': region_mean(den_res, mask), 'pixel_count': int(mask.sum().item()),
                })
            if save_plots and repeat == 0 and pair_id < args.save_maps_per_step:
                plot_records.append({
                    'sample_idx': sample_idx, 'outer_steps': state['outer_steps'], 'updates': state['updates'],
                    'native_sigma': state['native_sigma'], 'probe_sigma': args.probe_sigma,
                    'pair_id': pair_id, 'orientation': orientation, 'state': magnitude(state_overlap),
                    'den_a': magnitude(den_a), 'den_b': magnitude(den_b),
                    'den_res': den_res.detach().cpu().numpy(),
                    'boundary_mask': boundary_mask.detach().cpu().numpy(),
                })
    return rows, plot_records, distance_rows
def save_maps(records, maps_dir, residual_percentile):
    if not records:
        return
    den_values = np.concatenate([r['den_res'].ravel() for r in records])
    den_vmax = max(float(np.percentile(den_values, residual_percentile)), np.finfo(np.float32).eps)
    with (maps_dir.parent / 'visualization_scale.json').open('w') as f:
        json.dump({'residual_percentile': residual_percentile, 'denoise_residual_vmin': 0.0,
                   'denoise_residual_vmax': den_vmax}, f, indent=2)
    for record in records:
        gray_values = np.concatenate([record['state'].ravel(), record['den_a'].ravel(), record['den_b'].ravel()])
        gray_vmax = max(float(np.percentile(gray_values, 99.5)), np.finfo(np.float32).eps)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        panels = [
            (record['state'], 'State overlap', 'gray', 0.0, gray_vmax),
            (record['den_a'], 'Denoised A', 'gray', 0.0, gray_vmax),
            (record['den_b'], 'Denoised B', 'gray', 0.0, gray_vmax),
            (record['den_res'], 'Denoise residual', 'viridis', 0.0, den_vmax),
        ]
        for i, (image, title, cmap, vmin, vmax) in enumerate(panels):
            im = axes[i].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(title)
            axes[i].axis('off')
            if i == 3:
                axes[i].contour(record['boundary_mask'].astype(float), levels=[0.5], linewidths=0.6)
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        fig.suptitle(
            f'sample={record["sample_idx"]}, updates={record["updates"]}, '
            f'pair={record["pair_id"]} ({record["orientation"]}), '
            f'next sigma={record["native_sigma"]:.6f}, probe sigma={record["probe_sigma"]:.4f}'
        )
        fig.tight_layout()
        path = maps_dir / f's{record["sample_idx"]:03d}_step{record["outer_steps"]:03d}_pair{record["pair_id"]:02d}.png'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
    comparison_dir = maps_dir / 'step_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['sample_idx'], record['pair_id'])].append(record)
    for (sample_idx, pair_id), group in grouped.items():
        group = sorted(group, key=lambda r: r['updates'])
        fig, axes = plt.subplots(1, len(group), figsize=(4 * len(group), 4), squeeze=False)
        for axis, record in zip(axes[0], group):
            im = axis.imshow(record['den_res'], cmap='viridis', vmin=0.0, vmax=den_vmax)
            axis.contour(record['boundary_mask'].astype(float), levels=[0.5], linewidths=0.6)
            axis.set_title(f'{record["updates"]} updates')
            axis.axis('off')
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        fig.suptitle(f'sample={sample_idx}, pair={pair_id} ({group[0]["orientation"]})')
        fig.savefig(comparison_dir / f's{sample_idx:03d}_pair{pair_id:02d}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(f'Unified denoise residual scale: [0, {den_vmax:.6g}]')
def finite_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'mean': None, 'std': None, 'median': None, 'q25': None, 'q75': None,
                'min': None, 'max': None, 'count': 0}
    return {
        'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if values.size > 1 else 0.0,
        'median': float(np.median(values)), 'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)), 'min': float(values.min()),
        'max': float(values.max()), 'count': int(values.size),
    }
def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def read_csv(path, int_fields=INT_FIELDS, float_fields=FLOAT_FIELDS):
    rows = []
    if not path.is_file():
        return rows
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            for key in int_fields & row.keys():
                row[key] = int(row[key])
            for key in float_fields & row.keys():
                row[key] = float(row[key])
            rows.append(row)
    return rows
def average_repeats_by_pair(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['sample_idx'], row['outer_steps'], row['pair_id'])].append(row)
    pair_rows = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        result = {
            'sample_idx': first['sample_idx'], 'outer_steps': first['outer_steps'],
            'updates': first['updates'], 'native_sigma': first['native_sigma'],
            'probe_sigma': first['probe_sigma'], 'pair_id': first['pair_id'],
            'orientation': first['orientation'], 'box_a': first['box_a'], 'box_b': first['box_b'],
            'repeat_count': len(group),
        }
        for metric in METRICS:
            summary = finite_stats([row[metric] for row in group])
            for name in ('mean', 'std', 'median', 'min', 'max'):
                result[f'{metric}_repeat_{name}'] = summary[name]
        pair_rows.append(result)
    return pair_rows
def make_sample_rows(pair_rows):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row['sample_idx'], row['outer_steps'])].append(row)
    sample_rows = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        result = {'sample_idx': first['sample_idx'], 'outer_steps': first['outer_steps'],
                  'updates': first['updates'], 'pair_count': len(group)}
        for metric in METRICS:
            summary = finite_stats([row[f'{metric}_repeat_mean'] for row in group])
            for name in ('mean', 'std', 'median', 'q25', 'q75', 'min', 'max'):
                result[f'{metric}_pair_{name}'] = summary[name]
            repeat_std = finite_stats([row[f'{metric}_repeat_std'] for row in group])
            result[f'{metric}_repeat_std_pair_mean'] = repeat_std['mean']
        sample_rows.append(result)
    return sample_rows
def make_orientation_rows(pair_rows):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row['sample_idx'], row['outer_steps'], row['orientation'])].append(row)
    result_rows = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        result = {'sample_idx': first['sample_idx'], 'outer_steps': first['outer_steps'],
                  'updates': first['updates'], 'orientation': first['orientation'], 'pair_count': len(group)}
        for metric in METRICS:
            result[f'{metric}_pair_mean'] = finite_stats(
                [row[f'{metric}_repeat_mean'] for row in group]
            )['mean']
        result_rows.append(result)
    return result_rows
def aggregate_distance_profiles(rows):
    pair_grouped = defaultdict(list)
    for row in rows:
        key = (int(row['sample_idx']), int(row['outer_steps']), int(row['pair_id']), int(row['distance']))
        pair_grouped[key].append(row)
    pair_rows = []
    for (sample, step, pair, distance), group in sorted(pair_grouped.items()):
        source = group[0]
        values = [float(row['residual_mean']) for row in group]
        pair_rows.append({
            'sample_idx': sample, 'outer_steps': step, 'updates': int(source['updates']),
            'pair_id': pair, 'orientation': source['orientation'], 'distance': distance,
            'residual_repeat_mean': float(np.mean(values)),
            'residual_repeat_std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    sample_grouped = defaultdict(list)
    for row in pair_rows:
        sample_grouped[(row['sample_idx'], row['outer_steps'], row['distance'])].append(row['residual_repeat_mean'])
    sample_rows = []
    for (sample, step, distance), values in sorted(sample_grouped.items()):
        source = next(r for r in pair_rows if r['sample_idx'] == sample and r['outer_steps'] == step and r['distance'] == distance)
        summary = finite_stats(values)
        sample_rows.append({
            'sample_idx': sample, 'outer_steps': step, 'updates': source['updates'], 'distance': distance,
            'residual_pair_mean': summary['mean'], 'residual_pair_median': summary['median'],
            'residual_pair_q25': summary['q25'], 'residual_pair_q75': summary['q75'],
        })
    return pair_rows, sample_rows
def save_trend_plot(sample_rows, out_dir, metric, ylabel, filename):
    grouped = defaultdict(list)
    field = f'{metric}_pair_mean'
    for row in sample_rows:
        grouped[row['updates']].append(row[field])
    updates = sorted(grouped)
    summaries = [finite_stats(grouped[u]) for u in updates]
    fig, ax = plt.subplots(figsize=(7, 5))
    for x, values in grouped.items():
        jitter = np.linspace(-12, 12, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), x) + jitter, values, alpha=0.25, s=12)
    means = [s['mean'] for s in summaries]
    medians = [s['median'] for s in summaries]
    q25 = [s['q25'] for s in summaries]
    q75 = [s['q75'] for s in summaries]
    ax.plot(updates, means, marker='o', label='mean across samples')
    ax.plot(updates, medians, marker='s', label='median across samples')
    ax.fill_between(updates, q25, q75, alpha=0.2, label='sample 25–75%')
    ax.set_xlabel('Completed inner updates')
    ax.set_ylabel(ylabel)
    ax.set_xticks(updates)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
def save_orientation_plot(orientation_rows, out_dir, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    for orientation in ('horizontal', 'vertical'):
        grouped = defaultdict(list)
        field = f'{metric}_pair_mean'
        for row in orientation_rows:
            if row['orientation'] == orientation:
                grouped[row['updates']].append(row[field])
        updates = sorted(grouped)
        means = [finite_stats(grouped[u])['mean'] for u in updates]
        ax.plot(updates, means, marker='o', label=orientation)
    ax.set_xlabel('Completed inner updates')
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({row['updates'] for row in orientation_rows}))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
def save_distance_plot(sample_distance_rows, out_dir):
    grouped = defaultdict(list)
    for row in sample_distance_rows:
        grouped[(row['updates'], row['distance'])].append(row['residual_pair_mean'])
    fig, ax = plt.subplots(figsize=(7, 5))
    for updates in sorted({key[0] for key in grouped}):
        distances = sorted(key[1] for key in grouped if key[0] == updates)
        means = [finite_stats(grouped[(updates, d)])['mean'] for d in distances]
        ax.plot(distances, means, marker='o', label=f'{updates} updates')
    ax.set_xlabel('Distance to nearest patch boundary (pixels)')
    ax.set_ylabel('Mean denoise residual')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'boundary_distance_profile.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
def save_results(raw_rows, distance_rows, args, samples, checkpoints, out_dir):
    save_csv(out_dir / 'overlap_consistency_metrics.csv', raw_rows)
    pair_rows = average_repeats_by_pair(raw_rows)
    sample_rows = make_sample_rows(pair_rows)
    orientation_rows = make_orientation_rows(pair_rows)
    save_csv(out_dir / 'overlap_consistency_pair_summary.csv', pair_rows)
    save_csv(out_dir / 'overlap_consistency_sample_summary.csv', sample_rows)
    save_csv(out_dir / 'overlap_consistency_orientation_summary.csv', orientation_rows)
    distance_pair_rows, distance_sample_rows = aggregate_distance_profiles(distance_rows)
    save_csv(out_dir / 'boundary_distance_profile_pair_summary.csv', distance_pair_rows)
    save_csv(out_dir / 'boundary_distance_profile_sample_summary.csv', distance_sample_rows)
    global_by_step = {}
    for step in checkpoints:
        group = [row for row in sample_rows if row['outer_steps'] == step]
        metric_result = {}
        for metric in METRICS:
            metric_result[metric] = {
                'across_samples_of_pair_mean': finite_stats([r[f'{metric}_pair_mean'] for r in group]),
                'across_samples_of_pair_median': finite_stats([r[f'{metric}_pair_median'] for r in group]),
                'within_pair_repeat_std_across_samples': finite_stats(
                    [r[f'{metric}_repeat_std_pair_mean'] for r in group]
                ),
            }
        global_by_step[f'step_{step}'] = {'updates': group[0]['updates'], 'sample_count': len(group), **metric_result}
    orientation_global = {}
    for step in checkpoints:
        orientation_global[f'step_{step}'] = {}
        for orientation in ('horizontal', 'vertical'):
            group = [r for r in orientation_rows if r['outer_steps'] == step and r['orientation'] == orientation]
            orientation_global[f'step_{step}'][orientation] = {
                metric: finite_stats([r[f'{metric}_pair_mean'] for r in group]) for metric in METRICS
            }
    summary = {
        'config': vars(args), 'sample_indices': samples, 'sample_count': len(samples),
        'checkpoint_steps': checkpoints,
        'statistical_units': {
            'raw_repeat_level': 'Every patch pair and probe-noise repeat is retained.',
            'pair_level': 'Repeats are summarized within each spatial patch pair; mean/std/median/min/max are retained.',
            'global_level': 'Samples, not repeats or pairs, are the independent units for final cross-sample trends.',
        },
        'global_by_step': global_by_step,
        'global_by_step_and_orientation': orientation_global,
    }
    with (out_dir / 'overlap_consistency_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    save_trend_plot(sample_rows, out_dir, 'denoise_rmse', 'Denoise RMSE', 'trend_denoise_rmse.png')
    save_trend_plot(sample_rows, out_dir, 'denoise_rel_l2', 'Denoise relative L2', 'trend_denoise_rel_l2.png')
    save_trend_plot(sample_rows, out_dir, 'denoise_output_rms', 'Denoised output RMS energy', 'trend_denoise_energy.png')
    save_trend_plot(sample_rows, out_dir, 'boundary_interior_ratio', 'Boundary / interior residual ratio', 'trend_boundary_ratio.png')
    save_trend_plot(sample_rows, out_dir, 'denoise_rmse_repeat_std', 'Within-pair repeat SD of denoise RMSE',
                    'trend_noise_sensitivity.png')
    save_orientation_plot(orientation_rows, out_dir, 'denoise_rmse', 'Denoise RMSE',
                          'trend_orientation_denoise_rmse.png')
    save_orientation_plot(orientation_rows, out_dir, 'boundary_interior_ratio', 'Boundary / interior residual ratio',
                          'trend_orientation_boundary_ratio.png')
    save_distance_plot(distance_sample_rows, out_dir)
    print('\nGlobal summary across samples:')
    for step in checkpoints:
        item = global_by_step[f'step_{step}']
        rmse_stats = item['denoise_rmse']['across_samples_of_pair_mean']
        boundary_stats = item['boundary_interior_ratio']['across_samples_of_pair_mean']
        print(
            f'step={step}, updates={item["updates"]}, samples={item["sample_count"]}, '
            f'denoise_rmse={rmse_stats["mean"]:.6f}, '
            f'boundary/interior={boundary_stats["mean"]:.3f}'
        )
def validate(args, checkpoints, samples):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the original denoisedFromPatches().')
    if not samples:
        raise ValueError('No samples selected.')
    if args.image_size % args.psize != 0 or args.pad != args.psize:
        raise ValueError('This script assumes image_size % psize == 0 and pad == psize.')
    if not 0 < args.stride < args.psize:
        raise ValueError('stride must satisfy 0 < stride < psize.')
    if not checkpoints or min(checkpoints) < 0 or max(checkpoints) > args.num_steps:
        raise ValueError(f'checkpoint_steps must be within [0, {args.num_steps}].')
    if args.probe_sigma <= 0 or args.repeats < 1 or args.num_pairs < 2:
        raise ValueError('probe_sigma/repeats must be positive and num_pairs must be at least 2.')
    if not 0 < args.boundary_width < args.psize // 2:
        raise ValueError('boundary_width must satisfy 0 < boundary_width < psize/2.')
    if not 0 < args.residual_percentile <= 100:
        raise ValueError('residual_percentile must be within (0, 100].')
def main():
    args = parse_args()
    checkpoints = parse_ints(args.checkpoint_steps)
    samples = resolve_samples(args.sample_indices, args.val_dir)
    map_samples = resolve_map_samples(args.map_sample_indices, samples)
    validate(args, checkpoints, samples)
    device = torch.device('cuda')
    out_dir = Path(args.out_dir)
    per_sample_dir = out_dir / 'per_sample'
    maps_root = out_dir / 'maps'
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    net = load_model(args.model_path, device)
    pos = build_pos(args.image_size, args.pad, device)
    pairs = choose_pairs(args)
    expected_rows = len(checkpoints) * len(pairs) * args.repeats
    expected_distance_rows = expected_distance_rows_per_sample(pairs, checkpoints, args.repeats, args.psize)
    print(f'Samples: {len(samples)} -> {samples}')
    print(f'Fixed overlap pairs: {len(pairs)}')
    print(f'Captured outer steps: {checkpoints}')
    print(f'Expected rows per sample: metrics={expected_rows}, distance={expected_distance_rows}')
    for order, sample_idx in enumerate(samples, start=1):
        metric_path = per_sample_dir / f'sample_{sample_idx:03d}_metrics.csv'
        distance_path = per_sample_dir / f'sample_{sample_idx:03d}_distance.csv'
        existing = read_csv(metric_path)
        existing_distance = read_csv(
            distance_path,
            int_fields={'sample_idx', 'outer_steps', 'updates', 'pair_id', 'repeat', 'distance', 'pixel_count'},
            float_fields={'residual_mean'}
        )
        if args.resume and len(existing) == expected_rows and len(existing_distance) == expected_distance_rows:
            print(f'[{order}/{len(samples)}] sample_{sample_idx}: complete, skipped by --resume')
            continue
        print(f'\n[{order}/{len(samples)}] Processing sample_{sample_idx}.pt')
        set_seed(args.seed + sample_idx)
        measurement, operator = load_sample(args.val_dir, sample_idx, args.mask_select, device)
        states = run_trajectory(net, measurement, operator, pos, checkpoints, args, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 100000 + sample_idx)
        reference = states[checkpoints[0]]['x']
        noises = [
            torch.randn(reference.shape, dtype=reference.dtype, device=device, generator=generator)
            for _ in range(args.repeats)
        ]
        sample_rows, sample_distance_rows, sample_records = [], [], []
        for step in checkpoints:
            states[step]['outer_steps'] = step
            print(f'Probing step={step}, updates={states[step]["updates"]}')
            rows, records, distance_rows = probe_state(
                net, states[step], pos, pairs, noises, args, sample_idx, sample_idx in map_samples
            )
            sample_rows.extend(rows)
            sample_records.extend(records)
            sample_distance_rows.extend(distance_rows)
        save_csv(metric_path, sample_rows)
        save_csv(distance_path, sample_distance_rows)
        if sample_records:
            sample_maps_dir = maps_root / f'sample_{sample_idx:03d}'
            sample_maps_dir.mkdir(parents=True, exist_ok=True)
            save_maps(sample_records, sample_maps_dir, args.residual_percentile)
        del states, measurement, operator, noises
        torch.cuda.empty_cache()
        print(f'Saved sample_{sample_idx}: {len(sample_rows)} metrics rows.')
    all_rows, all_distance_rows = [], []
    for sample_idx in samples:
        metric_path = per_sample_dir / f'sample_{sample_idx:03d}_metrics.csv'
        distance_path = per_sample_dir / f'sample_{sample_idx:03d}_distance.csv'
        rows = read_csv(metric_path)
        if len(rows) != expected_rows:
            raise RuntimeError(f'{metric_path} has {len(rows)} rows; expected {expected_rows}.')
        all_rows.extend(rows)
        all_distance_rows.extend(read_csv(
            distance_path,
            int_fields={'sample_idx', 'outer_steps', 'updates', 'pair_id', 'repeat', 'distance', 'pixel_count'},
            float_fields={'residual_mean'}
        ))
    save_results(all_rows, all_distance_rows, args, samples, checkpoints, out_dir)
    print(f'\nAll outputs saved to: {out_dir}')
if __name__ == '__main__':
    main()