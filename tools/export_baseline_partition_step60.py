#!/usr/bin/env python3
"""Capture the fixed Baseline experiment, or redraw a previously saved .pt."""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_VAL = '/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB'
DEFAULT_MODEL = (
    '/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/baseline_seed123_to20k/brain/32dB/'
    '00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-baseline-pad96-seed123-to20k/'
    'network-snapshot-010200.pkl')
DEFAULT_OUTPUT = '/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/baseline_partition_view'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo_root', default=str(ROOT), help='Current PaDIS-MRI repository root')
    parser.add_argument('--val_dir', default=DEFAULT_VAL)
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--image_size', type=int, default=384, choices=[384])
    parser.add_argument('--pad', type=int, default=64, choices=[64])
    parser.add_argument('--psize', type=int, default=64, choices=[64])
    parser.add_argument('--mask_select', type=int, default=7, choices=[7])
    parser.add_argument('--steps', type=int, default=78, choices=[78])
    parser.add_argument('--inner_loops', type=int, default=10, choices=[10])
    parser.add_argument('--seed', type=int, default=123, choices=[123])
    parser.add_argument('--zeta', type=float, default=3.0, choices=[3.0])
    parser.add_argument('--figure_outer_step', type=int, default=60, choices=[60])
    parser.add_argument('--gpu', type=int, default=0, choices=[0], help='Physical GPU, exposed as cuda:0')
    parser.add_argument('--output_root', default=DEFAULT_OUTPUT, help='Parent for fresh run directories')
    parser.add_argument('--output_dir', default=None, help='Optional exact NEW directory, must not exist')
    parser.add_argument('--input_pt', default=None, help='Offline redraw, never loads a model or runs dps2')
    parser.add_argument('--roi', type=int, nargs=4, default=[144, 144, 96, 96],
                        metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                        help='Zero-based ROI in original 384 FOV; default is geometric center, not error-selected')
    parser.add_argument('--display_indices', type=int, nargs=4, default=[1, 3, 6, 10],
                        help='Four one-based inner-loop numbers for overview/ROI panels')
    parser.add_argument('--magnitude_vmax', type=float, default=None,
                        help='Override shared raw-unit magnitude reference; default=GT magnitude P99; display=clip(abs(I)/reference,0,1)')
    parser.add_argument('--difference_vmax', type=float, default=None,
                        help='Override shared raw-unit D_j/S upper limit; default=joint full-FOV P99 of all ten D_j and S; turbo display clipping only')
    parser.add_argument('--gt_residual', action='store_true',
                        help='Also export separate complex GT residual ROI maps |d_j - gt|')
    parser.add_argument('--capture_only', action='store_true', help='Persist complex tensors without plotting')
    parser.add_argument('--check_sources', action='store_true', help='Read-only source audit, no inference')
    args = parser.parse_args()
    x, y, width, height = args.roi
    if min(x, y) < 0 or min(width, height) <= 0 or x+width > 384 or y+height > 384:
        parser.error('--roi must be a non-empty rectangle inside the original 384x384 FOV')
    if len(set(args.display_indices)) != 4 or any(not 1 <= j <= 10 for j in args.display_indices):
        parser.error('--display_indices needs four distinct values in 1..10')
    for name in ['magnitude_vmax', 'difference_vmax']:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'--{name} must be positive and finite')
    if args.input_pt and args.capture_only:
        parser.error('--capture_only applies to a new capture, not offline redraw')
    return args


def new_output_dir(args):
    if args.output_dir:
        path = Path(args.output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    parent = Path(args.output_root).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    prefix = 'redraw' if args.input_pt else 'sample3_step060'
    path = parent / f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    path.mkdir(exist_ok=False)
    return path


def main():
    args = parse_args()
    if args.check_sources:
        from eval_partition_view.source_guard import check_sources
        import json
        print(json.dumps(check_sources(Path(args.repo_root).resolve()), indent=2))
        return
    if args.input_pt:
        pt_path = Path(args.input_pt).expanduser().resolve()
        if not pt_path.is_file():
            raise FileNotFoundError(pt_path)
    else:
        # Must occur before importing torch or any original PaDIS modules.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        for path in [Path(args.val_dir) / 'sample_3.pt', Path(args.model_path)]:
            if not path.is_file():
                raise FileNotFoundError(path)
    output_dir = new_output_dir(args)
    print(f'[output] New independent directory: {output_dir}', flush=True)
    if not args.input_pt:
        from eval_partition_view.capture import run_capture
        pt_path = run_capture(args, output_dir)
    if not args.capture_only:
        from eval_partition_view.visualize import export_from_pt
        export_from_pt(pt_path, output_dir, args)
    print(f'[done] {output_dir}', flush=True)


if __name__ == '__main__':
    main()
