"""Observe original dps2 calls without copying or changing its algorithm."""

import csv
import importlib
import json
import os
import pickle
import sys
from contextlib import AbstractContextManager
from pathlib import Path


class WithinStepCapture(AbstractContextManager):
    """Temporarily wrap only the two callable references used by dps2.

    Both original functions are called once, with the original arguments. The
    denoiser return object is returned unchanged, with its autograd graph intact.
    Observations use detached copies only. No RNG is sampled by these wrappers.
    """

    def __init__(self, recon_module, outer_step, inner_loops, image_size, pad,
                 psize, clean, output_dir, metadata):
        self.module = recon_module
        self.original_dps2 = recon_module.dps2
        self.original_indices = recon_module.getIndices
        self.original_denoise = recon_module.denoisedFromPatches
        self.outer_step = outer_step
        self.inner_loops = inner_loops
        self.image_size = image_size
        self.pad = pad
        self.psize = psize
        self.clean = clean.detach().cpu().clone()
        self.output_dir = Path(output_dir)
        self.metadata = metadata
        self.partition_calls = 0
        self.denoise_calls = 0
        self.partitions = {}
        self.records = []
        self.saved_path = None

    def _coordinates(self, caller):
        if caller.f_code is not self.original_dps2.__code__:
            raise RuntimeError('Capture received a call outside original dps2.')
        return int(caller.f_locals['i']) + 1, int(caller.f_locals['j']) + 1

    def __enter__(self):
        import torch

        def get_indices(*args, **kwargs):
            caller = sys._getframe(1)
            outer, inner = self._coordinates(caller)
            indices = self.original_indices(*args, **kwargs)
            self.partition_calls += 1
            expected = (outer - 1) * self.inner_loops + inner
            if self.partition_calls != expected:
                raise RuntimeError('Unexpected partition call order in dps2.')
            if outer == self.outer_step:
                # Copy the actual returned indices, never generate another grid.
                self.partitions[inner] = torch.tensor(
                    indices, dtype=torch.int64, device='cpu')
            return indices

        def denoise(*args, **kwargs):
            caller = sys._getframe(1)
            outer, inner = self._coordinates(caller)
            result = self.original_denoise(*args, **kwargs)
            self.denoise_calls += 1
            expected = (outer - 1) * self.inner_loops + inner
            if self.denoise_calls != expected or self.partition_calls != expected:
                raise RuntimeError('Unexpected denoiser/partition call order.')
            if outer == self.outer_step:
                if inner != len(self.records) + 1 or inner not in self.partitions:
                    raise RuntimeError('Target inner loops are not consecutive.')
                actual_indices = kwargs.get('indices', args[5] if len(args) > 5 else None)
                actual_tensor = torch.tensor(actual_indices, dtype=torch.int64)
                if not torch.equal(actual_tensor, self.partitions[inner]):
                    raise RuntimeError('Denoiser indices differ from getIndices return.')
                expected_size = self.image_size + 2 * self.pad
                if tuple(result.shape) != (1, 2, expected_size, expected_size):
                    raise RuntimeError(f'Unexpected raw denoiser shape: {result.shape}')
                with torch.no_grad():
                    # Same channel conversion as dps2, followed by its FOV crop.
                    complex_result = torch.complex(result[:, 0], result[:, 1])
                    start = self.pad
                    end = self.pad + self.image_size
                    full_fov = complex_result[0, start:end, start:end].cpu().clone()
                    sigma = caller.f_locals['t_cur'].detach().cpu().clone()
                self.records.append({'estimate': full_fov, 'sigma': sigma,
                                     'inner_loop': inner, 'call_number': expected})
                if inner == self.inner_loops:
                    self._save()  # Persist now, before continuing steps 61..78.
            return result  # Exact original return object, not the detached copy.

        self.module.getIndices = get_indices
        self.module.denoisedFromPatches = denoise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.getIndices = self.original_indices
        self.module.denoisedFromPatches = self.original_denoise
        return False

    def _save(self):
        import torch

        if self.saved_path is not None:
            raise RuntimeError('Target capture was already saved.')
        estimates = torch.stack([r['estimate'] for r in self.records])
        sigmas = torch.stack([r['sigma'] for r in self.records])
        indices = torch.stack([self.partitions[j] for j in range(1, self.inner_loops + 1)])
        gt = self.clean.squeeze(0).squeeze(0)
        if tuple(estimates.shape) != (self.inner_loops, self.image_size, self.image_size):
            raise RuntimeError('Unexpected captured estimate dimensions.')
        if not torch.is_complex(estimates) or not torch.isfinite(estimates).all():
            raise RuntimeError('Captured estimates must be finite complex tensors.')
        if tuple(gt.shape) != (self.image_size, self.image_size) or not torch.isfinite(gt).all():
            raise RuntimeError('GT must match the original FOV and be finite.')
        if not torch.equal(sigmas, sigmas[0].expand_as(sigmas)):
            raise RuntimeError('Target observations do not share one real sigma.')
        expected_patches = (self.image_size // self.psize + 1) ** 2
        if tuple(indices.shape) != (self.inner_loops, expected_patches, 4):
            raise RuntimeError('Unexpected partition shape.')
        offsets = indices[:, 0, [0, 2]].clone()
        metadata = dict(self.metadata)
        metadata.update({
            'schema': 'padis_baseline_within_step_v1',
            'outer_step_numbering': 'one-based, i=59 is outer step 60',
            'inner_loop_numbering': 'one-based, j=0..9 is inner loop 1..10',
            'captured_shape': list(estimates.shape),
            'captured_dtype': str(estimates.dtype),
            'gt_dtype': str(gt.dtype),
            'raw_return_shape': [1, 2, self.image_size + 2*self.pad,
                                 self.image_size + 2*self.pad],
            'fov_crop': [self.pad, self.pad + self.image_size,
                         self.pad, self.pad + self.image_size],
            'tensor_source': 'eval/recon.py::dps2 -> original denoisedFromPatches return',
            'capture_point': 'immediately after return, before score/DC/diffusion update',
            'capture_complete': True,
            'sigma_actual_float32': float(sigmas[0].item()),
            'partition_offsets_padded_row_col': offsets.tolist(),
            'partition_indices_coordinates': 'padded-FOV [row_start,row_end,col_start,col_end]',
            'denoiser_call_numbers': [r['call_number'] for r in self.records],
            'capture_scope': 'actual consecutive inner loops, changing state and noise',
            'mean_reference_usage': 'offline plotting only, never used by baseline',
        })
        payload = {
            'schema': metadata['schema'],
            'denoised_complex': estimates,
            'gt': gt,
            'patch_indices': indices,
            'partition_offsets': offsets,
            'sigmas': sigmas,
            'inner_loop_numbers': torch.arange(1, self.inner_loops + 1),
            'outer_step': self.outer_step,
            'metadata': metadata,
        }
        target = self.output_dir / f'baseline_sample3_step{self.outer_step:03d}.pt'
        # A new run directory is mandatory. Exclusive open also protects .pt.
        with target.open('xb') as handle:
            torch.save(payload, handle)
        with (self.output_dir / 'capture_metadata.json').open('x', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        with (self.output_dir / 'partitions.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['outer_step', 'inner_loop', 'sigma', 'offset_row',
                             'offset_col', 'patch_count', 'denoiser_call_number'])
            for r, off in zip(self.records, offsets.tolist()):
                writer.writerow([self.outer_step, r['inner_loop'], float(r['sigma']),
                                 *off, len(indices[0]), r['call_number']])
        self.saved_path = target
        print(f'[capture] Saved all {self.inner_loops} real full-FOV estimates: {target}', flush=True)

    def validate_completed_baseline(self, num_steps):
        expected = num_steps * self.inner_loops
        if self.partition_calls != expected or self.denoise_calls != expected:
            raise RuntimeError(f'Expected {expected} original calls, got '
                               f'{self.partition_calls}/{self.denoise_calls}.')
        if self.saved_path is None or len(self.records) != self.inner_loops:
            raise RuntimeError('Target outer step was not captured completely.')


def run_capture(args, output_dir):
    """Follow the original fixed-seed evaluator's single-sample execution order."""
    import torch

    from .source_guard import check_sources
    root = Path(args.repo_root).resolve()
    provenance = check_sources(root)
    sample_path = Path(args.val_dir) / 'sample_3.pt'
    model_path = Path(args.model_path)
    if not sample_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f'Required sample/checkpoint unavailable: {sample_path} / {model_path}')
    if not torch.cuda.is_available():
        raise RuntimeError('Real Baseline capture requires CUDA and the original PaDIS environment.')
    if os.environ.get('PADIS_PATCH_BATCH', '49') != '49':
        raise ValueError('Unset PADIS_PATCH_BATCH or use 49 to preserve the standard network call.')
    if os.environ.get('PADIS_PATCH_CHECKPOINT', '0') != '0':
        raise ValueError('Unset PADIS_PATCH_CHECKPOINT or use 0 for this Baseline capture.')
    # New process paths. No files under eval/ are written or replaced.
    for path in [root / 'train' / 'padis-mri', root, root / 'eval']:
        sys.path.insert(0, str(path))
    # Same imports as eval/run.py, including the fixed-seed implementation.
    import dnnlib
    import evaluator
    import evaluator_fixed_seed123
    baseline = importlib.import_module('recon')
    for module, rel in [(baseline, 'eval/recon.py'), (evaluator, 'eval/evaluator.py'),
                        (evaluator_fixed_seed123, 'eval/evaluator_fixed_seed123.py'),
                        (importlib.import_module('denoise_padding'), 'eval/denoise_padding.py')]:
        if Path(module.__file__).resolve() != root / rel:
            raise RuntimeError(f'Unexpected import path for {module.__name__}: {module.__file__}')
    if evaluator_fixed_seed123.dps2 is not baseline.dps2:
        raise RuntimeError('Fixed-seed evaluator does not refer to original Baseline dps2.')
    torch.cuda.set_device(0)
    with dnnlib.util.open_url(str(model_path), verbose=False) as handle:
        model = pickle.load(handle)['ema']
    model = model.to(torch.device('cuda')).eval()
    opt = evaluator_fixed_seed123.DPSHyperEvaluator(
        model=model, mask_select=args.mask_select, val_dir=args.val_dir,
        image_size=args.image_size, pad=args.pad, psize=args.psize,
        val_count=1, seed=args.seed, sample_indices=[3])
    # Identical ordering to fixed_seed123.evaluate's serial worker.
    opt.model = opt.model.cuda(0)
    opt.latents_pos = opt.latents_pos.to(0)
    sample_seed = opt._set_sample_seed(idx=3, gpu_id=0)
    opt._reset_sample_latent(gpu_id=0)
    measurement, gt, inverseop = opt._load_sample(3)
    measurement = measurement.cuda(0)
    gt = gt.cuda(0)
    inverseop.maps = inverseop.maps.cuda(0)
    inverseop.mask = inverseop.mask.cuda(0)
    metadata = {
        'figure_title': 'Within-step variation of full-FOV denoised estimates',
        'algorithm': 'original PaDIS Baseline dps2', 'lgfc_enabled': False,
        'val_dir': str(Path(args.val_dir).resolve()), 'sample_file': str(sample_path.resolve()),
        'sample_index': 3, 'model_path': str(model_path.resolve()),
        'model_weight': 'ema network loaded from checkpoint (not LGFC reference EMA)',
        'image_size': args.image_size, 'pad': args.pad, 'psize': args.psize,
        'steps': args.steps, 'inner_loops': args.inner_loops,
        'figure_outer_step': args.figure_outer_step, 'mask_select': args.mask_select,
        'seed': sample_seed, 'rng_policy': 'fixed_per_sample', 'zeta': args.zeta,
        'sigma_min': 0.003, 'sigma_max': 10.0, 'rho': 7.0,
        'sigma_schedule': 'original _ve_base_schedule and model.round_sigma',
        'physical_gpu': args.gpu, 'logical_gpu': 0,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'data_transform': 'original _load_sample: gt unchanged, fftmod(s_map/ksp), mask_7',
        'normalization': 'raw reconstruction tensors, no extra normalization or phase alignment',
        'source_provenance': provenance,
    }
    observer = WithinStepCapture(baseline, args.figure_outer_step, args.inner_loops,
                                 args.image_size, args.pad, args.psize, gt, output_dir, metadata)
    completed = False
    failure = None
    try:
        with observer:
            opt.dps2_wrapper(inverse_op=inverseop, measurement=measurement, clean=gt,
                             zeta=args.zeta, pad=args.pad, psize=args.psize,
                             num_steps=args.steps, inner_loops=args.inner_loops,
                             save_dir=None, tag='3_patch', save_intermediate=False)
        observer.validate_completed_baseline(args.steps)
        completed = True
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        status = {'baseline_completed': completed, 'configured_steps': args.steps,
                  'partition_calls': observer.partition_calls, 'denoiser_calls': observer.denoise_calls,
                  'capture_saved': observer.saved_path is not None,
                  'failure': failure, 'source_hooks_restored':
                  baseline.getIndices is observer.original_indices and
                  baseline.denoisedFromPatches is observer.original_denoise}
        with (Path(output_dir) / 'run_status.json').open('x', encoding='utf-8') as handle:
            json.dump(status, handle, indent=2)
    return observer.saved_path
