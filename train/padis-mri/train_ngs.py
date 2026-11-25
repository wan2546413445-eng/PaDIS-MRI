"""NGS-only training entry. Reuse PaDIS CLI and training loop without editing either."""
import json
import importlib.util
import inspect
from pathlib import Path
import os
import click
import train_overlap
from torch_utils import distributed as dist


def main():
    original=train_overlap.training_loop.training_loop
    def ngs_loop(**c):
        n=c['network_kwargs']; l=c['loss_kwargs']
        if n.get('model_type')!='SongUNet' or not str(n.get('class_name','')).endswith('.Patch_EDMPrecond'):
            raise click.ClickException('NGS requires --arch=ddpmpp --backbone=songunet --precond=pedm')
        if not str(l.get('class_name','')).endswith('.IndependentNoiseOverlapPatch_EDMLoss'):
            raise click.ClickException('NGS requires --overlap-mode=independent or same_center')
        if c.get('augment_kwargs') or c.get('train_on_latents'):
            raise click.ClickException('NGS requires no augmentation or latent training')
        if l.get('joint_context') or l.get('global_conditioning'):
            raise click.ClickException('Do not enable other global-conditioning experiments with NGS')
        n['class_name']='training.ngs_networks.NGSEDMPrecond'
        l['class_name']='training.ngs_loss.NGSOverlapLoss'
        l.pop('joint_context',None); l.pop('global_conditioning',None)
        if dist.get_rank()==0:
            with open(os.path.join(c['run_dir'],'training_options.json'),'w') as f:
                json.dump(c,f,indent=2)
            print('[NGS] model and loss:',n['class_name'],l['class_name'],flush=True)
        # Training snapshot generation imports the legacy partition function.
        # Route it to the NGS implementation in this process only.
        preview = original.__globals__.get('dps_uncond')
        if preview is not None:
            path = Path(__file__).resolve().parents[2] / 'eval' / 'denoise_padding_ngs.py'
            spec = importlib.util.spec_from_file_location('denoise_padding_ngs', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # dps_uncond is decorated with @torch.no_grad(); its outer wrapper
            # belongs to torch.utils._contextlib, not the PaDIS preview module.
            # Resolve the original function and patch its actual global binding.
            preview_impl = inspect.unwrap(preview)
            g = getattr(preview_impl, '__globals__', None)
            old = None if g is None else g.get('denoisedFromPatches')
            if not callable(old):
                raise RuntimeError('NGS preview: denoisedFromPatches missing in '
                                   + str(getattr(preview_impl, '__module__', '?')))
            g['denoisedFromPatches'] = module.denoisedFromPatches
            if dist.get_rank() == 0:
                print('[NGS] snapshot preview attached:', preview_impl.__module__, flush=True)
            try:
                return original(**c)
            finally:
                g['denoisedFromPatches'] = old
        return original(**c)
    # This assignment is process-local only; the source files are untouched.
    train_overlap.training_loop.training_loop=ngs_loop
    try:
        train_overlap.main()
    finally:
        train_overlap.training_loop.training_loop=original

if __name__=='__main__':
    main()
