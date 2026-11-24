#!/usr/bin/env python3
"""Patch existing PaDIS-MRI (main-WSY or GLF overlay) without replacing files.

Usage: python tools/install_joint_ct.py --root /path/to/PaDIS-MRI-main
Backups made with .pre_joint_ct.bak suffix; refuses unexpected source layouts.
"""
import argparse
from pathlib import Path
import re
import shutil

ROOT_THIS = Path(__file__).resolve().parent.parent


def once(s, a, b, where):
    n=s.count(a)
    if n!=1:
        raise RuntimeError(f'{where}: expected 1 source anchor, got {n}: {a[:100]!r}')
    return s.replace(a,b,1)


def patch_train(s):
    p='train/padis-mri/train_overlap.py'
    s=once(s, "@click.option('--lambda-overlap'", "@click.option('--joint-ct-context', type=bool, default=False, show_default=True,\n              help='Joint noisy local patch + downsampled noisy full-field condition')\n@click.option('--lambda-overlap'",p)
    s=once(s, '    opts = dnnlib.EasyDict(kwargs)\n', '''    opts = dnnlib.EasyDict(kwargs)
    if opts.joint_ct_context and (
        opts.backbone != 'songunet' or opts.precond != 'pedm'
        or opts.overlap_mode != 'independent'
    ):
        raise click.ClickException(
            '--joint-ct-context requires --backbone=songunet '
            '--precond=pedm --overlap-mode=independent')
''',p)
    s=once(s, "        c.network_kwargs.class_name = 'training.networks.Patch_EDMPrecond'", "        c.network_kwargs.class_name = ('training.joint_context.JointContextEDMPrecond'\n                                      if opts.joint_ct_context else 'training.networks.Patch_EDMPrecond')",p)
    # This occurs after *all* three overlap-mode branches, so independent loss
    # has already been set. No changes to the original multi-scale probability.
    anchor="    # Network options.\n"
    s=once(s,anchor,"    if opts.joint_ct_context:\n        c.loss_kwargs.joint_context = True\n\n"+anchor,p)
    return s


def patch_loss(s):
    p='train/padis-mri/training/patch_overlap_independent_loss.py'
    s=once(s, "            pair_geometry='shifted', spatial_weight_mode='distance',", "            pair_geometry='shifted', spatial_weight_mode='distance',\n            joint_context=False,",p)
    s=once(s, "        self.spatial_weight_mode = spatial_weight_mode", "        self.spatial_weight_mode = spatial_weight_mode\n        self.joint_context = bool(joint_context)\n        if self.joint_context and self.noise_mode != 'independent':\n            raise ValueError('Joint CT conditioning requires independent view noise')",p)
    # Keep the original noise path byte-identical for baseline when off.
    # Add full-field noise first only when new conditional mode is enabled.
    key="        if self.noise_mode == 'same':\n"
    s=once(s,key,'''        if self.joint_context:
            # Two independent full-field noises, shared clean source and sigma.
            # Crop identical local coordinates from the corresponding field.
            noisy_full1 = clean_full + sigma * torch.randn_like(clean_full)
            noisy_full2 = clean_full + sigma * torch.randn_like(clean_full)
            noisy1 = _extract_patches(noisy_full1, top1, left1, int(patch_size))
            noisy2 = _extract_patches(noisy_full2, top2, left2, int(patch_size))
        else:
'''+''.join('    '+ln if ln.strip() else ln for ln in key.splitlines(keepends=True)),p)
    # Indent the existing two-branch code under the new else.
    start=s.index('        else:\n            if self.noise_mode == \'same\':') if "        else:\n            if self.noise_mode == 'same':" in s else -1
    if start<0:
        # our inserted key was four-spaced so expect 12-space if statement
        raise RuntimeError('loss patch: failed to find condition anchor')
    until=s.index('        noisy_pair = torch.cat(',start)
    chunk=s[start+len('        else:\n'):until]
    # chunk includes a 12-indent if line and its old 8-indent body.
    # Simpler: restore indentation from the prior automatically reindented first
    # line: `            if ...` and preserve all lines but add 4 to those after.
    lines=chunk.splitlines(keepends=True)
    if not lines[0].startswith('            if self.noise_mode'):
        raise RuntimeError('bad condition indentation')
    new=[lines[0]]
    new.extend(('    '+l if l.strip() else l) for l in lines[1:])
    s=s[:start+len('        else:\n')]+''.join(new)+s[until:]

    # This loss supports the GLF overlay's net_kwargs, and the original direct
    # net(...) call. Maintain GLF behavior when joint-ct is not selected.
    if '        net_kwargs = {}\n' in s:
        anchor='        denoised_pair = net(\n'
        if anchor not in s: raise RuntimeError('unexpected GLF denoise call')
        s=once(s,anchor,'''        if self.joint_context:
            net_kwargs['global_context'] = torch.cat([
                torch.nn.functional.adaptive_avg_pool2d(noisy_full1, (64, 64)),
                torch.nn.functional.adaptive_avg_pool2d(noisy_full2, (64, 64)),
            ], dim=0)
        denoised_pair = net(
''',p)
    else:
        anchor='        denoised_pair = net(noisy_pair, sigma_pair, x_pos=pos_pair,class_labels=labels_pair, augment_labels=None,)'
        s=once(s,anchor,'''        extra = {}
        if self.joint_context:
            extra['global_context'] = torch.cat([
                torch.nn.functional.adaptive_avg_pool2d(noisy_full1, (64, 64)),
                torch.nn.functional.adaptive_avg_pool2d(noisy_full2, (64, 64)),
            ], dim=0)
        denoised_pair = net(noisy_pair, sigma_pair, x_pos=pos_pair,
                            class_labels=labels_pair, augment_labels=None,
                            **extra)''',p)
    return s


def patch_eval(s):
    p='eval/denoise_padding.py'
    anchor='    patch_batch = int(os.environ.get("PADIS_PATCH_BATCH", patches))'
    addition='''    # 3D CT-inspired conditional branch: SINGLE local network pass per patch.
    # Global condition is downsampled from the SAME noisy full image.  Keep
    # the full autograd graph for DPS likelihood backprop; no detach here.
    joint_context = None
    if bool(getattr(net, 'use_joint_global', False)):
        if psize != 64 or x_hat.shape[0] != 1:
            raise ValueError('JointContext DPS expects 64x64 patches, batch1')
        joint_context = torch.nn.functional.adaptive_avg_pool2d(x_hat, (64,64))

    def joint_call(xi, pi, ci):
        if joint_context is None:
            return net(xi, t_hat, pi, ci)
        return net(xi, t_hat, pi, ci,
                   global_context=joint_context.expand(xi.shape[0], -1, -1, -1))

'''
    s=once(s,anchor,addition+anchor,p)
    if '    def run_patch_batch(' in s:
        # GLF-overlaid evaluator: leave its existing model batching, checkpoint
        # and global encoder branches alone. Joint CT uses only the fallback.
        s=once(s, '        if prepared_global is None:\n            return net(xi, t_hat, pi, ci)',
               '        if prepared_global is None:\n            return joint_call(xi, pi, ci)', p)
    else:
        # main-WSY evaluator: default all/mini-batch/checkpoint paths.
        s=once(s,'        bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)',
               '        bigout = joint_call(x_input, pos_input, class_labels).to(torch.float64)',p)
        s=once(s,'                    return net(xi_, t_hat, pi_, ci)',
               '                    return joint_call(xi_, pi_, ci)',p)
        s=once(s,'                out = net(xi, t_hat, pi, ci)',
               '                out = joint_call(xi, pi, ci)',p)
    return s


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    root=args.root.resolve()
    targets=[('train/padis-mri/train_overlap.py',patch_train),
             ('train/padis-mri/training/patch_overlap_independent_loss.py',patch_loss),
             ('eval/denoise_padding.py',patch_eval)]
    modified={}
    for name,func in targets:
        f=root/name
        old=f.read_text()
        if 'joint_ct_context' in old or 'self.joint_context =' in old or 'def joint_call(' in old:
            raise RuntimeError(f'{f} already appears patched; refusing double-install')
        new=func(old)
        compile(new,str(f),'exec')
        modified[f]=new
    newmod=ROOT_THIS/'train/padis-mri/training/joint_context.py'
    dest=root/'train/padis-mri/training/joint_context.py'
    additions=[(newmod,dest)]
    additions += [
        (ROOT_THIS/f'bash/{name}',root/f'bash/{name}')
        for name in ('train_joint_ct_opa.sh','eval_joint_ct_pilot8.sh')
    ]
    for src,dst in additions:
        if dst.exists() and src.resolve() != dst.resolve():
            raise RuntimeError(f'{dst} already exists; refusing overwrite')
        if not src.exists():raise RuntimeError(f'{src} missing from package')
    compile(newmod.read_text(),str(newmod),'exec')
    for f in modified:
        backup=f.with_name(f.name+'.pre_joint_ct.bak')
        if backup.exists():raise RuntimeError(f'backup exists: {backup}')
        print(f'[validated] {f}')
    for src,dst in additions:
        print(f'[validated add] {dst}')
    if args.dry_run:
        print('DRY RUN: no files changed');return
    for f,new in modified.items():
        backup=f.with_name(f.name+'.pre_joint_ct.bak')
        shutil.copy2(f,backup)
        f.write_text(new)
    for src,dst in additions:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src,dst)
    print('Installed joint noisy full-field conditional network/loss/eval and bash scripts.')

if __name__=='__main__': main()
