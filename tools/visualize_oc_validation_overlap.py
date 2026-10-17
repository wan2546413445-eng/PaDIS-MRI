#!/usr/bin/env python3
"""Manual-pair OC training-branch diagnostic, with no reconstruction or training.

Run from the PaDIS-MRI repository root. Torch is imported only for capture;
--input_npz redraw works with NumPy, Matplotlib, and Pillow alone.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import math
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

DEFAULT_DATA = '/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB/noisy.pt'
REVIEWED_COMMIT = '4f58b8569a75e7e9c114b9944533e2dddebcc2cc'
# (x, y, orientation), zero based in ORIGINAL 384x384 FOV, not padded canvas.
# Manually chosen from the existing sample00 AGGA image, not from prediction error.
DEFAULT_PAIRS = [(72, 116, 'horizontal'), (224, 172, 'vertical')]
PATCH_SIZE, ORIGINAL_SIZE = 64, 384


def pair_geometry(x, y, orientation):
    p, stride = PATCH_SIZE, PATCH_SIZE // 2
    if orientation not in ('horizontal', 'vertical'):
        raise ValueError('orientation must be horizontal or vertical')
    x2, y2 = (x + stride, y) if orientation == 'horizontal' else (x, y + stride)
    if min(x, y, x2, y2) < 0 or max(x, x2) + p > ORIGINAL_SIZE or max(y, y2) + p > ORIGINAL_SIZE:
        raise ValueError('Both patches must be entirely inside the original 384x384 FOV')
    return {'x1': x, 'y1': y, 'x2': x2, 'y2': y2, 'horizontal': orientation == 'horizontal',
            'overlap_xywh': [max(x, x2), max(y, y2), p-stride if x != x2 else p,
                             p-stride if y != y2 else p]}


def aligned(a, b, horizontal):
    """NumPy version used for display validation; same orientation as repo helper."""
    o = PATCH_SIZE // 2
    if horizontal:
        return a[..., :, -o:], b[..., :, :o]
    return a[..., -o:, :].swapaxes(-1, -2), b[..., :o, :].swapaxes(-1, -2)


def oc_maps(a, b, d1, d2, sigma, sigma_data):
    delta = a - b  # two real/imag channels, not magnitude-image subtraction
    diff = np.sqrt(np.sum(delta.astype(np.float64)**2, axis=0))
    confidence_raw = np.abs(d1 - d2)
    confidence = confidence_raw / max(float(confidence_raw.mean()), 1e-12)
    weight = (sigma*sigma + sigma_data*sigma_data) / (sigma*sigma_data)**2
    # Exactly mean over the two channels of the current OC auxiliary_pixel.
    oc_pixel = weight * confidence * np.mean(delta.astype(np.float64)**2, axis=0)
    return diff, confidence, oc_pixel


def magnitude(x):
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float64)**2, axis=0))


def interior_boundary_maps(o1,o2,gt,d1,d2):
    """Geometry chooses the views, without looking at prediction or GT errors."""
    mask=d1>=d2  # same tie convention as AnchorOverlapConsistencyPatchEDMLoss
    valid=d1!=d2
    if not valid.any():
        raise ValueError('No pixels have distinct boundary distances')
    interior=np.where(mask[None],o1,o2)
    boundary=np.where(mask[None],o2,o1)
    maps={'interior_prediction':interior,'boundary_prediction':boundary,
          'interior_from_patch1':mask,'direction_valid':valid}
    for mode in ('complex','magnitude'):
        for label,pred in (('interior',interior),('boundary',boundary)):
            residual=(magnitude(pred.astype(np.float64)-gt.astype(np.float64))
                      if mode=='complex' else np.abs(magnitude(pred)-magnitude(gt)))
            maps[f'{label}_{mode}_residual']=residual
        gap=maps[f'boundary_{mode}_residual']**2-maps[f'interior_{mode}_residual']**2
        maps[f'{mode}_squared_error_gap']=gap
    return maps


def interior_boundary_metrics(arrays,rec):
    k=rec['prefix']; valid=arrays[k+'_direction_valid']
    conf=arrays[k+'_confidence'][valid]
    result={'model':rec['model'],'sigma':rec['sigma'],'group':rec['group'],
            'valid_pixels':int(valid.sum()),'equal_distance_pixels':int((~valid).sum()),
            'valid_fraction':float(valid.mean())}
    for mode in ('complex','magnitude'):
        ei=arrays[k+f'_interior_{mode}_residual'][valid]**2
        eb=arrays[k+f'_boundary_{mode}_residual'][valid]**2
        result.update({
            f'interior_{mode}_rmse':float(np.sqrt(ei.mean())),
            f'boundary_{mode}_rmse':float(np.sqrt(eb.mean())),
            f'{mode}_mean_squared_error_gap':float((eb-ei).mean()),
            f'{mode}_interior_lower_error_fraction':float((ei<eb).mean()),
            f'{mode}_boundary_lower_error_fraction':float((eb<ei).mean()),
            f'{mode}_equal_error_fraction':float((eb==ei).mean()),
            f'confidence_weighted_interior_{mode}_rmse':float(np.sqrt(np.sum(conf*ei)/conf.sum())),
            f'confidence_weighted_boundary_{mode}_rmse':float(np.sqrt(np.sum(conf*eb)/conf.sum()))})
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo_root', type=Path, default=Path.cwd())
    p.add_argument('--data', type=Path, default=Path(DEFAULT_DATA))
    p.add_argument('--sample_index', type=int, default=0)
    p.add_argument('--pad_width', type=int, choices=[64,96], default=64,
                   help='Match checkpoint training padding: pad64 -> 512, pad96 -> 576 positional normalization')
    p.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                   help='One or more official network-snapshot .pkl files; no default checkpoint')
    p.add_argument('--model_key', choices=['ema', 'net'], default='ema')
    p.add_argument('--pair', action='append', nargs=3, metavar=('X','Y','ORIENTATION'),
                   help='Specify exactly two manual anchor positions in the ORIGINAL FOV')
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.3],
                   help='Fixed diagnostic sigma(s); 0.3 is near exp(-1.2), not a training log event')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--noise_mode', choices=['independent','shared'], default='independent',
                   help='Current OC training uses independent; shared is a separately labelled context diagnostic')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--input_npz', type=Path, help='Redraw saved real predictions without loading Torch/model/data')
    p.add_argument('--latest_npz', action='store_true',
                   help='Redraw the newest overlap_predictions.npz under --output_root, selected by modification time')
    p.add_argument('--preview_only', action='store_true', help='Export patch positions and clean patches only, no checkpoint')
    p.add_argument('--output_dir', type=Path, help='NEW directory; existing directories are refused')
    p.add_argument('--output_root', type=Path, default=Path('/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/oc_training_overlap_view'))
    p.add_argument('--difference_vmax', type=float, help='Shared upper bound across all groups/models/sigmas; default joint P99')
    p.add_argument('--oc_vmax', type=float, help='Shared confidence/EDM weighted channel-MSE upper bound; default joint P99')
    p.add_argument('--gt_residual_vmax', type=float,
                   help='Shared GT residual upper bound; default joint P99 per residual definition')
    p.add_argument('--asset_long_edge', type=int, default=1024, help='Nearest-neighbor enlargement; no added image detail')
    a = p.parse_args()
    if a.latest_npz:
        if a.input_npz is not None:
            p.error('Use either --input_npz or --latest_npz')
        candidates=list(a.output_root.rglob('overlap_predictions.npz'))
        if not candidates:
            p.error(f'No overlap_predictions.npz found under {a.output_root}')
        a.input_npz=max(candidates,key=lambda path:(path.stat().st_mtime_ns,str(path)))
    if a.input_npz and (a.preview_only or a.checkpoint or a.pair):
        p.error('--input_npz cannot be combined with capture/selection options')
    if not a.input_npz and not a.preview_only and not a.checkpoint:
        p.error('Supply --checkpoint NAME=PATH or use --preview_only')
    if a.preview_only and a.checkpoint:
        p.error('--preview_only does not load checkpoints')
    if a.pair and len(a.pair) != 2:
        p.error('--pair must be supplied exactly twice')
    if not a.sigmas or len(set(a.sigmas)) != len(a.sigmas) or any(not math.isfinite(v) or v <= 0 for v in a.sigmas):
        p.error('--sigmas must be distinct finite positive values')
    if a.output_dir is not None and a.output_dir.exists():
        p.error('--output_dir must not already exist')
    if a.sample_index < 0 or a.seed < 0 or a.asset_long_edge < 64:
        p.error('sample index/seed must be nonnegative; asset_long_edge >= 64')
    for v in (a.difference_vmax, a.oc_vmax, a.gt_residual_vmax):
        if v is not None and (not math.isfinite(v) or v <= 0):
            p.error('Display upper bounds must be finite and positive')
    return a


def import_training_helpers(root):
    training_root = root / 'train/padis-mri'
    if not (training_root / 'training/patch_overlap_loss.py').is_file():
        raise FileNotFoundError('Run from the repository root or set --repo_root to PaDIS-MRI')
    sys.path.insert(0, str(root.resolve()))
    sys.path.insert(0, str(training_root.resolve()))
    helper = importlib.import_module('training.patch_overlap_loss')
    if Path(helper.__file__).resolve() != (training_root/'training/patch_overlap_loss.py').resolve():
        raise RuntimeError('A different training package is already imported')
    return helper

def load_image(a):
    import torch

    sample_path = a.data / f"sample_{a.sample_index}.pt"

    payload = torch.load(
        sample_path,
        map_location='cpu',
        weights_only=False
    )

    if 'gt' not in payload:
        raise ValueError(
            f"sample file does not contain gt: {payload.keys()}"
        )

    s = payload['gt']

    if torch.is_complex(s) and s.ndim == 2:
        image = torch.view_as_real(s).permute(2,0,1).contiguous().float()

    elif s.shape == (2,384,384):
        image = s.float()

    elif s.shape == (384,384,2):
        image = s.permute(2,0,1).contiguous().float()

    else:
        raise ValueError(
            f"Unexpected gt shape {s.shape}"
        )

    return image

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def capture(a):
    import torch
    import torch.nn.functional as F
    image = load_image(a)
    pairs = [pair_geometry(int(x),int(y),o) for x,y,o in (a.pair or DEFAULT_PAIRS)]
    meta = {'format_version':1, 'reviewed_repository':'wan2546413445-eng/PaDIS-MRI',
            'reviewed_branch':'main-WSY', 'reviewed_commit':REVIEWED_COMMIT,
            'data_path':str(a.data.resolve()), 'source_key':'x_est_gt', 'sample_index':a.sample_index,
            'source_tensor_sha256':hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            'image_shape':[384,384], 'padding':a.pad_width,
            'position_resolution':384+2*a.pad_width, 'patch_size':64,
            'selection':'manual anatomical locations chosen before inspecting prediction differences; no AGGA sampling',
            'selection_status':'proposed from existing AGGA raster; inspect preview against the original tensor',
            'pairs':pairs, 'seed':a.seed, 'noise_mode':a.noise_mode,
            'forward_mode':'eval, no_grad, dropout disabled; offline training-branch diagnostic',
            'sigma_policy':'fixed diagnostic sigma, not sampled training events',
            'model_key':a.model_key, 'records':[], 'models':[]}
    arrays = {'source_image':image.numpy()}
    for i,g in enumerate(pairs):
        for j in (1,2):
            arrays[f'pair{i+1}_clean{j}'] = image[:,g[f'y{j}']:g[f'y{j}']+64,g[f'x{j}']:g[f'x{j}']+64].numpy()
    if a.preview_only:
        return arrays, meta
    helper = import_training_helpers(a.repo_root)
    meta['helper_sha256'] = sha256(Path(helper.__file__))
    meta['oc_loss_source_sha256'] = sha256(a.repo_root/'train/padis-mri/training/patch_anchor_overlap_consistency_loss.py')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly use --device cpu for CPU forward')
    pad=a.pad_width
    resolution=384+2*pad
    full = F.pad(image[None],(pad,pad,pad,pad)).to(device)
    setups = []
    for i,g in enumerate(pairs):
        t1,l1,t2,l2 = [torch.tensor([g[k]+pad],device=device) for k in ('y1','x1','y2','x2')]
        c1,c2 = [helper._extract_patches(full,t,l,64) for t,l in ((t1,l1),(t2,l2))]
        p1,p2 = [helper._position_maps(t,l,1,64,resolution,device) for t,l in ((t1,l1),(t2,l2))]
        horizontal = torch.tensor([g['horizontal']],device=device)
        d1,d2 = helper._boundary_distances(horizontal,64,32,device,torch.float32)
        ca,cb = helper._overlap_views(c1,c2,horizontal,32)
        if not torch.equal(ca,cb):
            raise RuntimeError('Physical clean overlap is misaligned')
        generator = torch.Generator(device='cpu').manual_seed(a.seed + i)
        if a.noise_mode == 'independent':
            e1=torch.randn(c1.shape,generator=generator).to(device)
            e2=torch.randn(c2.shape,generator=generator).to(device)
        else:
            field=torch.randn((1,2,resolution,resolution),generator=generator).to(device)
            e1,e2=[helper._extract_patches(field,t,l,64) for t,l in ((t1,l1),(t2,l2))]
        setups.append((c1,c2,p1,p2,horizontal,d1,d2,e1,e2))
    names = set()
    for spec in a.checkpoint:
        if '=' not in spec:
            raise ValueError('--checkpoint must use NAME=/path/network-snapshot.pkl')
        name,path=spec.split('=',1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or name in names:
            raise ValueError('Model names must be unique and contain only letters/digits/_/-')
        names.add(name)
        checkpoint=Path(path).expanduser().resolve()
        if checkpoint.suffix != '.pkl' or not checkpoint.is_file():
            raise FileNotFoundError(f'Expected official .pkl snapshot: {checkpoint}')
        with checkpoint.open('rb') as f:
            snapshot=pickle.load(f)
        if a.model_key not in snapshot:
            raise KeyError(f'Checkpoint has no {a.model_key} model')
        net=snapshot[a.model_key].eval().requires_grad_(False).to(device)
        del snapshot
        if int(net.img_resolution) != resolution or int(net.label_dim) != 0:
            raise ValueError(f'Checkpoint img_resolution={net.img_resolution}, expected {resolution} for --pad_width {pad}; match the training padding explicitly')
        sigma_data=float(net.sigma_data)
        if not math.isfinite(sigma_data) or not math.isclose(sigma_data,0.5,rel_tol=0,abs_tol=1e-8):
            raise ValueError('Current reviewed training uses sigma_data=0.5 for both network and OC loss')
        meta['models'].append({'name':name,'checkpoint':str(checkpoint),'sha256':sha256(checkpoint),
                               'sigma_data':sigma_data,'img_resolution':int(net.img_resolution)})
        for si,sigma in enumerate(a.sigmas):
            for i,(c1,c2,p1,p2,horizontal,d1,d2,e1,e2) in enumerate(setups):
                n1,n2=c1+sigma*e1,c2+sigma*e2
                sig=torch.full((2,1,1,1),sigma,device=device)
                with torch.no_grad():
                    predictions=net(torch.cat([n1,n2]),sig,x_pos=torch.cat([p1,p2]),
                                    class_labels=None,augment_labels=None,force_fp32=True)
                if predictions.shape != (2,2,64,64) or not torch.isfinite(predictions).all():
                    raise ValueError('Non-finite or unexpected model predictions')
                oa,ob=helper._overlap_views(predictions[:1],predictions[1:],horizontal,32)
                prefix=f'{name}_s{si}_g{i+1}'
                rec={'prefix':prefix,'model':name,'sigma':sigma,'sigma_index':si,'group':i+1,
                     'sigma_data':sigma_data}
                for key,value in {'pred1':predictions[0], 'pred2':predictions[1],
                                  'overlap1':oa[0], 'overlap2':ob[0],
                                  'noisy1':n1[0], 'noisy2':n2[0],
                                  'noise1':e1[0], 'noise2':e2[0],
                                  'distance1':d1[0,0], 'distance2':d2[0,0]}.items():
                    arrays[f'{prefix}_{key}']=value.detach().cpu().numpy()
                # Compare diagnostic scalar against the exact current training reduction.
                confidence=(d1-d2).abs()
                confidence=confidence/confidence.mean((1,2,3),keepdim=True).clamp_min(1e-12)
                interior=torch.where(d1>=d2,oa,ob)
                boundary=torch.where(d1>=d2,ob,oa)
                weight=(sigma**2+sigma_data**2)/(sigma*sigma_data)**2
                exact=float((weight*confidence*(boundary-interior.detach()).square()).mean().cpu())
                rec['oc_raw_training_reduction']=exact
                meta['records'].append(rec)
        del net
    return arrays,meta


def validate_payload(arrays,meta):
    if meta.get('format_version') != 1 or len(meta.get('pairs',[])) != 2:
        raise ValueError('Unsupported payload; expected exactly two groups')
    if arrays['source_image'].shape != (2,384,384):
        raise ValueError('Bad source image shape')
    for group,g in enumerate(meta['pairs'],1):
        for j in (1,2):
            target=arrays['source_image'][:,g[f'y{j}']:g[f'y{j}']+64,g[f'x{j}']:g[f'x{j}']+64]
            if not np.array_equal(target,arrays[f'pair{group}_clean{j}']):
                raise ValueError('Saved clean patch disagrees with source GT coordinates')
        gt1,gt2=aligned(arrays[f'pair{group}_clean1'],arrays[f'pair{group}_clean2'],g['horizontal'])
        if not np.array_equal(gt1,gt2):
            raise ValueError('GT overlap is not physically aligned')
    for rec in meta['records']:
        k=rec['prefix']; g=meta['pairs'][rec['group']-1]
        pred1,pred2=arrays[k+'_pred1'],arrays[k+'_pred2']
        o1,o2=aligned(pred1,pred2,g['horizontal'])
        if not np.array_equal(o1,arrays[k+'_overlap1']) or not np.array_equal(o2,arrays[k+'_overlap2']):
            raise ValueError('Saved overlap disagrees with physical alignment')
        d,c,w=oc_maps(o1,o2,arrays[k+'_distance1'],arrays[k+'_distance2'],rec['sigma'],rec['sigma_data'])
        if not np.isclose(w.mean(),rec['oc_raw_training_reduction'],rtol=3e-5,atol=1e-7):
            raise ValueError('OC diagnostic differs from the exact training reduction')
        arrays[k+'_difference'],arrays[k+'_confidence'],arrays[k+'_oc_pixel']=d,c,w
        gt,_=aligned(arrays[f"pair{rec['group']}_clean1"],
                     arrays[f"pair{rec['group']}_clean2"],g['horizontal'])
        arrays[k+'_gt_overlap']=gt.copy()
        for j,o in ((1,o1),(2,o2)):
            arrays[k+f'_gt_residual{j}']=magnitude(o.astype(np.float64)-gt.astype(np.float64))
            arrays[k+f'_magnitude_residual{j}']=np.abs(magnitude(o)-magnitude(gt))
        # Validate the saved geometric distances against the actual patch edges.
        yy,xx=np.mgrid[:64,:64]
        edge=np.minimum.reduce((xx,yy,63-xx,63-yy))
        expected_d1,expected_d2=aligned(edge,edge,g['horizontal'])
        if not np.array_equal(expected_d1,arrays[k+'_distance1']) or not np.array_equal(expected_d2,arrays[k+'_distance2']):
            raise ValueError('Saved boundary distance disagrees with patch geometry')
        maps=interior_boundary_maps(o1,o2,gt,expected_d1,expected_d2)
        arrays.update({k+'_'+key:value for key,value in maps.items()})
    for key,x in arrays.items():
        if not np.all(np.isfinite(x)):
            raise ValueError(f'Non-finite saved data: {key}')


def save_asset(path,array,cmap,vmin,vmax,long_edge,valid=None):
    norm=np.clip((array-vmin)/(vmax-vmin),0,1)
    pixels=(plt.get_cmap(cmap)(norm)[...,:3]*255).round().astype(np.uint8)
    if valid is not None:
        pixels=np.concatenate((pixels,(valid.astype(np.uint8)*255)[...,None]),axis=-1)
    im=Image.fromarray(pixels)
    scale=long_edge/max(im.size)
    if scale > 1:
        im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
    im.save(path,dpi=(300,300))


def show(ax,array,title,cmap='gray',vmin=0,vmax=1):
    im=ax.imshow(array,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set_title(title,fontsize=11,pad=6)
    ax.axis('off')
    return im


def draw_locations(ax,image,g,zoom=False):
    ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    for j,color in ((1,'#36C6FF'),(2,'#FFB347')):
        ax.add_patch(Rectangle((g[f'x{j}']-.5,g[f'y{j}']-.5),64,64,fill=False,edgecolor=color,lw=1.7))
    x,y,w,h=g['overlap_xywh']
    ax.add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,edgecolor='#FFE75C',lw=1.2))
    if zoom:
        ax.set_xlim(min(g['x1'],g['x2'])-8,max(g['x1'],g['x2'])+72)
        ax.set_ylim(max(g['y1'],g['y2'])+72,min(g['y1'],g['y2'])-8)
    ax.axis('off')


def save_location_asset(path,image,g,zoom,long_edge):
    fig,ax=plt.subplots(figsize=(4,4))
    draw_locations(ax,image,g,zoom)
    fig.subplots_adjust(0,0,1,1)
    # Transparent outside the image aspect, no text/axes. Tight trim below.
    fig.savefig(path,dpi=256,pad_inches=0,bbox_inches='tight',facecolor='black')
    plt.close(fig)
    with Image.open(path) as im:
        im=im.convert('RGB')
        if max(im.size)<long_edge:
            scale=long_edge/max(im.size)
            im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
        im.save(path,dpi=(300,300))


def upper(arrays,keys,override):
    if override is not None:
        return override
    values=np.concatenate([arrays[k].ravel() for k in keys])
    p99=float(np.quantile(values,0.99))
    return p99 if p99>0 else 1.0


def export_gt_overviews(arrays,meta,recs,source,magref,resmax,magmax,out,assets,a):
    """GT-reference diagnosis with identical scales for both predictions."""
    name=recs[0]['model']; sigma_label=str(recs[0]['sigma']).replace('.','p')
    for mode,vmax in (('complex',resmax),('magnitude',magmax)):
        fig,axes=plt.subplots(2,7,figsize=(20,8),layout='constrained')
        residual_axes=[]
        for row,r in enumerate(recs):
            k=r['prefix'];group=r['group'];g=meta['pairs'][group-1]
            dest=assets/f'{name}_sigma{sigma_label}'/f'group{group}'
            draw_locations(axes[row,0],source,g,True)
            axes[row,0].set_title(f'Group {group}: manual pair',fontsize=11)
            for col,key,title in ((1,'gt_overlap','Aligned GT (x_est_gt)'),
                                  (2,'overlap1','Overlap prediction 1'),
                                  (3,'overlap2','Overlap prediction 2')):
                value=magnitude(arrays[k+'_'+key])/magref
                show(axes[row,col],value,title)
                if key=='gt_overlap':
                    save_asset(dest/'gt_overlap.png',value,'gray',0,1,a.asset_long_edge)
            for col,j in ((4,1),(5,2)):
                key=f'gt_residual{j}' if mode=='complex' else f'magnitude_residual{j}'
                value=arrays[k+'_'+key]
                rmse=float(np.sqrt(np.mean(value**2)))
                im=show(axes[row,col],value,f'Prediction {j} vs GT\nRMSE={rmse:.4g}',
                        'turbo',0,vmax)
                save_asset(dest/(key+'.png'),value,'turbo',0,vmax,a.asset_long_edge)
                residual_axes.append(axes[row,col])
            diffmax=meta['display']['difference_range'][1]
            diffim=show(axes[row,6],arrays[k+'_difference'],'Prediction 1 vs 2\nComplex difference',
                        'turbo',0,diffmax)
        residual_label='|D - GT| (complex residual)' if mode=='complex' else '||D| - |GT|| (magnitude residual)'
        cbar=fig.colorbar(im,ax=residual_axes,shrink=.85,pad=.02,ticks=[0,vmax/2,vmax])
        cbar.set_label(residual_label,fontsize=10)
        cbar.ax.set_yticklabels([f'{v:.4g}' for v in (0,vmax/2,vmax)])
        cbar=fig.colorbar(diffim,ax=list(axes[:,6]),shrink=.85,pad=.02,ticks=[0,diffmax/2,diffmax])
        cbar.set_label('|D1 - D2| (complex difference)',fontsize=10)
        cbar.ax.set_yticklabels([f'{v:.4g}' for v in (0,diffmax/2,diffmax)])
        fig.suptitle(f"{name} | sample {meta['sample_index']} | sigma={recs[0]['sigma']:g} | "
                     f"{meta['noise_mode']} noise | GT residual: {mode}\n"
                     '64x64 patches, shift32, 2 manual pairs | residual RMSE: lower is better',fontsize=13)
        fig.savefig(out/f'overview_gt_{mode}_{name}_sigma{sigma_label}.png',dpi=300,facecolor='white')
        plt.close(fig)


def export_interior_boundary(arrays,meta,recs,source,magref,out,assets,a):
    name=recs[0]['model']; sigma_label=str(recs[0]['sigma']).replace('.','p')
    for mode in ('complex','magnitude'):
        ranges=meta['display']['interior_boundary_ranges'][mode]
        vmax=ranges['residual_max']; gapmax=ranges['absolute_squared_error_gap_max']
        fig,axes=plt.subplots(2,7,figsize=(20,8),layout='constrained')
        res_axes=[]
        for row,r in enumerate(recs):
            k=r['prefix'];valid=arrays[k+'_direction_valid'];group=r['group']
            dest=assets/f'{name}_sigma{sigma_label}'/f'group{group}'
            draw_locations(axes[row,0],source,meta['pairs'][group-1],True)
            axes[row,0].set_title(f'Group {group}: manual pair',fontsize=11)
            for col,key,title in ((1,'gt_overlap','Aligned GT'),
                                  (2,'interior_prediction','Interior prediction\nPixelwise selected'),
                                  (3,'boundary_prediction','Boundary prediction\nPixelwise selected')):
                value=magnitude(arrays[k+'_'+key])/magref
                show(axes[row,col],value,title)
                save_asset(dest/(key+'.png'),value,'gray',0,1,a.asset_long_edge)
            metrics=interior_boundary_metrics(arrays,r)
            for col,label in ((4,'interior'),(5,'boundary')):
                key=f'{label}_{mode}_residual';value=arrays[k+'_'+key]
                rmse=metrics[f'{label}_{mode}_rmse']
                im=show(axes[row,col],np.ma.array(value,mask=~valid),
                        f'{label.title()} residual to GT\nRMSE={rmse:.5f}','turbo',0,vmax)
                res_axes.append(axes[row,col])
                save_asset(dest/(key+'.png'),value,'turbo',0,vmax,a.asset_long_edge,valid)
            key=f'{mode}_squared_error_gap';value=arrays[k+'_'+key]
            gapim=show(axes[row,6],np.ma.array(value,mask=~valid),
                       'Boundary - Interior\nSquared GT error',
                       'coolwarm',-gapmax,gapmax)
            save_asset(dest/(key+'.png'),value,'coolwarm',-gapmax,gapmax,a.asset_long_edge,valid)
            save_asset(dest/'direction_valid.png',valid.astype(float),'gray',0,1,a.asset_long_edge)
            save_asset(dest/'interior_from_patch1.png',arrays[k+'_interior_from_patch1'].astype(float),
                       'gray',0,1,a.asset_long_edge,valid)
        cbar=fig.colorbar(im,ax=res_axes,shrink=.85,pad=.02,ticks=[0,vmax/2,vmax])
        cbar.set_label(f'GT {mode} residual (raw units)',fontsize=10)
        cbar.ax.set_yticklabels([f'{v:.4g}' for v in (0,vmax/2,vmax)])
        cbar=fig.colorbar(gapim,ax=list(axes[:,6]),shrink=.85,pad=.02,ticks=[-gapmax,0,gapmax])
        cbar.set_label('Signed squared-error difference',fontsize=10)
        cbar.ax.set_yticklabels([f'{v:.4g}' for v in (-gapmax,0,gapmax)])
        fig.suptitle(f"{name} | sample {meta['sample_index']} | sigma={recs[0]['sigma']:g} | "
                     f"{meta['noise_mode']} noise | {mode} GT error\n"
                     'Pixelwise edge-distance selection | red: boundary error higher; blue: interior error higher | white: equal distances, excluded',
                     fontsize=12)
        fig.savefig(out/f'overview_interior_boundary_{mode}_{name}_sigma{sigma_label}.png',
                    dpi=300,facecolor='white')
        plt.close(fig)


def export(arrays,meta,out,a):
    validate_payload(arrays,meta)
    assets=out/'assets'; assets.mkdir()
    magref=float(np.quantile(magnitude(arrays['source_image']),0.99))
    if magref <= 0:
        raise ValueError('Source P99 must be positive')
    source=np.clip(magnitude(arrays['source_image'])/magref,0,1)
    save_asset(assets/'source_magnitude.png',source,'gray',0,1,a.asset_long_edge)
    fig,axes=plt.subplots(1,2,figsize=(9,4.7),layout='constrained')
    for i,g in enumerate(meta['pairs']):
        draw_locations(axes[i],source,g)
        axes[i].set_title(f"Group {i+1}: patch1 ({g['x1']},{g['y1']}), patch2 ({g['x2']},{g['y2']})",fontsize=10)
        save_location_asset(assets/f'group{i+1}_locations.png',source,g,False,a.asset_long_edge)
        save_location_asset(assets/f'group{i+1}_locations_zoom.png',source,g,True,a.asset_long_edge)
        for j in (1,2):
            save_asset(assets/f'group{i+1}_clean_patch{j}.png',magnitude(arrays[f'pair{i+1}_clean{j}'])/magref,'gray',0,1,a.asset_long_edge)
    fig.suptitle(f"Manual pairs | sample {meta['sample_index']} | 64x64, shift32 | original FOV | cyan=patch1, orange=patch2, yellow=overlap",fontsize=12)
    fig.savefig(out/'overview_locations.png',dpi=300,facecolor='white')
    plt.close(fig)
    if not meta['records']:
        meta['display']={'magnitude_reference':'source magnitude P99','magnitude_vmax_raw':magref}
        return
    diffmax=upper(arrays,[r['prefix']+'_difference' for r in meta['records']],a.difference_vmax)
    ocmax=upper(arrays,[r['prefix']+'_oc_pixel' for r in meta['records']],a.oc_vmax)
    resmax=upper(arrays,[r['prefix']+f'_gt_residual{j}' for r in meta['records'] for j in (1,2)],
                 a.gt_residual_vmax)
    magmax=upper(arrays,[r['prefix']+f'_magnitude_residual{j}' for r in meta['records'] for j in (1,2)],
                 a.gt_residual_vmax)
    confmax=max(float(arrays[r['prefix']+'_confidence'].max()) for r in meta['records'])
    meta['display']={'magnitude_reference':'source magnitude P99','magnitude_vmax_raw':magref,
                     'difference_range':[0,diffmax],'oc_pixel_range':[0,ocmax],
                     'gt_residual_range':[0,resmax],'magnitude_residual_range':[0,magmax],
                     'gt_reference':'source x_est_gt, same physical overlap; training target',
                     'gt_residual_definition':'sqrt(sum_channels((prediction-GT)^2)), raw units',
                     'magnitude_residual_definition':'abs(magnitude(prediction)-magnitude(GT)), raw units',
                     'confidence_range':[0,confmax], 'range_scope':'shared across BOTH groups, ALL models and sigmas',
                     'heatmap':'turbo','difference_definition':'sqrt(delta_real^2+delta_imag^2), raw tensor units',
                     'oc_pixel_definition':'EDM_weight * normalized_boundary_confidence * mean(delta_channels^2)',
                     'oc_pixel_excludes_lambda':True,'assets':'image only; nearest-neighbor enlargement, no added detail'}
    ranges={}
    for mode in ('complex','magnitude'):
        residuals=[];gaps=[]
        for r in meta['records']:
            k=r['prefix'];valid=arrays[k+'_direction_valid']
            residuals.extend(arrays[k+f'_{label}_{mode}_residual'][valid] for label in ('interior','boundary'))
            gaps.append(np.abs(arrays[k+f'_{mode}_squared_error_gap'][valid]))
        paired_resmax=float(np.quantile(np.concatenate(residuals),.99))
        paired_gapmax=float(np.quantile(np.concatenate(gaps),.99))
        ranges[mode]={'residual_max':paired_resmax if paired_resmax>0 else 1.0,
                      'absolute_squared_error_gap_max':paired_gapmax if paired_gapmax>0 else 1.0}
    meta['display']['interior_boundary_ranges']=ranges
    meta['interior_boundary_diagnostic']={
        'selection':'pixelwise d1>=d2 -> interior=pred1, boundary=pred2; otherwise reversed',
        'ties':'same selection convention as loss; equal distances excluded from paired statistics and heatmaps',
        'error_gap':'boundary GT residual squared minus interior GT residual squared; positive means boundary worse',
        'weighted_metrics':'OC normalized geometric confidence, normalized again by sum over valid pixels',
        'prediction_maps':'pixelwise composites of two forward outputs, not extra network predictions',
        'scope':'same physical overlap pixels, single saved noise realization; no prespecified outcome'}
    combinations=list(dict.fromkeys((r['model'],r['sigma_index']) for r in meta['records']))
    for name,si in combinations:
        recs=[r for r in meta['records'] if r['model']==name and r['sigma_index']==si]
        sigma_label=str(recs[0]['sigma']).replace('.','p')
        fig,axes=plt.subplots(2,8,figsize=(22,7.8),layout='constrained')
        figin,axin=plt.subplots(2,4,figsize=(12,7),layout='constrained')
        heat_axes=[]; oc_axes=[]
        for row,r in enumerate(recs):
            k=r['prefix']; group=r['group']; g=meta['pairs'][group-1]
            dest=assets/f'{name}_sigma{sigma_label}'/f'group{group}'; dest.mkdir(parents=True)
            draw_locations(axes[row,0],source,g,True)
            axes[row,0].set_title(f'Group {group}: manual pair',fontsize=11)
            save_location_asset(dest/'locations_zoom.png',source,g,True,a.asset_long_edge)
            panels=[('pred1','Patch 1 prediction','gray',1),('pred2','Patch 2 prediction','gray',1),
                    ('overlap1','Aligned overlap 1','gray',1),('overlap2','Aligned overlap 2','gray',1),
                    ('difference','Complex prediction difference','turbo',diffmax),
                    ('confidence','Boundary confidence','viridis',confmax),
                    ('oc_pixel','Weighted OC channel MSE','turbo',ocmax)]
            for col,(key,title,cmap,vmax) in enumerate(panels,1):
                data=arrays[k+'_'+key]
                value=magnitude(data)/magref if data.ndim==3 else data
                im=show(axes[row,col],value,title,cmap,0,vmax)
                save_asset(dest/(key+'.png'),value,cmap,0,vmax,a.asset_long_edge)
                if key=='difference': heat_axes.append(axes[row,col]); diffim=im
                if key=='oc_pixel': oc_axes.append(axes[row,col]); ocim=im
            for col,(key,title) in enumerate([('clean1','Source patch 1'),('clean2','Source patch 2'),
                                             ('noisy1','Noisy patch 1'),('noisy2','Noisy patch 2')]):
                data=arrays[f'pair{group}_{key}'] if key.startswith('clean') else arrays[k+'_'+key]
                value=magnitude(data)/magref
                show(axin[row,col],value,f'Group {group}: {title}')
                save_asset(dest/(key+'.png'),value,'gray',0,1,a.asset_long_edge)
        for f in (fig,figin):
            f.suptitle(f"{name} | sample {meta['sample_index']} | sigma={recs[0]['sigma']:g} | {meta['noise_mode']} noise | offline training-branch forward",fontsize=14)
        for im,axlist,label,vmax in ((diffim,heat_axes,'Raw complex difference',diffmax),
                                     (ocim,oc_axes,'Confidence / EDM weighted MSE (before lambda)',ocmax)):
            cbar=fig.colorbar(im,ax=axlist,shrink=.85,pad=.02,ticks=[0,vmax/2,vmax])
            cbar.set_label(label,fontsize=9)
            cbar.ax.set_yticklabels([f'{v:.4g}' for v in (0,vmax/2,vmax)])
        fig.savefig(out/f'overview_{name}_sigma{sigma_label}.png',dpi=300,facecolor='white')
        figin.savefig(out/f'overview_inputs_{name}_sigma{sigma_label}.png',dpi=300,facecolor='white')
        plt.close(fig); plt.close(figin)
        export_gt_overviews(arrays,meta,recs,source,magref,resmax,magmax,out,assets,a)
        export_interior_boundary(arrays,meta,recs,source,magref,out,assets,a)


def main():
    a=parse_args()
    if a.input_npz:
        print(f'Redraw source: {a.input_npz.resolve()}')
        with np.load(a.input_npz,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata_json'].item()))
            arrays={k:archive[k].copy() for k in archive.files if k!='metadata_json'}
    else:
        arrays,meta=capture(a)
    validate_payload(arrays,meta)
    out=a.output_dir or a.output_root/f"sample{meta['sample_index']}_manual_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True,exist_ok=False)
    # Save raw real/imag tensors before display clipping. Replay contains actual model outputs.
    np.savez_compressed(out/'overlap_predictions.npz',**arrays,metadata_json=np.array(json.dumps(meta,ensure_ascii=False)))
    export(arrays,meta,out,a)
    (out/'visualization_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'overlap_metrics.csv').open('w',encoding='utf-8') as f:
        f.write('model,sigma,group,complex_diff_mean,complex_diff_rms,oc_raw_training_reduction,'
                'pred1_gt_complex_mae,pred1_gt_complex_rmse,pred2_gt_complex_mae,pred2_gt_complex_rmse,'
                'pred1_gt_magnitude_mae,pred1_gt_magnitude_rmse,pred2_gt_magnitude_mae,pred2_gt_magnitude_rmse\n')
        for r in meta['records']:
            d=arrays[r['prefix']+'_difference']
            metrics=[]
            for kind in ('gt_residual','magnitude_residual'):
                for j in (1,2):
                    residual=arrays[r['prefix']+f'_{kind}{j}']
                    metrics.extend((float(residual.mean()),float(np.sqrt(np.mean(residual**2)))))
            values=','.join(f'{v:.9g}' for v in metrics)
            f.write(f"{r['model']},{r['sigma']},{r['group']},{d.mean():.9g},{np.sqrt((d*d).mean()):.9g},{r['oc_raw_training_reduction']:.9g},{values}\n")
    paired=[interior_boundary_metrics(arrays,r) for r in meta['records']]
    if paired:
        with (out/'interior_boundary_metrics.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
        for row in paired:
            print(f"{row['model']} sigma={row['sigma']:g} group={row['group']}: "
                  f"magnitude RMSE Interior={row['interior_magnitude_rmse']:.5f}, "
                  f"Boundary={row['boundary_magnitude_rmse']:.5f}; "
                  f"Interior lower-error pixels={100*row['magnitude_interior_lower_error_fraction']:.1f}% "
                  f"of {row['valid_pixels']} valid pixels")
    print(f'Saved: {out.resolve()}')
    print('No reconstruction, backpropagation, training, AGGA sampling, or Git write was performed.')


if __name__=='__main__':
    main()
