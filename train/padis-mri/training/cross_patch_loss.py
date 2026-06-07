import warnings
import numpy as np
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence


def _coord_token(i,j,ps,H,W,device):
    cy=i+ps/2.0-0.5; cx=j+ps/2.0-0.5
    x=(cx/(W-1)-0.5)*2.0; y=(cy/(H-1)-0.5)*2.0
    return torch.tensor([x,y],device=device,dtype=torch.float32)

def patchify_set(images, patch_size=64, K=8, local_k=3, global_k=4, context_mode='global_random', overlap_ratio=0.5, debug=False):
    B,C,H,W=images.shape; device=images.device
    assert K==1+local_k+global_k
    max_i,max_j=H-patch_size,W-patch_size
    clean=torch.empty(B,K,C,patch_size,patch_size,device=device,dtype=images.dtype)
    pos=torch.empty(B,K,2,patch_size,patch_size,device=device,dtype=images.dtype)
    tok=torch.empty(B,K,2,device=device,dtype=images.dtype)
    gx=torch.arange(patch_size,device=device).view(1,-1).repeat(patch_size,1)
    gy=torch.arange(patch_size,device=device).view(-1,1).repeat(1,patch_size)
    for b in range(B):
        coords=[]
        ai=int(torch.randint(0,max_i+1,(1,),device=device)); aj=int(torch.randint(0,max_j+1,(1,),device=device)); coords.append((ai,aj))
        if context_mode == 'global_random':
            tried = 0
            while len(coords) < 1 + local_k and tried < 200:
                tried += 1
                jitter = patch_size // 2
                ni = max(0, min(max_i, ai + int(torch.randint(-jitter, jitter + 1, (1,), device=device))))
                nj = max(0, min(max_j, aj + int(torch.randint(-jitter, jitter + 1, (1,), device=device))))
                if (ni, nj) not in coords: coords.append((ni, nj))
            while len(coords) < 1 + local_k: coords.append((ai, aj))
            tried = 0
            far = patch_size
            while len(coords) < K and tried < 400:
                tried += 1
                gi = int(torch.randint(0, max_i + 1, (1,), device=device));
                gj = int(torch.randint(0, max_j + 1, (1,), device=device))
                if (gi, gj) in coords: continue
                if abs(gi - ai) + abs(gj - aj) < far: continue
                coords.append((gi, gj))
            while len(coords) < K: coords.append((ai, aj))
        elif context_mode == 'local_only':
            tried = 0
            jitter=patch_size//2
            while len(coords) < K and tried < 400:
                tried += 1
                ni = max(0, min(max_i, ai + int(torch.randint(-jitter, jitter + 1, (1,), device=device))))
                nj = max(0, min(max_j, aj + int(torch.randint(-jitter, jitter + 1, (1,), device=device))))
                if (ni, nj) not in coords: coords.append((ni, nj))
            while len(coords) < K: coords.append((ai, aj))
        elif context_mode == 'overlap':
            tried = 0
            jitter = max(0, int(round(patch_size * (1.0 - overlap_ratio))))
            while len(coords) < K and tried < 400:
                tried += 1
                ni = max(0, min(max_i, ai + int(torch.randint(-jitter, jitter + 1, (1,), device=device))))
                nj = max(0, min(max_j, aj + int(torch.randint(-jitter, jitter + 1, (1,), device=device))))
                if (ni, nj) not in coords: coords.append((ni, nj))
            while len(coords) < K: coords.append((ai, aj))
        else:
            raise ValueError(f'Unknown cp_context_mode: {context_mode}')


        for k,(i,j) in enumerate(coords):
            clean[b,k]=images[b,:,i:i+patch_size,j:j+patch_size]
            xp=(gx+j)/(W-1); yp=(gy+i)/(H-1)
            pos[b,k,0]=(xp-0.5)*2.0; pos[b,k,1]=(yp-0.5)*2.0
            tok[b,k]=_coord_token(i,j,patch_size,H,W,device)
    if debug:
        assert clean.shape==(B,K,C,patch_size,patch_size)
        assert pos.shape==(B,K,2,patch_size,patch_size)
        assert tok.shape==(B,K,2)
    return clean,pos,tok

@persistence.persistent_class
class CrossPatch_EDMLoss:
    def __init__(self,P_mean=-1.2,P_std=1.2,sigma_data=0.5,cp_k=8,cp_local_k=3,cp_global_k=4,cp_patch_size=64,cp_context_mode='global_random',cp_target_only_loss=False,cp_overlap_ratio=0.5,cp_debug=False):
        self.P_mean=P_mean; self.P_std=P_std; self.sigma_data=sigma_data
        self.cp_k = cp_k;
        self.cp_local_k = cp_local_k;
        self.cp_global_k = cp_global_k;
        self.cp_patch_size = cp_patch_size
        self.cp_context_mode = cp_context_mode;
        self.cp_target_only_loss = cp_target_only_loss;
        self.cp_overlap_ratio = cp_overlap_ratio;
        self.cp_debug = cp_debug


    def __call__(self,net,images,patch_size,resolution,labels=None,augment_pipe=None):
        if augment_pipe is not None:
            if self.cp_debug:
                warnings.warn('CrossPatch ignores augment_pipe.')
            else:
                raise RuntimeError('CrossPatch does not support augment_pipe. Set --augment=0.')
        clean, x_pos, coords = patchify_set(images, patch_size=patch_size, K=self.cp_k, local_k=self.cp_local_k,
                                            global_k=self.cp_global_k, context_mode=self.cp_context_mode,
                                            overlap_ratio=self.cp_overlap_ratio, debug=self.cp_debug)

        B=images.shape[0]
        sigma=((torch.randn([B,1,1,1,1],device=images.device)*self.P_std)+self.P_mean).exp()
        weight=(sigma**2+self.sigma_data**2)/(sigma*self.sigma_data)**2
        yn=clean+torch.randn_like(clean)*sigma
        D=net(yn,sigma,x_pos=x_pos,patch_coords=coords,class_labels=labels)
        if self.cp_debug: assert D.shape==clean.shape
        if self.cp_target_only_loss:
            return weight * ((D[:, :1] - clean[:, :1]) ** 2)

        return weight*((D-clean)**2)
