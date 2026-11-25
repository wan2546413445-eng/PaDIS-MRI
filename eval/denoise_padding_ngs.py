"""PaDIS random partition denoising with NGS context support.
NGS-only entry point; original denoise_padding.py is untouched.
No changes to DPS update formulas, patch grid or residual aggregation.
"""
import os
import random
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint



def getIndices(spaced, patches, pad, psize, freezeindex=False):
    a,b=0,0
    if pad>0:
        a=random.randint(0,pad-1)
        b=random.randint(0,pad-1)
    if freezeindex:
        a,b=0,0
    return [[spaced[p]+a,spaced[p]+a+psize,spaced[q]+b,spaced[q]+b+psize]
            for p in range(patches) for q in range(patches)]


def denoisedFromPatches(net,x,t_hat,latents_pos,class_labels,indices,pad=96,t_goal=-1,avg=1,spaced=[],wrong=False):
    if len(spaced)>1:
        indices=getIndices(spaced,5,24,56)
    x_hat=x if wrong else x.clone()
    N=x_hat.shape[-1]
    psize=indices[0][1]-indices[0][0]
    patches=len(indices)
    pad=int(N-np.sqrt(patches)*psize)
    output=torch.zeros_like(x_hat)
    x_input=torch.stack([x_hat[0,:,r0:r1,c0:c1] for r0,r1,c0,c1 in indices])
    pos_input=torch.stack([latents_pos[0,:,r0:r1,c0:c1] for r0,r1,c0,c1 in indices])
    patch_coords=torch.as_tensor([[z[0],z[2]] for z in indices],device=x_hat.device,dtype=torch.long)
    patch_batch=int(os.environ.get('PADIS_PATCH_BATCH',patches))
    if patch_batch<1:
        raise ValueError('PADIS_PATCH_BATCH must be positive')
    use_ckpt=os.environ.get('PADIS_PATCH_CHECKPOINT','0')=='1'
    prepared_global=None
    joint_context=None
    if bool(getattr(net,'use_ngs',False)):
        if psize==64:
            prepared_global=net.prepare_global(x_hat,t_hat)
            patch_batch=min(patch_batch,8)
    elif bool(getattr(net,'use_global_fusion',False)):
        prepared_global=net.prepare_global(x_hat,t_hat)
    if bool(getattr(net,'use_joint_global',False)):
        if psize!=64 or x_hat.shape[0]!=1:
            raise ValueError('JointContext DPS expects 64x64 patches, batch1')
        joint_context=torch.nn.functional.adaptive_avg_pool2d(x_hat,(64,64))
    def labels_slice(start,end):
        if class_labels is None or class_labels.shape[0]==1:
            return class_labels
        return class_labels[start:end]
    chunks=[]
    for start in range(0,patches,patch_batch):
        end=min(start+patch_batch,patches)
        xi,pi,pc=x_input[start:end],pos_input[start:end],patch_coords[start:end]
        ci=labels_slice(start,end)
        if prepared_global is not None:
            def run(xi_,pi_,pc_,ctx_,ci_=ci):
                return net(xi_,t_hat,x_pos=pi_,class_labels=ci_,prepared_global=ctx_,patch_coords=pc_,
                           global_batch_indices=torch.zeros(xi_.shape[0],device=xi_.device,dtype=torch.long))
            args=(xi,pi,pc,prepared_global)
        elif joint_context is not None:
            def run(xi_,pi_,ctx_,ci_=ci):
                return net(xi_,t_hat,pi_,ci_,global_context=ctx_.expand(xi_.shape[0],-1,-1,-1))
            args=(xi,pi,joint_context)
        else:
            def run(xi_,pi_,ci_=ci):
                return net(xi_,t_hat,pi_,ci_)
            args=(xi,pi)
        out=checkpoint(run,*args,use_reentrant=False) if use_ckpt else run(*args)
        chunks.append(out.to(torch.float64))
    bigout=torch.cat(chunks,dim=0)
    for i,(r0,r1,c0,c1) in enumerate(indices):
        output[0,:,r0:r1,c0:c1]+=bigout[i]
        output[0,:,r0:r1,c0:c1]-=x_hat[0,:,r0:r1,c0:c1]
    x_hat=x_hat+output
    temp=t_goal+torch.randn_like(x_hat)*t_goal
    if pad==0:
        return x_hat
    temp[:,:,pad:N-pad,pad:N-pad]=x_hat[:,:,pad:N-pad,pad:N-pad]
    return temp
