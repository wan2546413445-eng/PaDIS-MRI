
import torch, numpy as np, random
from denoise_padding import getIndices as cross_getIndices

def _coord(i,j,ps,H,W,dev):
    cy=i+ps/2-0.5; cx=j+ps/2-0.5
    return torch.tensor([(cx/(W-1)-0.5)*2, (cy/(H-1)-0.5)*2],device=dev)

def denoisedFromCrossPatchSets(net,x_hat,t_hat,latents_pos,class_labels,indices,cp_k=8,cp_local_k=3,cp_global_k=4,cp_eval_batch_size=2,cp_debug=False):
    assert cp_k==1+cp_local_k+cp_global_k
    dev=x_hat.device; _,C,H,W=x_hat.shape; ps=indices[0][1]-indices[0][0]
    out=torch.zeros_like(x_hat); wgt=torch.zeros_like(x_hat)
    for s in range(0,len(indices),cp_eval_batch_size):
        chunk=indices[s:s+cp_eval_batch_size]; b=len(chunk)
        xs=torch.empty(b,cp_k,C,ps,ps,device=dev); pos=torch.empty(b,cp_k,2,ps,ps,device=dev); tok=torch.empty(b,cp_k,2,device=dev)
        for bi,z in enumerate(chunk):
            coords=[(z[0],z[2])]
            while len(coords)<cp_k: coords.append((random.randint(0,H-ps),random.randint(0,W-ps)))
            for k,(i,j) in enumerate(coords):
                xs[bi,k]=x_hat[0,:,i:i+ps,j:j+ps]; pos[bi,k]=latents_pos[0,:,i:i+ps,j:j+ps]; tok[bi,k]=_coord(i,j,ps,H,W,dev)
        t_b=t_hat.repeat(b) if t_hat.ndim==0 else t_hat.repeat(b)
        pred=net(xs,t_b.view(b,1,1,1,1),x_pos=pos,patch_coords=tok,class_labels=None)
        if cp_debug: assert pred.shape==(b,cp_k,C,ps,ps)
        for bi,z in enumerate(chunk):
            out[0,:,z[0]:z[1],z[2]:z[3]] += pred[bi,0]
            wgt[0,:,z[0]:z[1],z[2]:z[3]] += 1
    D=out/torch.clamp(wgt,min=1.0)
    if cp_debug: assert D.shape==x_hat.shape
    return D
