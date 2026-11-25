"""Independent OPA+NGS loss. Same OPA reductions as main-WSY; no original edits."""
import torch
from torch_utils import persistence, training_stats
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import (_boundary_distances, _extract_patches,
    _overlap_views, _position_maps, _sample_overlap_geometry)

@persistence.persistent_class
class NGSOverlapLoss(Patch_EDMLoss):
    def __init__(self,P_mean=-1.2,P_std=1.2,sigma_data=0.5,lambda_overlap=1.,
                 active_patch_size=64,noise_mode='independent',alignment_mode='directional',
                 pair_geometry='shifted',spatial_weight_mode='distance'):
        super().__init__(P_mean=P_mean,P_std=P_std,sigma_data=sigma_data)
        if lambda_overlap < 0 or noise_mode not in ('same','independent'):
            raise ValueError('Invalid OPA noise mode / weight')
        if alignment_mode not in ('directional','symmetric') or pair_geometry not in ('shifted','identical') or spatial_weight_mode not in ('distance','uniform'):
            raise ValueError('Invalid OPA configuration')
        if pair_geometry == 'identical' and (spatial_weight_mode != 'uniform' or (lambda_overlap > 0 and alignment_mode != 'symmetric')):
            raise ValueError('Identical OPA requires uniform weights and symmetric alignment')
        self.lambda_overlap=float(lambda_overlap)
        self.active_patch_size=int(active_patch_size)
        self.noise_mode=noise_mode
        self.alignment_mode=alignment_mode
        self.pair_geometry=pair_geometry
        self.spatial_weight_mode=spatial_weight_mode

    def __call__(self,net,images,patch_size,resolution,labels=None,augment_pipe=None):
        if int(patch_size)!=self.active_patch_size:
            return super().__call__(net=net,images=images,patch_size=patch_size,resolution=resolution,labels=labels,augment_pipe=augment_pipe)
        if augment_pipe is not None:
            raise ValueError('NGS requires --augment=0')
        b=images.shape[0]; p=int(patch_size)
        t1,l1,t2,l2,horizontal,overlap=_sample_overlap_geometry(images,p)
        if self.pair_geometry=='identical':
            t2,l2=t1,l1
        clean1=_extract_patches(images,t1,l1,p)
        clean2=_extract_patches(images,t2,l2,p)
        pos1=_position_maps(t1,l1,b,p,resolution,images.device)
        pos2=_position_maps(t2,l2,b,p,resolution,images.device)
        sigma=(torch.randn([b,1,1,1],device=images.device)*self.P_std+self.P_mean).exp()
        weight=(sigma.square()+self.sigma_data**2)/(sigma*self.sigma_data).square()
        full1=images+sigma*torch.randn_like(images)
        full2=full1 if self.noise_mode=='same' else images+sigma*torch.randn_like(images)
        noisy1=_extract_patches(full1,t1,l1,p)
        noisy2=_extract_patches(full2,t2,l2,p)
        sigmas=torch.cat([sigma,sigma])
        coords=torch.cat([torch.stack([t1,l1],1),torch.stack([t2,l2],1)],0)
        pair=net(torch.cat([noisy1,noisy2]),sigmas,x_pos=torch.cat([pos1,pos2]),
                 class_labels=None if labels is None else torch.cat([labels,labels]),
                 augment_labels=None,full_noisy=torch.cat([full1,full2]),
                 full_noisy_sigma=sigmas,patch_coords=coords)
        d1,d2=pair[:b],pair[b:]
        edm=.5*(weight*(d1-clean1).square()+weight*(d2-clean2).square())
        training_stats.report('Loss/edm',edm.detach())
        if self.lambda_overlap==0:
            zero=edm.detach().new_zeros([])
            for key in ('raw','raw_ratio','weighted','ratio'):
                training_stats.report('Loss/overlap_center_'+key,zero)
            return edm
        if self.pair_geometry=='identical':
            interior,boundary=d1,d2
        else:
            o1,o2=_overlap_views(d1,d2,horizontal,overlap)
            dist1,dist2=_boundary_distances(horizontal,p,overlap,images.device,images.dtype)
            interior=torch.where(dist1>=dist2,o1,o2)
            boundary=torch.where(dist1>=dist2,o2,o1)
        if self.spatial_weight_mode=='uniform':
            confidence=torch.ones_like(interior[:,:1])
        else:
            confidence=(dist1-dist2).abs()
            confidence=confidence/confidence.mean(dim=(1,2,3),keepdim=True).clamp_min(1e-12)
        ref=interior.detach() if self.alignment_mode=='directional' else interior
        aux=(weight*confidence*(boundary-ref).square()).mean(dim=(1,2,3),keepdim=True)
        aux_map=aux.expand_as(edm)
        weighted=self.lambda_overlap*aux_map
        mean=edm.detach().mean().clamp_min(1e-12)
        for key,value in [('raw',aux.detach()),('raw_ratio',aux_map.detach().mean()/mean),('weighted',weighted.detach()),('ratio',weighted.detach().mean()/mean),('lambda',edm.detach().new_tensor(self.lambda_overlap))]:
            training_stats.report('Loss/overlap_center_'+key,value)
        return edm+weighted
