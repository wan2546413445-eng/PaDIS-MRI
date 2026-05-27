
import numpy as np, torch, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence
from torch.nn.functional import silu
from training.networks import Linear, Conv2d, GroupNorm, UNetBlock, PositionalEmbedding, FourierEmbedding, SongUNet

@persistence.persistent_class
class CrossPatchSongUNet(SongUNet):
    def __init__(self, *args, cp_num_heads=4, cp_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        bottleneck = [m.out_channels for m in self.enc.values() if hasattr(m,'out_channels')][-1]
        self.coord_proj=Linear(2,bottleneck)
        self.token_ln=torch.nn.LayerNorm(bottleneck)
        layer=torch.nn.TransformerEncoderLayer(d_model=bottleneck,nhead=cp_num_heads,dim_feedforward=bottleneck*4,batch_first=True)
        self.cross_attn=torch.nn.TransformerEncoder(layer,num_layers=cp_depth)
        self.film=Linear(bottleneck,bottleneck*2)
    def forward(self,x,noise_labels,patch_coords,class_labels=None,augment_labels=None):
        B,K,C,H,W=x.shape; x=x.reshape(B*K,C,H,W)
        emb=self.map_noise(noise_labels.reshape(B,1).repeat(1,K).reshape(B*K))
        emb=silu(self.map_layer0(emb)); emb=self.map_layer1(emb)
        skips=[]
        for block in self.enc.values():
            x=block(x,emb) if isinstance(block,UNetBlock) else block(x)
            skips.append(x)
        D=x.shape[1]; h,w=x.shape[-2:]
        tokens=x.view(B,K,D,h,w).mean((-1,-2)) + self.coord_proj(patch_coords.to(x.dtype))
        tokens=self.cross_attn(self.token_ln(tokens))
        gamma,beta=self.film(tokens.reshape(B*K,D)).chunk(2,1)
        x=x*(1+gamma.view(B*K,D,1,1))+beta.view(B*K,D,1,1)
        for block in self.dec.values():
            if isinstance(block,UNetBlock):
                x=block(torch.cat([x,skips.pop()],dim=1) if x.shape[1]!=block.in_channels else x,emb)
            else: x=block(x)
        return x.reshape(B,K,x.shape[1],x.shape[2],x.shape[3])

@persistence.persistent_class
class CrossPatch_EDMPrecond(torch.nn.Module):
    def __init__(self,img_resolution=64,data_channels=2,sigma_data=0.5,model_type='CrossPatchSongUNet',cp_num_heads=4,cp_depth=2,**model_kwargs):
        super().__init__(); self.img_resolution=img_resolution; self.data_channels=data_channels; self.sigma_data=sigma_data
        self.model=CrossPatchSongUNet(img_resolution=img_resolution,in_channels=data_channels+2,out_channels=data_channels,cp_num_heads=cp_num_heads,cp_depth=cp_depth,**model_kwargs)
    def forward(self,x,sigma,x_pos=None,patch_coords=None,class_labels=None,force_fp32=False,**kwargs):
        sigma=sigma.reshape(x.shape[0],1,1,1,1).to(x.dtype)
        c_skip=self.sigma_data**2/(sigma**2+self.sigma_data**2)
        c_out=sigma*self.sigma_data/(sigma**2+self.sigma_data**2).sqrt()
        c_in=1/(self.sigma_data**2+sigma**2).sqrt()
        c_noise=sigma.log().reshape(x.shape[0]) / 4
        x_in=torch.cat([c_in*x, x_pos.to(x.dtype)], dim=2)
        F=self.model(x_in,c_noise,patch_coords=patch_coords,class_labels=class_labels)
        return c_skip*x + c_out*F
    def round_sigma(self,sigma): return torch.as_tensor(sigma)
